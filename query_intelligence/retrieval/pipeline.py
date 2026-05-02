from __future__ import annotations

import logging
import psycopg
import time
from datetime import date, timedelta
from pathlib import Path
from threading import Lock, Thread
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

from ..config import Settings
from ..data_loader import load_documents, load_structured_data
from ..integrations.akshare_macro_provider import AKShareMacroProvider
from ..integrations.akshare_market_provider import AKShareMarketProvider
from ..integrations.akshare_provider import AKShareNewsProvider
from ..integrations.cninfo_provider import CninfoAnnouncementProvider
from ..integrations.tushare_provider import TushareMarketProvider, TushareNewsProvider
from ..repositories.postgres_repository import PostgresDocumentRepository, PostgresStructuredRepository
from .api_retriever import APIRetriever
from .deduper import Deduper
from .doc_retriever import DocumentRetriever
from .feature_builder import FeatureBuilder
from .packager import RetrievalPackager
from .query_builder import QueryBuilder
from .ranker import BaselineRanker, HybridRanker
from .market_analyzer import MarketAnalyzer
from .selector import DocumentSelector
from .sql_retriever import SQLRetriever


class RetrievalPipeline:
    def __init__(
        self,
        query_builder: QueryBuilder,
        doc_retriever: DocumentRetriever,
        sql_retriever: SQLRetriever,
        api_retriever: APIRetriever,
        feature_builder: FeatureBuilder,
        ranker: BaselineRanker,
        deduper: Deduper,
        selector: DocumentSelector,
        packager: RetrievalPackager,
    ) -> None:
        self.query_builder = query_builder
        self.doc_retriever = doc_retriever
        self.sql_retriever = sql_retriever
        self.api_retriever = api_retriever
        self.feature_builder = feature_builder
        self.ranker = ranker
        self.deduper = deduper
        self.selector = selector
        self.packager = packager
        self.market_provider = None
        self.macro_provider = None
        self.news_providers: list = []
        self.announcement_provider = None
        self._announcement_retry_after_monotonic = 0.0
        self._announcement_cooldown_seconds = 300.0
        self._announcement_stuck_worker: Thread | None = None
        self._announcement_fetch_inflight = False
        self._announcement_breaker_lock = Lock()
        self.market_analyzer = MarketAnalyzer()

    @classmethod
    def build_demo(cls) -> "RetrievalPipeline":
        structured = load_structured_data()
        instance = cls(
            query_builder=QueryBuilder(),
            doc_retriever=DocumentRetriever(load_documents()),
            sql_retriever=SQLRetriever(structured),
            api_retriever=APIRetriever(structured),
            feature_builder=FeatureBuilder(),
            ranker=BaselineRanker(),
            deduper=Deduper(),
            selector=DocumentSelector(),
            packager=RetrievalPackager(),
        )
        return instance

    @classmethod
    def build_default(cls, settings: Settings) -> "RetrievalPipeline":
        if settings.use_postgres_retrieval and settings.postgres_dsn:
            connection = psycopg.connect(settings.postgres_dsn, row_factory=dict_row)
            text_retriever = PostgresDocumentRepository(connection)
            sql_retriever = PostgresStructuredRepository(connection)
        else:
            structured = load_structured_data()
            text_retriever = DocumentRetriever(load_documents())
            sql_retriever = SQLRetriever(structured)

        if settings.use_live_market and settings.tushare_token:
            market_provider = TushareMarketProvider.from_token(settings.tushare_token)
            api_retriever = APIRetriever(load_structured_data())
        elif settings.use_live_market:
            market_provider = AKShareMarketProvider.from_import(timeout=settings.request_timeout_seconds)
            api_retriever = APIRetriever(load_structured_data())
        else:
            structured = load_structured_data()
            market_provider = None
            api_retriever = APIRetriever(structured)

        ranker_path = Path(settings.models_dir) / "ranker.joblib"
        ranker = HybridRanker.load_model(ranker_path) if ranker_path.exists() else BaselineRanker()

        instance = cls(
            query_builder=QueryBuilder(),
            doc_retriever=text_retriever,
            sql_retriever=sql_retriever,
            api_retriever=api_retriever,
            feature_builder=FeatureBuilder(),
            ranker=ranker,
            deduper=Deduper(),
            selector=DocumentSelector(),
            packager=RetrievalPackager(),
        )
        if settings.use_live_market:
            instance.market_provider = market_provider
        if settings.use_live_macro:
            instance.macro_provider = AKShareMacroProvider.from_import()
        if settings.use_live_news:
            instance.news_providers.append(AKShareNewsProvider.from_import())
            if settings.tushare_token:
                instance.news_providers.append(TushareNewsProvider.from_token(settings.tushare_token))
        if settings.use_live_announcement:
            instance.announcement_provider = CninfoAnnouncementProvider(
                url=settings.cninfo_announcement_url,
                static_base=settings.cninfo_static_base,
                timeout=settings.request_timeout_seconds,
            )
        return instance

    def run(self, nlu_result: dict, top_k: int, debug: bool) -> dict:
        if "out_of_scope_query" in nlu_result.get("risk_flags", []):
            out_of_scope_nlu = {
                **nlu_result,
                "required_evidence_types": [],
                "source_plan": [],
            }
            return self.packager.build(out_of_scope_nlu, [], [], [], 0, [])
        if "missing_entity" in nlu_result.get("missing_slots", []) and not nlu_result.get("entities"):
            early_exit_nlu = {
                **nlu_result,
                "required_evidence_types": [],
                "source_plan": [],
            }
            return self.packager.build(early_exit_nlu, [], [], [], 0, [])

        query_bundle = self.query_builder.build(nlu_result)
        doc_candidates = self.doc_retriever.search(query_bundle, top_k=max(top_k * 2, 10))
        doc_candidates.extend(self._fetch_live_docs(query_bundle, top_k))
        total_candidates = len(doc_candidates)

        ranked_docs = []
        for candidate in doc_candidates:
            features = self.feature_builder.build(query_bundle, candidate)
            rank_score = self.ranker.score(features)
            ranked_docs.append(
                {
                    **candidate,
                    "feature_json": features,
                    "rank_score": rank_score,
                    "reason": [name for name, value in features.items() if isinstance(value, (int, float)) and value >= 0.8][:3],
                }
            )
        ranked_docs.sort(key=lambda item: item["rank_score"], reverse=True)
        selected_docs = self.selector.select(ranked_docs, query_bundle.get("source_plan", []), top_k)
        deduped_docs, groups = self.deduper.dedupe(selected_docs)

        structured_items = self._fetch_structured_items(query_bundle)
        structured_items = self._enrich_with_analysis(structured_items)
        analysis_summary = self.market_analyzer.build_analysis_summary(structured_items, nlu_result, deduped_docs)
        executed_sources = self._compute_executed_sources(deduped_docs, structured_items)
        return self.packager.build(nlu_result, deduped_docs, structured_items, groups, total_candidates, executed_sources, analysis_summary)

    def _enrich_with_analysis(self, structured_items: list[dict]) -> list[dict]:
        for item in structured_items:
            if item.get("source_type") in {"market_api", "index_daily"} and item.get("payload"):
                try:
                    self.market_analyzer.enrich_payload(item["payload"])
                except Exception as exc:
                    logger.warning("Market analyzer failed for %s: %s", item.get("evidence_id"), exc)
        return structured_items

    def _fetch_structured_items(self, query_bundle: dict) -> list[dict]:
        structured_items = self.api_retriever.fetch(query_bundle) + self.sql_retriever.fetch(query_bundle)
        structured_items = self._merge_live_macro_items(query_bundle, structured_items)
        requested_live_sources = self._requested_live_structured_sources(query_bundle)
        if not (self.market_provider and query_bundle.get("symbols") and requested_live_sources):
            return structured_items

        product_type = query_bundle.get("product_type", "stock")
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=365)).strftime("%Y%m%d")
        live_price_items: list[dict] = []
        live_fundamental_items: list[dict] = []
        live_industry_items: list[dict] = []
        live_fund_items: list[dict] = []
        live_index_items: list[dict] = []
        live_provider_warning_items: list[dict] = []
        live_symbols: set[str] = set()
        failed_live_market_symbols: set[str] = set()
        live_fundamental_symbols: set[str] = set()
        live_industry_names: set[str] = set()

        for symbol, canonical_name in self._iter_entity_targets(query_bundle):
            if not symbol:
                continue
            try:
                if isinstance(self.market_provider, TushareMarketProvider):
                    live_market = self.market_provider.fetch_bundle(symbol, start_date=start_date, end_date=end_date)
                else:
                    live_market = self.market_provider.fetch_bundle(
                        symbol=symbol,
                        canonical_name=canonical_name,
                        product_type=product_type,
                        start_date=start_date,
                        end_date=end_date,
                    )
            except Exception as exc:
                logger.warning("Market provider fetch failed for %s: %s", symbol, exc)
                failed_live_market_symbols.add(symbol)
                live_provider_warning_items.append(self._build_live_provider_warning_item(symbol, exc))
                continue

            live_symbols.add(symbol)
            if "market_api" in requested_live_sources:
                live_price_items.append(
                    {
                        "evidence_id": f"price_{symbol}",
                        "source_type": "market_api",
                        "source_name": live_market.get("source_name"),
                        "provider": live_market.get("source_name"),
                        "payload": live_market["payload"],
                    }
                )
            if live_market.get("fundamental_payload") and "fundamental_sql" in requested_live_sources:
                live_fundamental_symbols.add(symbol)
                fundamental_payload = {"symbol": symbol, **live_market["fundamental_payload"]}
                fundamental_source_name = fundamental_payload.get("source_name") or live_market.get("source_name")
                live_fundamental_items.append(
                    {
                        "evidence_id": f"fundamental_{symbol}",
                        "source_type": "fundamental_sql",
                        "source_name": fundamental_source_name,
                        "provider": fundamental_source_name,
                        "payload": fundamental_payload,
                    }
                )
            industry_snapshot = live_market["payload"].get("industry_snapshot")
            if industry_snapshot and "industry_sql" in requested_live_sources:
                industry_name = industry_snapshot["industry_name"]
                if industry_name not in live_industry_names:
                    live_industry_names.add(industry_name)
                    industry_source_name = industry_snapshot.get("source_name") or live_market.get("source_name")
                    live_industry_items.append(
                        {
                            "evidence_id": f"industry_{industry_name}",
                            "source_type": "industry_sql",
                            "source_name": industry_source_name,
                            "provider": industry_source_name,
                            "payload": industry_snapshot,
                        }
                    )
            requested_fund_sources = requested_live_sources.intersection({"fund_nav", "fund_fee", "fund_redemption", "fund_profile"})
            if product_type in {"etf", "fund"} and ("market_api" in requested_live_sources or requested_fund_sources):
                live_fund_items.extend(self._build_live_fund_items(symbol, live_market))
            requested_index_sources = requested_live_sources.intersection({"index_daily", "index_valuation"})
            if product_type == "index" and ("market_api" in requested_live_sources or requested_index_sources):
                live_index_items.extend(self._build_live_index_items(symbol, live_market))

        filtered_items = []
        for item in structured_items:
            payload = item.get("payload", {})
            if item["source_type"] == "market_api" and payload.get("symbol") in live_symbols:
                continue
            if item["source_type"] == "market_api" and payload.get("symbol") in failed_live_market_symbols:
                continue
            if item["source_type"] == "fundamental_sql" and payload.get("symbol") in live_fundamental_symbols:
                continue
            if item["source_type"] == "industry_sql" and payload.get("industry_name") in live_industry_names:
                continue
            filtered_items.append(item)
        return live_price_items + live_fund_items + live_index_items + live_provider_warning_items + filtered_items + live_fundamental_items + live_industry_items

    def _build_live_provider_warning_item(self, symbol: str, exc: Exception) -> dict:
        text = str(exc).strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        warning = f"market_provider_fetch_failed:{symbol}:{type(exc).__name__}:{text}"
        return {
            "evidence_id": f"market_provider_warning_{symbol}",
            "source_type": "provider_warning",
            "source_name": type(self.market_provider).__name__ if self.market_provider else "market_provider",
            "provider": type(self.market_provider).__name__ if self.market_provider else "market_provider",
            "payload": {
                "symbol": symbol,
                "provider_warnings": [warning],
            },
        }

    def _requested_live_structured_sources(self, query_bundle: dict) -> set[str]:
        requested = set(query_bundle.get("source_plan", [])).intersection(
            {
                "market_api",
                "fundamental_sql",
                "industry_sql",
                "fund_nav",
                "fund_fee",
                "fund_redemption",
                "fund_profile",
                "index_daily",
                "index_valuation",
            }
        )
        product_type = query_bundle.get("product_type")
        intent_labels = set(query_bundle.get("intent_labels", []))
        topic_labels = set(query_bundle.get("topic_labels", []))
        normalized_query = str(query_bundle.get("normalized_query", ""))
        if product_type in {"etf", "fund"} and (
            topic_labels.intersection({"product_mechanism", "comparison"})
            or intent_labels.intersection({"product_info", "trading_rule_fee", "peer_compare"})
            or any(term in normalized_query for term in ["费率", "申赎", "赎回", "净值", "定投", "机制"])
        ):
            requested.update({"fund_nav", "fund_fee", "fund_profile", "fund_redemption"})
        if product_type == "index":
            requested.update({"index_daily", "index_valuation"})
            if "market_api" in set(query_bundle.get("source_plan", [])):
                requested.add("market_api")
        return requested

    def _merge_live_macro_items(self, query_bundle: dict, structured_items: list[dict]) -> list[dict]:
        if "macro_sql" not in set(query_bundle.get("source_plan", [])) or not self.macro_provider:
            return structured_items
        try:
            live_items = self.macro_provider.fetch_indicators(query_bundle)
        except Exception as exc:
            logger.warning("Macro provider fetch failed: %s", exc)
            return structured_items
        if not live_items:
            return structured_items

        live_codes = {
            self._macro_indicator_code(item.get("payload", {}))
            for item in live_items
            if item.get("source_type") in {"macro_indicator", "policy_event"}
        }
        live_codes.discard("")
        filtered_items = [
            item
            for item in structured_items
            if not (item.get("source_type") == "macro_sql" and self._macro_indicator_code(item.get("payload", {})) in live_codes)
        ]
        return live_items + filtered_items

    def _macro_indicator_code(self, payload: dict) -> str:
        code = str(payload.get("indicator_code", "")).upper()
        if code.endswith("_CN"):
            code = code.removesuffix("_CN")
        if "10Y" in code:
            return "CN10Y"
        return code

    def _build_live_fund_items(self, symbol: str, live_market: dict) -> list[dict]:
        items = []
        for payload_key, source_type in (
            ("fund_nav_payload", "fund_nav"),
            ("fund_fee_payload", "fund_fee"),
            ("fund_redemption_payload", "fund_redemption"),
            ("fund_profile_payload", "fund_profile"),
        ):
            payload = live_market.get(payload_key)
            if not payload:
                continue
            source_name = payload.get("source_name") or live_market.get("source_name")
            items.append(
                {
                    "evidence_id": f"{source_type}_{symbol}",
                    "source_type": source_type,
                    "source_name": source_name,
                    "provider": payload.get("provider") or source_name,
                    "payload": {"symbol": symbol, **payload},
                }
            )
        return items

    def _build_live_index_items(self, symbol: str, live_market: dict) -> list[dict]:
        items = []
        for payload_key, source_type in (
            ("index_daily_payload", "index_daily"),
            ("index_valuation_payload", "index_valuation"),
        ):
            payload = live_market.get(payload_key)
            if not payload:
                continue
            source_name = payload.get("source_name") or live_market.get("source_name")
            items.append(
                {
                    "evidence_id": f"{source_type}_{symbol}",
                    "source_type": source_type,
                    "source_name": source_name,
                    "provider": payload.get("provider") or source_name,
                    "payload": {"symbol": symbol, **payload},
                }
            )
        return items

    def _fetch_live_docs(self, query_bundle: dict, top_k: int) -> list[dict]:
        docs: list[dict] = []
        entity_targets = list(self._iter_entity_targets(query_bundle))
        if self.news_providers and "news" in query_bundle.get("source_plan", []) and entity_targets:
            for provider in self.news_providers:
                for symbol, canonical_name in entity_targets:
                    try:
                        docs.extend(
                            provider.fetch_news(
                                symbol=symbol or canonical_name,
                                canonical_name=canonical_name or symbol,
                                limit=min(top_k, 10),
                            )
                        )
                    except Exception:
                        logger.warning("News provider %s failed for %s", type(provider).__name__, symbol or canonical_name, exc_info=True)
        if (
            self.announcement_provider
            and "announcement" in query_bundle.get("source_plan", [])
        ):
            should_fetch_announcements = True
            cooldown_elapsed = False
            retry_after_seconds = 0.0
            with self._announcement_breaker_lock:
                now = time.monotonic()
                if self._announcement_stuck_worker and not self._announcement_stuck_worker.is_alive():
                    self._announcement_stuck_worker = None
                if self._announcement_stuck_worker and self._announcement_stuck_worker.is_alive():
                    should_fetch_announcements = False
                    logger.warning("Announcement provider previous worker still running, skipping announcement fetch")
                elif self._announcement_fetch_inflight:
                    should_fetch_announcements = False
                    logger.warning("Announcement provider fetch already in-flight, skipping announcement fetch")
                elif self._announcement_retry_after_monotonic > now:
                    should_fetch_announcements = False
                    retry_after_seconds = round(self._announcement_retry_after_monotonic - now, 1)
                    logger.warning("Announcement provider is in cooldown (retry in %ss), skipping announcement fetch", retry_after_seconds)
                else:
                    cooldown_elapsed = self._announcement_retry_after_monotonic > 0
                    self._announcement_retry_after_monotonic = 0.0
                    self._announcement_fetch_inflight = True

            if cooldown_elapsed:
                logger.info("Announcement provider cooldown elapsed, retrying announcement fetch")

            if should_fetch_announcements:
                try:
                    for symbol, _ in entity_targets:
                        if not symbol:
                            continue
                        ann_docs: list[dict] = []
                        worker_error: Exception | None = None
                        wait_timeout = getattr(self.announcement_provider, "timeout", 15) + 5

                        def _fetch() -> None:
                            nonlocal ann_docs, worker_error
                            try:
                                ann_docs = self.announcement_provider.fetch_announcements(
                                    symbol,
                                    limit=min(top_k, 10),
                                )
                            except Exception as exc:  # noqa: BLE001
                                worker_error = exc

                        worker = Thread(target=_fetch, daemon=True)
                        worker.start()
                        worker.join(timeout=wait_timeout)

                        if worker.is_alive():
                            with self._announcement_breaker_lock:
                                self._announcement_stuck_worker = worker
                                self._announcement_retry_after_monotonic = time.monotonic() + self._announcement_cooldown_seconds
                            logger.warning(
                                "Announcement provider timed out (%ss) for %s, entering cooldown for %ss",
                                wait_timeout,
                                symbol,
                                self._announcement_cooldown_seconds,
                            )
                            break
                        if worker_error is not None:
                            logger.warning("Announcement provider failed for %s: %s", symbol, worker_error)
                            continue
                        docs.extend(ann_docs)
                finally:
                    with self._announcement_breaker_lock:
                        self._announcement_fetch_inflight = False
        for doc in docs:
            doc.setdefault("retrieval_score", 0.5)
        return docs

    def _iter_entity_targets(self, query_bundle: dict):
        entity_names = query_bundle.get("entity_names", [])
        symbols = query_bundle.get("symbols", [])
        count = max(len(entity_names), len(symbols))
        for index in range(count):
            symbol = symbols[index] if index < len(symbols) else ""
            canonical_name = entity_names[index] if index < len(entity_names) else symbol
            if symbol or canonical_name:
                yield symbol, canonical_name

    def _compute_executed_sources(self, deduped_docs: list[dict], structured_items: list[dict]) -> list[str]:
        executed_sources: list[str] = []
        for item in [*deduped_docs, *structured_items]:
            source_type = item.get("source_type")
            if source_type and source_type not in executed_sources:
                executed_sources.append(source_type)
        return executed_sources
