from __future__ import annotations

import logging
import re
from pathlib import Path
from uuid import uuid4

from joblib import load

from ..config import Settings
from ..data_loader import load_aliases, load_entities, load_seed_aliases, load_seed_entities, load_synonyms
from ..query_terms import contains_bucket_market_term
from ..training_data import filter_rows_for_label, load_training_rows
from .classifiers import INTENT_SAMPLES, TOPIC_SAMPLES, MultiLabelClassifier, ProductTypeClassifier, SingleLabelTextClassifier
from .clarification_gate import ClarificationGate
from .entity_boundary_crf import EntityBoundaryCRF
from .entity_linker import EntityLinker
from .entity_resolver import EntityResolver
from .normalizer import QueryNormalizer
from .out_of_scope_detector import OutOfScopeDetector
from .question_style_reranker import QuestionStyleReranker
from .slot_extractor import SlotExtractor
from .source_planner import (
    MLSourcePlanner,
    SourcePlanner,
    looks_like_company_fundamental_query,
    looks_like_disclosure_query,
    looks_like_general_finance_query,
)
from .source_plan_reranker import SourcePlanReranker
from .typo_linker import TypoLinker

GENERIC_PRODUCT_TARGETS = {
    "etf",
    "lof",
    "基金",
    "公募基金",
    "场内基金",
    "场外基金",
    "指数基金",
    "指数",
    "债券基金",
    "偏股混合基金",
    "沪深300指数基金",
    "红利低波etf",
    "证券etf",
    "创业板etf",
    "中证a500etf",
}

logger = logging.getLogger(__name__)


class NLUPipeline:
    def __init__(
        self,
        normalizer: QueryNormalizer,
        entity_resolver: EntityResolver,
        product_classifier: ProductTypeClassifier,
        intent_classifier: MultiLabelClassifier,
        topic_classifier: MultiLabelClassifier,
        slot_extractor: SlotExtractor,
        source_planner: SourcePlanner,
        question_style_classifier: SingleLabelTextClassifier | None = None,
        question_style_reranker: QuestionStyleReranker | None = None,
        sentiment_classifier: SingleLabelTextClassifier | None = None,
        clarification_gate: ClarificationGate | None = None,
        out_of_scope_detector: OutOfScopeDetector | None = None,
    ) -> None:
        self.normalizer = normalizer
        self.entity_resolver = entity_resolver
        self.product_classifier = product_classifier
        self.intent_classifier = intent_classifier
        self.topic_classifier = topic_classifier
        self.slot_extractor = slot_extractor
        self.source_planner = source_planner
        self.question_style_classifier = question_style_classifier
        self.question_style_reranker = question_style_reranker
        self.sentiment_classifier = sentiment_classifier
        self.clarification_gate = clarification_gate
        self.out_of_scope_detector = out_of_scope_detector

    @classmethod
    def build_demo(cls) -> "NLUPipeline":
        synonyms = load_synonyms()
        entities = load_seed_entities()
        aliases = load_seed_aliases()
        default_records_path = Path("data/query_labels.csv")
        boundary_model = EntityBoundaryCRF.build_from_queries([text for text, _ in INTENT_SAMPLES + [(q, []) for q, _ in TOPIC_SAMPLES]], [row["normalized_alias"] for row in aliases])
        typo_linker = TypoLinker.build_from_aliases(aliases)
        return cls(
            normalizer=QueryNormalizer(synonyms),
            entity_resolver=EntityResolver(entities, aliases, linker=EntityLinker.build_from_catalog(entities, aliases), boundary_model=boundary_model, typo_linker=typo_linker),
            product_classifier=ProductTypeClassifier.build_demo(),
            intent_classifier=MultiLabelClassifier.build_demo(INTENT_SAMPLES),
            topic_classifier=MultiLabelClassifier.build_demo(TOPIC_SAMPLES),
            slot_extractor=SlotExtractor(synonyms),
            source_planner=SourcePlanner(),
            question_style_classifier=None,
            question_style_reranker=QuestionStyleReranker.build_from_dataset(default_records_path) if default_records_path.exists() else None,
            sentiment_classifier=None,
            clarification_gate=ClarificationGate.build_from_dataset(default_records_path) if default_records_path.exists() else None,
            out_of_scope_detector=OutOfScopeDetector.build_from_dataset(default_records_path) if default_records_path.exists() else None,
        )

    @classmethod
    def build_default(cls, settings: Settings) -> "NLUPipeline":
        synonyms = load_synonyms()
        product_model_path = Path(settings.models_dir) / "product_type.joblib"
        intent_model_path = Path(settings.models_dir) / "intent_ovr.joblib"
        topic_model_path = Path(settings.models_dir) / "topic_ovr.joblib"
        question_style_model_path = Path(settings.models_dir) / "question_style.joblib"
        question_style_reranker_model_path = Path(settings.models_dir) / "question_style_reranker.joblib"
        sentiment_model_path = Path(settings.models_dir) / "sentiment.joblib"
        entity_crf_model_path = Path(settings.models_dir) / "entity_crf.joblib"
        clarification_model_path = Path(settings.models_dir) / "clarification_gate.joblib"
        typo_linker_model_path = Path(settings.models_dir) / "typo_linker.joblib"
        source_plan_reranker_model_path = Path(settings.models_dir) / "source_plan_reranker.joblib"
        out_of_scope_model_path = Path(settings.models_dir) / "out_of_scope_detector.joblib"
        training_manifest_path = Path(settings.training_manifest_path) if settings.training_manifest_path else None
        training_manifest_exists = training_manifest_path.exists() if training_manifest_path else False

        if training_manifest_exists:
            records = load_training_rows(training_manifest_path)
            entities = load_entities()
            aliases = load_aliases()
            product_rows = filter_rows_for_label(records, "product_type")
            intent_rows = filter_rows_for_label(records, "intent_labels")
            topic_rows = filter_rows_for_label(records, "topic_labels")
            question_style_rows = filter_rows_for_label(records, "question_style")
            sentiment_rows = filter_rows_for_label(records, "sentiment_label")
            product_classifier = ProductTypeClassifier.build_from_records(product_rows or records)
            intent_classifier = MultiLabelClassifier.build_from_records(intent_rows or records, "intent_labels")
            topic_classifier = MultiLabelClassifier.build_from_records(topic_rows or records, "topic_labels")
            source_planner = SourcePlanner(
                ml_planner=MLSourcePlanner.build_from_records(records),
                source_plan_reranker=SourcePlanReranker.build_from_dataset(training_manifest_path),
            )
            question_style_classifier = SingleLabelTextClassifier.build_from_records(question_style_rows or records, "question_style")
            question_style_reranker = QuestionStyleReranker.build_from_dataset(training_manifest_path)
            sentiment_classifier = SingleLabelTextClassifier.build_from_records(sentiment_rows or records, "sentiment_label")
            boundary_model = EntityBoundaryCRF.build_from_queries([record["query"] for record in records], [row["normalized_alias"] for row in aliases])
            clarification_gate = ClarificationGate.build_from_dataset(training_manifest_path)
            typo_linker = TypoLinker.build_from_aliases(aliases)
            out_of_scope_detector = OutOfScopeDetector.build_from_dataset(training_manifest_path)
        elif settings.training_dataset_path and Path(settings.training_dataset_path).exists():
            records = load_training_rows(settings.training_dataset_path)
            entities = load_entities()
            aliases = load_aliases()
            product_rows = filter_rows_for_label(records, "product_type")
            intent_rows = filter_rows_for_label(records, "intent_labels")
            topic_rows = filter_rows_for_label(records, "topic_labels")
            question_style_rows = filter_rows_for_label(records, "question_style")
            sentiment_rows = filter_rows_for_label(records, "sentiment_label")
            product_classifier = ProductTypeClassifier.build_from_records(product_rows or records)
            intent_classifier = MultiLabelClassifier.build_from_records(intent_rows or records, "intent_labels")
            topic_classifier = MultiLabelClassifier.build_from_records(topic_rows or records, "topic_labels")
            source_planner = SourcePlanner(
                ml_planner=MLSourcePlanner.build_from_records(records),
                source_plan_reranker=SourcePlanReranker.build_from_dataset(settings.training_dataset_path),
            )
            question_style_classifier = SingleLabelTextClassifier.build_from_records(question_style_rows or records, "question_style")
            question_style_reranker = QuestionStyleReranker.build_from_dataset(settings.training_dataset_path)
            sentiment_classifier = SingleLabelTextClassifier.build_from_records(sentiment_rows or records, "sentiment_label")
            boundary_model = EntityBoundaryCRF.build_from_queries([record["query"] for record in records], [row["normalized_alias"] for row in aliases])
            clarification_gate = ClarificationGate.build_from_dataset(settings.training_dataset_path)
            typo_linker = TypoLinker.build_from_aliases(aliases)
            out_of_scope_detector = OutOfScopeDetector.build_from_dataset(settings.training_dataset_path)
        else:
            default_records_path = Path("data/query_labels.csv")
            default_records = load_training_rows(default_records_path) if default_records_path.exists() else None
            entities = load_entities()
            aliases = load_aliases()
            product_classifier = ProductTypeClassifier.load_model(product_model_path) if product_model_path.exists() else ProductTypeClassifier.build_demo()
            intent_classifier = MultiLabelClassifier.load_model(intent_model_path) if intent_model_path.exists() else MultiLabelClassifier.build_demo(INTENT_SAMPLES)
            topic_classifier = MultiLabelClassifier.load_model(topic_model_path) if topic_model_path.exists() else MultiLabelClassifier.build_demo(TOPIC_SAMPLES)
            ml_planner = MLSourcePlanner.build_from_records(default_records) if default_records else None
            source_plan_reranker = (
                cls._load_optional_model(SourcePlanReranker.load_model, source_plan_reranker_model_path, "source_plan_reranker")
                if source_plan_reranker_model_path.exists()
                else SourcePlanReranker.build_from_dataset(default_records_path)
                if default_records
                else None
            )
            source_planner = SourcePlanner(
                ml_planner=ml_planner,
                source_plan_reranker=source_plan_reranker,
            )
            question_style_classifier = (
                SingleLabelTextClassifier.load_model(question_style_model_path)
                if question_style_model_path.exists()
                else SingleLabelTextClassifier.build_from_records(default_records, "question_style")
                if default_records
                else None
            )
            question_style_reranker = (
                cls._load_optional_model(QuestionStyleReranker.load_model, question_style_reranker_model_path, "question_style_reranker")
                if question_style_reranker_model_path.exists()
                else QuestionStyleReranker.build_from_dataset(default_records_path)
                if default_records
                else None
            )
            sentiment_classifier = (
                cls._load_optional_model(SingleLabelTextClassifier.load_model, sentiment_model_path, "sentiment")
                if sentiment_model_path.exists()
                else SingleLabelTextClassifier.build_from_records(default_records, "sentiment_label")
                if default_records
                else None
            )
            boundary_model = (
                cls._load_optional_model(load, entity_crf_model_path, "entity_crf")
                if entity_crf_model_path.exists()
                else EntityBoundaryCRF.build_from_queries([record["query"] for record in default_records], [row["normalized_alias"] for row in aliases])
                if default_records
                else None
            )
            clarification_gate = (
                cls._load_optional_model(ClarificationGate.load_model, clarification_model_path, "clarification_gate")
                if clarification_model_path.exists()
                else ClarificationGate.build_from_dataset(default_records_path)
                if default_records
                else None
            )
            typo_linker = (
                TypoLinker.load_model(str(typo_linker_model_path))
                if typo_linker_model_path.exists()
                else TypoLinker.build_from_aliases(aliases)
            )
            out_of_scope_detector = (
                cls._load_optional_model(OutOfScopeDetector.load_model, out_of_scope_model_path, "out_of_scope_detector")
                if out_of_scope_model_path.exists()
                else OutOfScopeDetector.build_from_dataset(default_records_path)
                if default_records
                else None
            )

        return cls(
            normalizer=QueryNormalizer(synonyms),
            entity_resolver=EntityResolver(entities, aliases, linker=EntityLinker.build_from_catalog(entities, aliases), boundary_model=boundary_model, typo_linker=typo_linker),
            product_classifier=product_classifier,
            intent_classifier=intent_classifier,
            topic_classifier=topic_classifier,
            slot_extractor=SlotExtractor(synonyms),
            source_planner=source_planner,
            question_style_classifier=question_style_classifier,
            question_style_reranker=question_style_reranker,
            sentiment_classifier=sentiment_classifier,
            clarification_gate=clarification_gate,
            out_of_scope_detector=out_of_scope_detector,
        )

    @staticmethod
    def _load_optional_model(loader, path: Path, model_name: str):
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Optional NLU model %s failed to load from %s and will be disabled: %s", model_name, path, exc)
            return None

    def run(self, query: str, user_profile: dict, dialog_context: list, debug: bool) -> dict:
        normalized_query, normalization_trace = self.normalizer.normalize(query)
        time_scope = self.normalizer.detect_time_scope(normalized_query)
        operation = self.normalizer.detect_operation(normalized_query)

        precheck_entities, precheck_comparison_targets, precheck_trace = self.entity_resolver.resolve_exact(normalized_query)
        out_of_scope_score = self.out_of_scope_detector.predict_probability(query) if self.out_of_scope_detector else None
        if out_of_scope_score is not None and self._should_early_reject_out_of_scope(query, operation, out_of_scope_score, precheck_entities):
            return self._build_out_of_scope_result(query, normalized_query, out_of_scope_score)

        if precheck_entities:
            entities = precheck_entities
            comparison_targets = self._normalize_comparison_targets(precheck_comparison_targets)
            entity_trace = precheck_trace
        elif contains_bucket_market_term(normalized_query):
            entities = []
            comparison_targets = self._normalize_comparison_targets(precheck_comparison_targets)
            entity_trace = precheck_trace
        else:
            entities, comparison_targets, entity_trace = self.entity_resolver.resolve(normalized_query)
            comparison_targets = self._normalize_comparison_targets(comparison_targets)
        if not entities:
            context_entities, context_trace = self._resolve_context_entities(user_profile, dialog_context)
            if context_entities:
                entities = context_entities
                entity_trace.extend(context_trace)
        semantic_query = normalized_query
        product = self.product_classifier.predict(normalized_query, entities)
        product = self._postprocess_product_prediction(semantic_query, product)
        if self._has_explicit_generic_product_compare_signal(semantic_query, comparison_targets) or self._is_generic_product_compare_query(semantic_query, comparison_targets, product["label"]):
            product["label"] = "etf" if any(term in semantic_query for term in ["ETF", "etf", "LOF", "lof"]) else "fund"
            product["score"] = max(product["score"], 0.95)
        intents = self._apply_intent_rules(semantic_query, self.intent_classifier.predict(normalized_query))
        topics = self._apply_topic_rules(semantic_query, self.topic_classifier.predict(normalized_query))
        intents = self._suppress_intent_noise(semantic_query, intents)
        topics = self._suppress_topic_noise(semantic_query, topics)
        intents = self._suppress_compare_intent_noise(semantic_query, intents, entities, comparison_targets, product["label"])
        topics = self._suppress_compare_topic_noise(semantic_query, topics, entities, comparison_targets, product["label"])
        slot_info = self.slot_extractor.extract(semantic_query, normalized_query, resolved_entities=entities)
        question_style = self.question_style_classifier.predict(normalized_query)["label"] if self.question_style_classifier else self.slot_extractor.detect_question_style(semantic_query, intents)
        question_style = self._postprocess_question_style(semantic_query, question_style, intents, comparison_targets)
        question_style, question_style_score = self._apply_question_style_reranker(
            query=semantic_query,
            question_style=question_style,
            product_type=product["label"],
            intents=intents,
            topics=topics,
            entities=entities,
            comparison_targets=comparison_targets,
        )
        question_style = self._postprocess_question_style(semantic_query, question_style, intents, comparison_targets)
        if (
            question_style != "advice"
            and (
                self._has_explicit_generic_product_compare_signal(semantic_query, comparison_targets)
                or self._is_generic_product_compare_query(semantic_query, comparison_targets, product["label"])
            )
        ):
            question_style = "compare"
        requires_entity_clarification = self._requires_entity_clarification(
            semantic_query,
            product["label"],
            entities,
            intents,
            topics,
            comparison_targets,
            slot_info["missing_slots"],
        )
        requires_entity_clarification, clarification_score = self._apply_clarification_gate(
            query=semantic_query,
            product_type=product["label"],
            intents=intents,
            topics=topics,
            entities=entities,
            comparison_targets=comparison_targets,
            time_scope=time_scope,
            base_requires_clarification=requires_entity_clarification,
        )
        if requires_entity_clarification and "missing_entity" not in slot_info["missing_slots"]:
            slot_info["missing_slots"].append("missing_entity")
        if not requires_entity_clarification and "missing_entity" in slot_info["missing_slots"]:
            slot_info["missing_slots"] = [slot for slot in slot_info["missing_slots"] if slot != "missing_entity"]
        plan = self.source_planner.plan(
            normalized_query,
            product["label"],
            intents,
            topics,
            time_scope,
            missing_slots=slot_info["missing_slots"] if requires_entity_clarification else [],
        )
        raw_sentiment = self.sentiment_classifier.predict(normalized_query)["label"] if self.sentiment_classifier else slot_info["sentiment_of_user"]
        sentiment = self._normalize_sentiment_label(raw_sentiment)

        entity_not_found_is_material = not entities and "missing_entity" in slot_info["missing_slots"]
        risk_flags = []
        if "entity_ambiguous" in entity_trace:
            risk_flags.append("entity_ambiguous")
        if entity_not_found_is_material and "entity_not_found" in entity_trace:
            risk_flags.append("entity_not_found")
        if question_style in {"advice", "forecast"}:
            risk_flags.append("investment_advice_like")
        if requires_entity_clarification:
            risk_flags.append("clarification_required")

        explainability = {
            "matched_rules": normalization_trace + [item for item in entity_trace if ":" in item] + ([f"question_style_ml:{question_style_score:.2f}"] if question_style_score is not None else []) + ([f"clarification_ml:{clarification_score:.2f}"] if clarification_score is not None else []),
            "top_features": [item["label"] for item in intents[:2]] + slot_info["keywords"][:2],
        }

        confidence_parts = [product["score"]]
        confidence_parts.extend(item["score"] for item in intents[:2])
        confidence = round(sum(confidence_parts) / max(len(confidence_parts), 1), 2)

        return {
            "query_id": str(uuid4()),
            "raw_query": query,
            "normalized_query": normalized_query,
            "question_style": question_style,
            "product_type": {"label": product["label"], "score": product["score"]},
            "intent_labels": intents,
            "topic_labels": topics,
            "entities": entities,
            "comparison_targets": comparison_targets,
            "keywords": slot_info["keywords"],
            "time_scope": time_scope,
            "forecast_horizon": slot_info["forecast_horizon"],
            "sentiment_of_user": sentiment,
            "operation_preference": operation,
            "required_evidence_types": plan["required_evidence_types"],
            "source_plan": plan["source_plan"],
            "risk_flags": risk_flags,
            "missing_slots": slot_info["missing_slots"],
            "confidence": confidence,
            "explainability": explainability,
        }

    def _build_out_of_scope_result(self, query: str, normalized_query: str, out_of_scope_score: float) -> dict:
        return {
            "query_id": str(uuid4()),
            "raw_query": query,
            "normalized_query": normalized_query,
            "question_style": "fact",
            "product_type": {"label": "out_of_scope", "score": round(out_of_scope_score, 2)},
            "intent_labels": [],
            "topic_labels": [],
            "entities": [],
            "comparison_targets": [],
            "keywords": [],
            "time_scope": "unspecified",
            "forecast_horizon": "short_term",
            "sentiment_of_user": "neutral",
            "operation_preference": "unknown",
            "required_evidence_types": [],
            "source_plan": [],
            "risk_flags": ["out_of_scope_query"],
            "missing_slots": [],
            "confidence": round(out_of_scope_score, 2),
            "explainability": {
                "matched_rules": [f"out_of_scope_ml:{out_of_scope_score:.2f}"],
                "top_features": [],
            },
        }

    def _has_finance_action_signal(self, query: str, operation: str) -> bool:
        zh_action_markers = [
            "能买吗",
            "可以买",
            "买入",
            "卖出",
            "止盈",
            "止损",
            "加仓",
            "减仓",
            "补仓",
            "抄底",
            "仓位",
            "持有",
            "拿吗",
            "值得拿",
            "还值得",
            "定投",
            "股价",
            "行情",
            "估值",
            "财报",
            "基金",
            "上证50",
            "沪深300",
            "中证500",
            "中证1000",
            "创业板指",
            "科创50",
            "ETF",
            "LOF",
            "指数",
            "新股",
            "申赎",
            "费率",
            "研报",
            "强推",
            "市盈率",
            "GDP",
            "CPI",
            "PPI",
            "美联储议息",
            "场内",
            "场外",
            "A股",
            "港股",
            "美股",
            "公告",
            "新闻",
        ]
        english_action_phrases = [
            "should i buy",
            "should i sell",
            "should i hold",
            "still worth buying",
            "worth holding",
            "would you still hold",
            "cut the loss",
            "stop loss",
            "take profit",
            "average down",
            "add to position",
            "trim position",
        ]
        english_finance_context = [
            "share price",
            "stock price",
            "stock market",
            "mutual fund",
            "index fund",
            "bond fund",
            "net profit",
        ]
        english_finance_tokens = [
            "etf",
            "lof",
            "stock",
            "stocks",
            "share",
            "shares",
            "bond",
            "bonds",
            "equity",
            "equities",
            "earnings",
            "valuation",
            "dividend",
            "dividends",
            "portfolio",
            "position",
            "invest",
            "investment",
            "investing",
            "gold",
            "hedge",
            "inflation",
            "trading",
            "trader",
            "broker",
            "brokerage",
            "options",
            "option",
            "futures",
            "forex",
            "cash",
            "credit",
            "mortgage",
            "loan",
            "tax",
            "taxes",
            "insurance",
            "ira",
            "401k",
            "bank",
            "account",
            "debt",
            "budget",
        ]
        lowered = query.lower()
        if looks_like_company_fundamental_query(query) or looks_like_disclosure_query(query):
            return True
        if looks_like_general_finance_query(query):
            return True
        if contains_bucket_market_term(query):
            return True
        if any(marker in query for marker in zh_action_markers):
            return True
        if any(phrase in lowered for phrase in english_action_phrases):
            return True
        if operation != "unknown" and (
            any(phrase in lowered for phrase in english_finance_context) or self._contains_english_token(lowered, english_finance_tokens)
        ):
            return True
        return False

    @staticmethod
    def _contains_english_token(lowered_query: str, tokens: list[str]) -> bool:
        escaped = "|".join(re.escape(token) for token in tokens)
        return bool(re.search(rf"\b(?:{escaped})\b", lowered_query))

    def _has_strong_finance_anchor(self, query: str) -> bool:
        lowered = query.lower()
        zh_anchors = [
            "股票",
            "基金",
            "指数",
            "a股",
            "港股",
            "美股",
            "财报",
            "估值",
            "营收",
            "净利润",
            "公告",
            "研报",
            "强推",
            "市盈率",
            "市净率",
            "费率",
            "申赎",
            "新股",
            "gdp",
            "cpi",
            "ppi",
            "美联储议息",
            "央行",
            "金融业",
            "定投",
            "场内",
            "场外",
            "止损",
            "止盈",
            "回撤",
            "波动",
            "仓位",
            "板块",
            "白酒板块",
            "券商板块",
            "宏观",
            "降息",
            "降准",
            "贵州茅台",
            "五粮液",
            "中国平安",
            "平安银行",
            "中国太保",
            "中国人寿",
            "中国人保",
            "新华保险",
            "上证50",
            "沪深300",
            "中证500",
            "中证1000",
            "创业板指",
            "科创50",
            "太保寿险",
            "人保寿险",
            "上市险企",
            "保险公司",
        ]
        english_phrases = [
            "share price",
            "stock price",
            "stock market",
            "mutual fund",
            "index fund",
            "bond fund",
            "net profit",
            "ping an",
        ]
        english_tokens = [
            "etf",
            "lof",
            "stock",
            "stocks",
            "share",
            "shares",
            "fund",
            "funds",
            "bond",
            "bonds",
            "equity",
            "equities",
            "earnings",
            "valuation",
            "dividend",
            "dividends",
            "portfolio",
            "cpi",
            "pmi",
            "gdp",
            "moutai",
            "wuliangye",
            "pingan",
            "fundamental",
            "fundamentals",
            "invest",
            "investment",
            "investing",
            "gold",
            "hedge",
            "inflation",
            "trading",
            "trader",
            "broker",
            "brokerage",
            "options",
            "option",
            "futures",
            "forex",
            "credit",
            "mortgage",
            "loan",
            "tax",
            "taxes",
            "insurance",
            "ira",
            "401k",
            "bank",
            "account",
            "debt",
            "budget",
        ]
        return (
            looks_like_company_fundamental_query(query)
            or looks_like_disclosure_query(query)
            or
            looks_like_general_finance_query(query)
            or any(anchor in lowered for anchor in zh_anchors)
            or any(phrase in lowered for phrase in english_phrases)
            or self._contains_english_token(lowered, english_tokens)
        )

    def _looks_like_general_utility_query(self, query: str) -> bool:
        lowered = query.lower()
        utility_markers = [
            "translate",
            "翻译",
            "weather",
            "天气",
            "python",
            "sql",
            "电脑",
            "笔记本",
            "台式机",
            "suv",
            "mpv",
            "酒店",
            "餐厅",
            "景点",
            "医院",
            "人均消费",
            "电话",
            "预算",
            "推荐一辆",
            "推荐一款",
            "写个",
            "joke",
            "nba",
        ]
        return any(marker in lowered for marker in utility_markers)

    def _should_early_reject_out_of_scope(self, query: str, operation: str, out_of_scope_score: float, resolved_entities: list[dict] | None = None) -> bool:
        if out_of_scope_score < 0.65:
            return False
        if any(entity.get("entity_type") in {"stock", "etf", "fund", "index"} for entity in resolved_entities or []):
            return False
        if self._looks_like_unresolved_listed_company_target_query(query):
            return False
        if self._looks_like_unresolved_finance_target_query(query):
            return False
        if self._is_underspecified_investment_query(query):
            return False
        if self._has_strong_finance_anchor(query):
            return False
        if self._has_finance_action_signal(query, operation):
            return False
        if self._looks_like_general_utility_query(query):
            return True
        return True

    def _looks_like_unresolved_listed_company_target_query(self, query: str) -> bool:
        if self._looks_like_general_utility_query(query):
            return False
        target_question_markers = [
            "你觉得",
            "怎么看",
            "怎么样",
            "如何看",
            "能买吗",
            "可以买",
            "值得买吗",
            "值得入手",
            "值得拿",
            "还值得",
            "还能拿",
            "适合拿",
            "哪个好",
            "哪个更好",
            "走势",
            "后面",
            "前景",
        ]
        if not any(marker in query for marker in target_question_markers):
            return False
        listed_company_context = [
            "科技",
            "汽车",
            "股份",
            "集团",
            "银行",
            "保险",
            "证券",
            "医药",
            "生物",
            "半导体",
            "新能源",
            "电子",
            "电力",
            "能源",
            "控股",
        ]
        return any(marker in query for marker in listed_company_context)

    def _looks_like_unresolved_finance_target_query(self, query: str) -> bool:
        lowered = query.lower()
        if re.search(r"(?<!\d)[034568]\d{5}(?:\.(?:sh|sz|bj))?(?!\d)", lowered):
            return True
        target_question_markers = [
            "你觉得",
            "怎么看",
            "怎么样",
            "如何看",
            "能买吗",
            "可以买",
            "值得买吗",
            "值得拿",
            "还值得",
            "还能拿",
            "适合拿",
            "哪个好",
            "哪个更好",
            "走势",
            "后面",
            "前景",
        ]
        if not any(marker in query for marker in target_question_markers):
            return False
        finance_context_markers = [
            "股票",
            "股价",
            "行情",
            "财报",
            "估值",
            "基本面",
            "公告",
            "研报",
            "基金",
            "ETF",
            "etf",
            "LOF",
            "lof",
            "指数",
        ]
        if any(marker in query for marker in finance_context_markers):
            return True
        return False

    def _is_underspecified_investment_query(self, query: str) -> bool:
        referential_markers = ["这个", "它", "这只", "这一只", "那只", "这支", "那支", "这只票", "那只票"]
        action_markers = [
            "能买吗",
            "可以买",
            "值不值得",
            "还值得",
            "还适合拿",
            "还值得拿",
            "持有",
            "卖吗",
            "买吗",
            "拿吗",
            "还能拿吗",
            "止损",
            "止盈",
            "加仓",
            "减仓",
            "补仓",
            "适合定投",
            "继续定投",
            "适合继续定投",
        ]
        lowered = query.lower()
        english_referential = any(marker in lowered for marker in ["this", "it"])
        english_action = any(
            marker in lowered
            for marker in [
                "should i buy",
                "should i sell",
                "still worth buying",
                "would you still hold",
                "cut the loss",
                "should i cut the loss",
                "stop loss",
                "should i stop loss",
                "take profit",
                "should i take profit",
                "hold it",
                "is this still worth buying",
                "is it still worth buying",
            ]
        )
        short_finance_followups = [
            "后面还值得拿吗",
            "还值得拿吗",
            "还能拿吗",
            "值不值得拿",
            "还适合买入吗",
            "现在适合买入吗",
            "该不该止损",
            "要不要止损",
            "该不该止盈",
            "要不要止盈",
            "should i cut the loss here",
            "should i stop loss here",
            "should i take profit here",
            "is this still worth buying",
            "is it still worth buying",
        ]
        return (any(marker in query for marker in referential_markers) and any(marker in query for marker in action_markers)) or (
            english_referential and english_action
        ) or any(marker in lowered for marker in [marker.lower() for marker in short_finance_followups])

    def _postprocess_product_prediction(self, query: str, product: dict) -> dict:
        lowered = query.lower()
        patched = dict(product)
        if self._has_etf_or_lof_token(query):
            patched["label"] = "etf"
            patched["score"] = max(float(product["score"]), 0.99)
            return patched
        if any(term in query for term in ["基金", "公募", "私募", "定投", "申赎"]) or self._contains_english_token(lowered, ["fund", "funds"]):
            patched["label"] = "fund"
            patched["score"] = max(float(product["score"]), 0.92)
            return patched
        macro_terms = ["宏观", "降息", "降准", "cpi", "pmi", "gdp", "macro", "policy", "美联储", "央行", "财政赤字"]
        if any(term in lowered for term in macro_terms):
            patched["label"] = "macro"
            patched["score"] = max(float(product["score"]), 0.92)
            return patched
        if any(term in query for term in ["板块", "行业"]) and any(term in lowered for term in macro_terms):
            patched["label"] = "macro"
            patched["score"] = max(float(product["score"]), 0.9)
            return patched
        if any(term in lowered for term in ["营收", "净利润", "earnings", "revenue", "net profit", "rally", "fall", "selloff", "pullback", "drawdown"]) and not (
            self._has_etf_or_lof_token(query)
            or self._contains_english_token(lowered, ["fund", "funds"])
            or any(term in lowered for term in ["macro", "cpi", "pmi", "gdp"])
        ):
            patched["label"] = "stock"
            patched["score"] = max(float(product["score"]), 0.88)
            return patched
        return patched

    @staticmethod
    def _has_etf_or_lof_token(query: str) -> bool:
        return bool(re.search(r"(?i)(?:^|[^a-z])(?:etf|lof)(?:$|[^a-z])", query))

    def _apply_intent_rules(self, query: str, intents: list[dict]) -> list[dict]:
        merged = {item["label"]: item["score"] for item in intents}
        lowered = query.lower()
        if any(term in query for term in ["为什么", "为啥", "咋", "咋回事"]) and any(term in query for term in ["跌", "涨"]):
            self._apply_low_confidence_boost(merged, "market_explanation", 0.88, max_existing=0.45)
        if "why" in lowered and any(term in lowered for term in ["fall", "rally", "drop", "surge"]):
            self._apply_low_confidence_boost(merged, "market_explanation", 0.88, max_existing=0.45)
        if contains_bucket_market_term(query) and any(term in query for term in ["能买吗", "买吗", "还能买", "值得拿", "还能拿", "持有"]):
            self._apply_low_confidence_boost(merged, "market_explanation", 0.76, max_existing=0.45)
        if any(
            term in query
            for term in [
                "值得拿",
                "值得买",
                "值得买吗",
                "值得入手",
                "值不值得买",
                "值不值得入手",
                "值得持有",
                "继续拿",
                "还能拿吗",
                "还能不能拿",
                "值不值得拿",
                "还适合拿吗",
                "适合买入",
                "适合入手",
                "适合长期拿",
                "适合长期持有",
                "长期拿",
                "长期持有",
                "长期定投",
            ]
        ):
            self._apply_low_confidence_boost(
                merged,
                "hold_judgment",
                0.8,
                max_existing=0.5,
                blockers={"buy_sell_timing": 0.72},
            )
            self._apply_low_confidence_boost(merged, "buy_sell_timing", 0.74, max_existing=0.5)
        if any(term in query for term in ["买吗", "能买吗", "还能买", "上车", "买入", "入手", "值得买", "值得入手", "该不该买", "要不要买", "卖出", "止盈", "止损", "适合买"]):
            self._apply_low_confidence_boost(
                merged,
                "buy_sell_timing",
                0.78,
                max_existing=0.5,
            )
        if "新闻" in query:
            self._apply_low_confidence_boost(merged, "event_news_query", 0.76, max_existing=0.45)
        if any(term in lowered for term in ["news", "headline", "announcement", "update"]):
            self._apply_low_confidence_boost(merged, "event_news_query", 0.76, max_existing=0.45)
        if any(term in query for term in ["费率", "申赎"]):
            self._apply_low_confidence_boost(merged, "trading_rule_fee", 0.84, max_existing=0.45)
        if any(term in query for term in ["是什么产品", "产品概要", "产品说明"]):
            self._apply_low_confidence_boost(merged, "product_info", 0.8, max_existing=0.45)
        if any(term in query for term in ["ETF", "etf", "LOF", "lof", "基金"]) and any(
            term in query for term in ["区别", "不同", "一回事", "哪个", "谁更", "更适合", "更稳", "波动更小", "适合新手"]
        ):
            self._apply_low_confidence_boost(merged, "product_info", 0.82, max_existing=0.55)
            self._apply_low_confidence_boost(merged, "peer_compare", 0.72, max_existing=0.55)
        if any(term in query for term in ["风险高吗", "波动大吗"]):
            self._apply_low_confidence_boost(merged, "risk_analysis", 0.76, max_existing=0.45)
        if any(term in query for term in ["基本面", "业绩", "财报", "营收", "净利润", "毛利率"]):
            self._apply_low_confidence_boost(merged, "fundamental_analysis", 0.84, max_existing=0.45)
        if any(term in lowered for term in ["revenue", "net profit", "earnings", "fundamental", "fundamentals"]):
            self._apply_low_confidence_boost(merged, "fundamental_analysis", 0.84, max_existing=0.45)
        if (
            any(term in query for term in ["宏观", "降息", "降准", "cpi", "pmi", "财政赤字", "美联储议息"])
            or any(term in lowered for term in ["macro", "cpi", "pmi", "gdp", "fed", "fomc", "affect", "impact", "what happens"])
        ) and (
            any(term in query for term in ["影响", "怎么走"])
            or any(term in lowered for term in ["affect", "impact", "what happens"])
        ):
            self._apply_low_confidence_boost(merged, "macro_policy_impact", 0.9, max_existing=0.45)
        if any(term in query for term in ["会怎么走", "怎么走"]) or any(term in lowered for term in ["what happens to", "how will"]):
            self._apply_low_confidence_boost(merged, "price_query", 0.78, max_existing=0.55)
        return self._sorted_scores(merged)

    def _apply_topic_rules(self, query: str, topics: list[dict]) -> list[dict]:
        merged = {item["label"]: item["score"] for item in topics}
        lowered = query.lower()
        if any(term in query for term in ["跌", "涨", "行情"]):
            self._apply_low_confidence_boost(merged, "price", 0.88, max_existing=0.45)
        if any(term in query for term in ["为什么", "为啥"]) and any(term in query for term in ["跌", "涨"]):
            if self._has_advice_or_hold_marker(query):
                self._apply_low_confidence_boost(merged, "news", 0.74, max_existing=0.45)
            if self._has_advice_or_hold_marker(query) and any(term in query for term in ["茅台", "五粮液", "平安", "白酒", "银行", "券商", "消费"]):
                self._apply_low_confidence_boost(merged, "industry", 0.72, max_existing=0.45)
        if any(term in lowered for term in ["fall", "rally", "drop", "surge", "price", "how much", "what happens"]):
            self._apply_low_confidence_boost(merged, "price", 0.82, max_existing=0.45)
        if contains_bucket_market_term(query) and any(term in query for term in ["最近", "今天", "能买吗", "买吗", "还能买", "值得拿", "还能拿"]):
            self._apply_low_confidence_boost(merged, "news", 0.74, max_existing=0.45)
            self._apply_low_confidence_boost(merged, "macro", 0.7, max_existing=0.45)
        if any(term in query for term in ["新闻", "公告", "消息"]):
            self._apply_low_confidence_boost(merged, "news", 0.74, max_existing=0.45)
        if any(term in lowered for term in ["news", "headline", "announcement", "update"]):
            self._apply_low_confidence_boost(merged, "news", 0.74, max_existing=0.45)
        if any(term in query for term in ["行业", "板块", "白酒", "银行股", "科技股", "消费股", "地产股", "券商股"]):
            self._apply_low_confidence_boost(merged, "industry", 0.72, max_existing=0.45)
        if any(term in lowered for term in ["sector", "industry", "board"]) or any(term in query for term in ["板块", "行业"]):
            self._apply_low_confidence_boost(merged, "industry", 0.72, max_existing=0.45)
        if any(term in query for term in ["宏观", "降息", "降准", "CPI", "PMI", "财政赤字", "美联储议息"]) or any(
            term in lowered for term in ["macro", "cpi", "pmi", "gdp", "policy", "fed", "fomc"]
        ):
            self._apply_low_confidence_boost(merged, "macro", 0.84, max_existing=0.45)
            self._apply_low_confidence_boost(merged, "policy", 0.84, max_existing=0.45)
        if any(term in query for term in ["风险", "波动", "回撤"]) and not (any(term in query for term in ["为什么", "为啥"]) or "why" in lowered):
            self._apply_low_confidence_boost(merged, "risk", 0.68, max_existing=0.45)
        if any(term in query for term in ["定投", "申赎", "费率", "LOF", "区别"]):
            self._apply_low_confidence_boost(merged, "product_mechanism", 0.8, max_existing=0.45)
        if any(term in query for term in ["区别", "哪个好", "有什么不同"]):
            self._apply_low_confidence_boost(merged, "comparison", 0.72, max_existing=0.45)
        if any(term in query for term in ["基本面", "业绩", "财报", "营收", "净利润", "毛利率"]):
            self._apply_low_confidence_boost(merged, "fundamentals", 0.84, max_existing=0.45)
        if any(term in lowered for term in ["revenue", "net profit", "earnings", "fundamental", "fundamentals"]):
            self._apply_low_confidence_boost(merged, "fundamentals", 0.84, max_existing=0.45)
        if any(term in query for term in ["会怎么走", "怎么走"]) or any(term in lowered for term in ["what happens to", "how will"]):
            self._apply_low_confidence_boost(merged, "price", 0.82, max_existing=0.6)
        if any(term in query for term in ["哪个好", "谁更", "更便宜", "更值得"]) and "ETF" not in query and "etf" not in query:
            self._apply_low_confidence_boost(merged, "fundamentals", 0.38, max_existing=0.35)
            self._apply_low_confidence_boost(merged, "valuation", 0.36, max_existing=0.35)
        return self._sorted_scores(merged)

    def _has_advice_or_hold_marker(self, query: str) -> bool:
        lowered = query.lower()
        return any(
            term in query
            for term in ["值得", "持有", "拿吗", "能买吗", "买吗", "适合", "定投", "止盈", "止损", "加仓", "减仓"]
        ) or any(term in lowered for term in ["should i", "worth", "hold", "buy", "sell"])

    def _suppress_intent_noise(self, query: str, intents: list[dict]) -> list[dict]:
        lowered = query.lower()
        if any(term in query for term in ["费率", "申赎", "规则", "包含哪些", "包含什么"]):
            intents = [
                item
                for item in intents
                if item["label"] in {"product_info", "trading_rule_fee", "peer_compare"}
                or item["score"] >= 0.98
            ]
        fundamental_query = any(term in lowered for term in ["revenue", "net profit", "fundamental", "fundamentals", "earnings"]) or any(
            term in query for term in ["营收", "净利润", "基本面", "财报"]
        )
        if fundamental_query:
            price_context = any(term in lowered for term in ["price", "stock price", "share price"]) or any(term in query for term in ["股价", "价格", "涨", "跌", "行情", "怎么走"])
            intents = [item for item in intents if item["label"] != "price_query" or price_context]
        price_outlook_query = any(term in query for term in ["会怎么走", "怎么走"]) or any(term in lowered for term in ["what happens to", "how will"])
        if any(term in lowered for term in ["cpi", "pmi", "macro", "policy", "fed", "fomc"]) or any(term in query for term in ["降息", "降准", "宏观", "美联储议息"]):
            intents = [
                item
                for item in intents
                if item["label"] != "valuation_analysis"
                and (item["label"] != "price_query" or price_outlook_query or item["score"] >= 0.82)
            ]
        if any(item["label"] == "macro_policy_impact" for item in intents):
            intents = [item for item in intents if item["label"] != "price_query" or item["score"] >= 0.9]
        if any(term in lowered for term in ["what happens to", "会怎么走", "how does"]) and any(
            term in lowered for term in ["cpi", "pmi", "macro", "policy", "fed", "fomc", "降息", "降准", "财政赤字", "美联储议息"]
        ):
            intents = [item for item in intents if item["label"] != "valuation_analysis" or item["score"] >= 0.85]
        if any(term in lowered for term in ["why"]) or any(term in query for term in ["为什么", "为啥"]):
            intents = [item for item in intents if item["label"] != "price_query" or item["score"] >= 0.8]
            if any(item["label"] == "market_explanation" for item in intents) and not any(term in query for term in ["风险", "波动", "回撤"]):
                intents = [item for item in intents if item["label"] != "risk_analysis" or item["score"] >= 0.85]
        return intents

    def _suppress_topic_noise(self, query: str, topics: list[dict]) -> list[dict]:
        lowered = query.lower()
        if any(term in query for term in ["费率", "申赎", "规则", "包含哪些", "包含什么"]):
            topics = [
                item
                for item in topics
                if item["label"] in {"product_mechanism", "comparison"}
                or item["score"] >= 0.98
            ]
        fundamental_query = any(term in lowered for term in ["revenue", "net profit", "fundamental", "fundamentals", "earnings"]) or any(
            term in query for term in ["营收", "净利润", "基本面", "财报"]
        )
        if fundamental_query:
            price_context = any(term in lowered for term in ["price", "stock price", "share price"]) or any(term in query for term in ["股价", "价格", "涨", "跌", "行情", "怎么走"])
            topics = [item for item in topics if item["label"] != "price" or price_context]
        price_outlook_query = any(term in query for term in ["会怎么走", "怎么走"]) or any(term in lowered for term in ["what happens to", "how will"])
        if any(term in lowered for term in ["cpi", "pmi", "macro", "policy", "fed", "fomc"]) or any(term in query for term in ["降息", "降准", "宏观", "美联储议息"]):
            topics = [
                item
                for item in topics
                if item["label"] not in {"valuation"} or item["score"] >= 0.82
            ]
            if not price_outlook_query:
                topics = [item for item in topics if item["label"] != "price" or item["score"] >= 0.86]
        if any(term in lowered for term in ["what happens to", "会怎么走", "how does"]) and any(
            term in lowered for term in ["cpi", "pmi", "macro", "policy", "fed", "fomc", "降息", "降准", "财政赤字", "美联储议息"]
        ):
            topics = [item for item in topics if item["label"] != "valuation" or item["score"] >= 0.85]
        if any(term in lowered for term in ["why"]) or any(term in query for term in ["为什么", "为啥"]):
            topics = [
                item
                for item in topics
                if item["label"] not in {"product_mechanism", "risk", "news", "industry"} or item["score"] >= 0.9 or self._has_advice_or_hold_marker(query)
            ]
        return topics

    def _apply_low_confidence_boost(
        self,
        merged: dict[str, float],
        label: str,
        score: float,
        max_existing: float,
        blockers: dict[str, float] | None = None,
    ) -> None:
        blockers = blockers or {}
        if any(merged.get(blocking_label, 0.0) >= blocking_score for blocking_label, blocking_score in blockers.items()):
            return
        if merged.get(label, 0.0) <= max_existing:
            merged[label] = max(merged.get(label, 0.0), score)

    def _apply_clarification_gate(
        self,
        query: str,
        product_type: str,
        intents: list[dict],
        topics: list[dict],
        entities: list[dict],
        comparison_targets: list[str],
        time_scope: str,
        base_requires_clarification: bool,
    ) -> tuple[bool, float | None]:
        if self.clarification_gate is None:
            return base_requires_clarification, None

        intent_labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        if self._is_generic_product_compare_query(query, comparison_targets, product_type):
            probability = self.clarification_gate.predict_probability(
                query=query,
                product_type=product_type,
                intent_labels=list(intent_labels),
                topic_labels=list(topic_labels),
                time_scope=time_scope,
                entity_count=len(entities),
                comparison_target_count=len(comparison_targets),
            )
            return False, probability
        if (
            looks_like_general_finance_query(query)
            or looks_like_company_fundamental_query(query)
            or looks_like_disclosure_query(query)
        ) and not self._is_underspecified_investment_query(query):
            probability = self.clarification_gate.predict_probability(
                query=query,
                product_type=product_type,
                intent_labels=list(intent_labels),
                topic_labels=list(topic_labels),
                time_scope=time_scope,
                entity_count=len(entities),
                comparison_target_count=len(comparison_targets),
            )
            return False, probability
        if not entities and contains_bucket_market_term(query):
            probability = self.clarification_gate.predict_probability(
                query=query,
                product_type=product_type,
                intent_labels=list(intent_labels),
                topic_labels=list(topic_labels),
                time_scope=time_scope,
                entity_count=0,
                comparison_target_count=len(comparison_targets),
            )
            return False, probability
        if not entities and self._is_underspecified_investment_query(query):
            probability = self.clarification_gate.predict_probability(
                query=query,
                product_type=product_type,
                intent_labels=list(intent_labels),
                topic_labels=list(topic_labels),
                time_scope=time_scope,
                entity_count=0,
                comparison_target_count=len(comparison_targets),
            )
            return True, probability
        probability = self.clarification_gate.predict_probability(
            query=query,
            product_type=product_type,
            intent_labels=list(intent_labels),
            topic_labels=list(topic_labels),
            time_scope=time_scope,
            entity_count=len(entities),
            comparison_target_count=len(comparison_targets),
        )
        if self._is_generic_product_compare(comparison_targets):
            return False, probability
        if product_type in {"fund", "etf"} and intent_labels.issubset({"product_info", "trading_rule_fee", "peer_compare"}) and topic_labels.issubset({"product_mechanism", "comparison"}):
            return False, probability
        if comparison_targets and not self._comparison_targets_covered(comparison_targets, entities):
            return True, probability
        if entities:
            return False, probability
        if probability >= 0.8:
            return True, probability
        if probability <= 0.2:
            return False, probability
        return base_requires_clarification, probability

    def _apply_question_style_reranker(
        self,
        query: str,
        question_style: str,
        product_type: str,
        intents: list[dict],
        topics: list[dict],
        entities: list[dict],
        comparison_targets: list[str],
    ) -> tuple[str, float | None]:
        if self.question_style_reranker is None:
            return question_style, None

        prediction = self.question_style_reranker.predict(
            query=query,
            base_style=question_style,
            product_type=product_type,
            intent_labels=[item["label"] for item in intents],
            topic_labels=[item["label"] for item in topics],
            entity_count=len(entities),
            comparison_target_count=len(comparison_targets),
        )
        threshold = 0.5 if prediction["label"] in {"forecast", "compare", "advice"} else 0.55
        if prediction["score"] >= threshold:
            return prediction["label"], prediction["score"]
        return question_style, prediction["score"]

    @staticmethod
    def _normalize_sentiment_label(label: str) -> str:
        return {
            "positive": "bullish",
            "negative": "bearish",
            "neutral": "neutral",
            "bullish": "bullish",
            "bearish": "bearish",
            "anxious": "anxious",
        }.get(str(label).strip().lower(), "neutral")

    def _resolve_context_entities(self, user_profile: dict, dialog_context: list) -> tuple[list[dict], list[str]]:
        dialog_entities = self._resolve_entities_from_dialog_context(dialog_context)
        if dialog_entities:
            entity = dialog_entities[0]
            return dialog_entities, [f"context_dialog:{entity['canonical_name']}"]

        profile_entities = self._resolve_entities_from_user_profile(user_profile)
        if profile_entities:
            entity = profile_entities[0]
            return profile_entities, [f"context_profile:{entity['canonical_name']}"]
        return [], []

    def _resolve_entities_from_dialog_context(self, dialog_context: list) -> list[dict]:
        texts = [text for text in self._extract_dialog_context_strings(dialog_context) if text.strip()]
        for text in reversed(texts):
            normalized_text, _ = self.normalizer.normalize(text)
            entities, _, _ = self.entity_resolver.resolve(normalized_text)
            if len(entities) == 1:
                return [self._tag_context_entity(entities[0], "dialog")]
            if len(entities) > 1:
                continue
        return []

    def _resolve_entities_from_user_profile(self, user_profile: dict) -> list[dict]:
        texts = [text for text in self._extract_user_profile_strings(user_profile) if text.strip()]
        resolved_by_id: dict[int, dict] = {}
        for text in texts:
            normalized_text, _ = self.normalizer.normalize(text)
            entities, _, _ = self.entity_resolver.resolve(normalized_text)
            for entity in entities:
                resolved_by_id[int(entity["entity_id"])] = self._tag_context_entity(entity, "profile")
        if len(resolved_by_id) == 1:
            return [next(iter(resolved_by_id.values()))]
        return []

    def _extract_strings(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            strings: list[str] = []
            for nested in value.values():
                strings.extend(self._extract_strings(nested))
            return strings
        if isinstance(value, (list, tuple, set)):
            strings: list[str] = []
            for nested in value:
                strings.extend(self._extract_strings(nested))
            return strings
        return []

    def _extract_dialog_context_strings(self, dialog_context: list) -> list[str]:
        strings: list[str] = []
        for item in dialog_context:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role and role != "user":
                continue
            for field in ("content", "query", "text"):
                if field in item:
                    strings.extend(self._extract_strings(item[field]))
        return strings

    def _extract_user_profile_strings(self, user_profile: dict) -> list[str]:
        profile_fields = {
            "entity",
            "entities",
            "focus_entity",
            "focus_entities",
            "holding",
            "holdings",
            "position",
            "positions",
            "portfolio",
            "preferred_symbol",
            "preferred_symbols",
            "security",
            "securities",
            "symbol",
            "symbols",
            "ticker",
            "tickers",
            "watchlist",
        }
        strings: list[str] = []
        for key, value in user_profile.items():
            if key in profile_fields:
                strings.extend(self._extract_strings(value))
        return strings

    def _tag_context_entity(self, entity: dict, source: str) -> dict:
        tagged = dict(entity)
        tagged["match_type"] = f"context_{source}"
        return tagged

    def _normalize_comparison_targets(self, comparison_targets: list[str]) -> list[str]:
        normalized_targets: list[str] = []
        for target in comparison_targets:
            normalized_target, _ = self.normalizer.normalize(target)
            normalized_target = self._strip_comparison_target_suffix(normalized_target)
            if self._is_generic_product_target(normalized_target):
                canonical = normalized_target
            else:
                entities, _, _ = self.entity_resolver.resolve(normalized_target)
                canonical = entities[0]["canonical_name"] if len(entities) == 1 else normalized_target
            if canonical and canonical not in normalized_targets:
                normalized_targets.append(canonical)
        return normalized_targets

    def _strip_comparison_target_suffix(self, target: str) -> str:
        suffixes = ["估值", "风险", "波动", "价格", "行情", "新闻", "基本面", "费率", "回撤"]
        for suffix in suffixes:
            if target.endswith(suffix) and len(target) > len(suffix):
                return target[: -len(suffix)]
        return target

    def _is_generic_product_target(self, target: str) -> bool:
        normalized = target.strip().lower()
        return normalized in GENERIC_PRODUCT_TARGETS or any(term in normalized for term in ["etf", "lof", "基金", "fund"])

    def _postprocess_question_style(
        self,
        query: str,
        question_style: str,
        intents: list[dict],
        comparison_targets: list[str],
    ) -> str:
        intent_labels = {item["label"] for item in intents}
        compare_markers = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "有啥不同", "差别", "一回事吗"]
        advice_terms = ["值得", "能买吗", "买吗", "买入", "卖出", "上车", "止盈", "止损", "持有", "拿吗", "还能拿吗", "还能不能拿", "值不值得拿", "不适合", "适不适合", "适合长期拿", "适合长期持有", "长期拿", "长期持有", "长期定投", "适合定投", "继续定投", "适合继续定投", "更适合", "更稳"]
        forecast_terms = ["未来", "接下来", "后市", "会怎么走", "后面会怎么走", "怎么看后市"]
        lowered = query.lower()
        english_advice_terms = ["should i", "worth", "suitable", "good to buy", "hold it", "cut the loss"]
        english_compare_terms = ["compare", "difference", "better"]
        open_evaluation_terms = ["你觉得", "怎么看", "如何看", "评价一下", "分析一下", "从投资角度", "后面怎么看", "后续怎么看"]
        has_advice_marker = any(term in query for term in advice_terms) or any(term in lowered for term in english_advice_terms)
        has_why_marker = any(term in query for term in ["为什么", "为啥"]) or "why" in lowered
        if any(term in query for term in open_evaluation_terms) and intent_labels.intersection(
            {"hold_judgment", "buy_sell_timing", "market_explanation", "fundamental_analysis", "valuation_analysis", "risk_analysis"}
        ):
            return "advice"
        if any(term in query for term in ["适合长期拿", "适合长期持有", "长期拿", "长期持有", "长期定投", "适合定投", "继续定投", "适合继续定投"]):
            return "advice"
        if has_advice_marker and has_why_marker:
            return "advice"
        if has_advice_marker and (
            "peer_compare" in intent_labels or comparison_targets or any(term in query for term in compare_markers) or any(term in lowered for term in english_compare_terms)
        ):
            return "advice"
        if has_why_marker:
            return "why"
        if "peer_compare" in intent_labels or comparison_targets or any(term in query for term in compare_markers) or any(term in lowered for term in english_compare_terms):
            return "compare"
        if any(term in query for term in advice_terms) and intent_labels.intersection({"hold_judgment", "buy_sell_timing"}):
            return "advice"
        if any(term in lowered for term in english_advice_terms):
            return "advice"
        if any(term in query for term in forecast_terms):
            return "forecast"
        if any(term in lowered for term in ["future", "outlook", "what happens to", "how will", "will it"]):
            return "forecast"
        if any(term in query for term in compare_markers) and intent_labels.intersection({"product_info", "trading_rule_fee"}):
            return "compare"
        if any(term in query for term in advice_terms) and intent_labels.intersection({"risk_analysis", "fundamental_analysis", "valuation_analysis"}):
            return "advice"
        if intent_labels.intersection({"fundamental_analysis", "valuation_analysis", "risk_analysis"}):
            return "fact"
        return question_style

    def _suppress_compare_intent_noise(
        self,
        query: str,
        intents: list[dict],
        entities: list[dict],
        comparison_targets: list[str],
        product_type: str,
    ) -> list[dict]:
        if self._is_generic_product_compare_query(query, comparison_targets, product_type):
            return [item for item in intents if item["label"] in {"peer_compare", "product_info", "trading_rule_fee"}]
        if self._has_explicit_compare_signal(query, entities, comparison_targets):
            return intents
        return [item for item in intents if not (item["label"] == "peer_compare" and item["score"] < 0.55)]

    def _suppress_compare_topic_noise(
        self,
        query: str,
        topics: list[dict],
        entities: list[dict],
        comparison_targets: list[str],
        product_type: str,
    ) -> list[dict]:
        if self._is_generic_product_compare_query(query, comparison_targets, product_type):
            return [item for item in topics if item["label"] in {"comparison", "product_mechanism"}]
        if self._has_explicit_compare_signal(query, entities, comparison_targets):
            return topics
        return [item for item in topics if not (item["label"] == "comparison" and item["score"] < 0.55)]

    def _requires_entity_clarification(
        self,
        query: str,
        product_type: str,
        entities: list[dict],
        intents: list[dict],
        topics: list[dict],
        comparison_targets: list[str],
        missing_slots: list[str],
    ) -> bool:
        intent_labels = {item["label"] for item in intents}
        topic_labels = {item["label"] for item in topics}
        compare_like = self._is_compare_like_query(query, intent_labels, comparison_targets)
        entity_dependent_intents = {
            "market_explanation",
            "hold_judgment",
            "buy_sell_timing",
            "peer_compare",
            "fundamental_analysis",
            "valuation_analysis",
            "risk_analysis",
            "event_news_query",
        }

        if product_type in {"fund", "etf"} and "trading_rule_fee" in intent_labels:
            return False
        if self._is_generic_product_compare_query(query, comparison_targets, product_type):
            return False
        if looks_like_company_fundamental_query(query) or looks_like_disclosure_query(query):
            return False
        if looks_like_general_finance_query(query) and not self._is_underspecified_investment_query(query):
            return False
        if product_type in {"fund", "etf"} and intent_labels.issubset({"product_info", "trading_rule_fee"}) and topic_labels.issubset({"product_mechanism", "comparison"}):
            return False
        if product_type in {"fund", "etf"} and self._is_generic_product_compare(comparison_targets) and topic_labels.issubset({"product_mechanism", "comparison"}):
            return False
        if compare_like:
            if comparison_targets:
                return not self._comparison_targets_covered(comparison_targets, entities)
            return len(entities) < 2
        if entities:
            return False
        if self._is_bucket_market_query(query, compare_like):
            return False
        if self._is_underspecified_investment_query(query):
            return True
        if "missing_entity" in missing_slots:
            return True
        if product_type in {"macro", "generic_market"}:
            return False
        if intent_labels.intersection(entity_dependent_intents):
            return True
        if product_type in {"stock", "index"} and topic_labels.intersection({"comparison", "fundamentals", "valuation", "risk", "price", "news"}):
            return True
        return False

    def _is_compare_like_query(self, query: str, intent_labels: set[str], comparison_targets: list[str]) -> bool:
        compare_markers = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "差别"]
        return "peer_compare" in intent_labels or bool(comparison_targets) or any(term in query for term in compare_markers)

    def _has_explicit_compare_signal(self, query: str, entities: list[dict], comparison_targets: list[str]) -> bool:
        compare_markers = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "差别"]
        return bool(comparison_targets) or len(entities) >= 2 or any(term in query for term in compare_markers)

    def _is_generic_product_compare(self, comparison_targets: list[str]) -> bool:
        return bool(comparison_targets) and all(self._is_generic_product_target(target) for target in comparison_targets)

    def _is_generic_product_compare_query(self, query: str, comparison_targets: list[str], product_type: str) -> bool:
        compare_markers = ["区别", "不同", "一回事", "哪个", "谁更", "更适合", "更稳", "波动更小", "适合新手", "compare", "difference", "better"]
        generic_terms = ["ETF", "LOF", "基金", "债券基金", "偏股混合", "指数基金", "etf", "lof", "fund"]
        return (
            (product_type in {"fund", "etf"} or any(term in query for term in generic_terms))
            and any(term in query for term in compare_markers)
            and (self._is_generic_product_compare(comparison_targets) or any(term in query for term in generic_terms))
        )

    def _has_explicit_generic_product_compare_signal(self, query: str, comparison_targets: list[str]) -> bool:
        compare_markers = ["区别", "不同", "一回事", "哪个", "谁更", "更适合", "更稳", "波动更小", "适合新手", "compare", "difference", "better"]
        generic_terms = ["ETF", "LOF", "基金", "债券基金", "偏股混合", "指数基金", "etf", "lof", "fund"]
        return any(term in query for term in compare_markers) and (
            self._is_generic_product_compare(comparison_targets) or any(term in query for term in generic_terms)
        )

    def _is_bucket_market_query(self, query: str, compare_like: bool) -> bool:
        if compare_like:
            return False
        return contains_bucket_market_term(query)

    def _comparison_targets_covered(self, comparison_targets: list[str], entities: list[dict]) -> bool:
        if not comparison_targets:
            return len(entities) >= 2
        if not entities:
            return False
        entity_tokens: list[str] = []
        for entity in entities:
            for value in [entity.get("mention"), entity.get("canonical_name"), entity.get("symbol")]:
                if value and value not in entity_tokens:
                    entity_tokens.append(value)
        for target in comparison_targets:
            if not any(target == token or target in token or token in target for token in entity_tokens):
                return False
        return True

    def _sorted_scores(self, items: dict[str, float]) -> list[dict]:
        return [
            {"label": label, "score": round(score, 2)}
            for label, score in sorted(items.items(), key=lambda item: item[1], reverse=True)
            if score >= 0.35
        ][:4]
