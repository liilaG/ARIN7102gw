from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from query_intelligence.bootstrap_public_data import PublicDataBootstrapper
from query_intelligence.config import Settings
from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider
from query_intelligence.integrations.akshare_provider import AKShareNewsProvider
from query_intelligence.integrations.cninfo_provider import CninfoAnnouncementProvider
from query_intelligence.integrations.efinance_provider import EFinanceETFProvider
from query_intelligence.integrations.tushare_provider import TushareMarketProvider, TushareNewsProvider
from query_intelligence.retrieval.pipeline import RetrievalPipeline
from query_intelligence.nlu.classifiers import ProductTypeClassifier
from query_intelligence.repositories.postgres_repository import PostgresDocumentRepository, PostgresStructuredRepository
from query_intelligence.training_data import load_training_rows


class _FakeTushareClient:
    def daily(self, ts_code: str, start_date: str, end_date: str, fields: str):  # noqa: ARG002
        return [
            {
                "ts_code": ts_code,
                "trade_date": "20260422",
                "open": 1600.0,
                "high": 1612.0,
                "low": 1570.0,
                "close": 1578.2,
                "vol": 120000.0,
                "amount": 186000000.0,
                "pct_chg": -2.31,
            }
        ]

    def fina_indicator(self, ts_code: str, fields: str):  # noqa: ARG002
        return [
            {
                "ts_code": ts_code,
                "end_date": "20251231",
                "roe": 33.0,
                "grossprofit_margin": 91.2,
                "netprofit_yoy": 12.1,
                "profit_dedt": 85000000000.0,
            }
        ]

    def major_news(self, src: str, start_date: str, end_date: str, fields: str | None = None):  # noqa: ARG002
        return [
            {
                "title": "贵州茅台短期承压，消费板块回调",
                "content": "消费板块短期回调，贵州茅台价格承压。",
                "pub_time": "2026-04-22 10:00:00",
                "src": src,
            }
        ]

    def daily_basic(self, ts_code: str, trade_date: str, fields: str):  # noqa: ARG002
        self.last_daily_basic_trade_date = trade_date
        if trade_date != "20260422":
            return []
        return [
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "pe_ttm": 22.8,
                "pb": 7.4,
            }
        ]


class _FakeTushareClientNoFina(_FakeTushareClient):
    def fina_indicator(self, ts_code: str, fields: str):  # noqa: ARG002
        raise RuntimeError("no permission for fina_indicator")


class _FakeAkshareModule:
    @staticmethod
    def stock_news_em(symbol: str):
        return [
            {
                "关键词": symbol,
                "新闻标题": "贵州茅台短期承压，白酒板块回调",
                "新闻内容": "白酒板块短线回调，贵州茅台股价承压。",
                "发布时间": "2026-04-22 09:30:00",
                "文章来源": "东方财富",
                "新闻链接": "https://example.com/news-1",
            }
        ]

    @staticmethod
    def stock_zh_a_hist(symbol: str, period: str, start_date: str, end_date: str, adjust: str, timeout=None):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "开盘": 1600.0,
                "最高": 1612.0,
                "最低": 1570.0,
                "收盘": 1578.2,
                "涨跌幅": -2.31,
                "成交量": 120000.0,
                "成交额": 186000000.0,
            }
        ]

    @staticmethod
    def fund_etf_hist_em(symbol: str, period: str, start_date: str, end_date: str, adjust: str):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "开盘": 4.12,
                "最高": 4.18,
                "最低": 4.10,
                "收盘": 4.16,
                "涨跌幅": 0.48,
                "成交量": 880000.0,
                "成交额": 3560000.0,
            }
        ]

    @staticmethod
    def fund_etf_hist_sina(symbol: str):
        return [
            {
                "date": "2026-04-22",
                "open": 4.12,
                "high": 4.18,
                "low": 4.10,
                "close": 4.16,
                "volume": 880000.0,
            }
        ]

    @staticmethod
    def stock_financial_analysis_indicator(symbol: str, start_year: str):  # noqa: ARG002
        return [
            {
                "日期": "2020-03-31",
                "净资产收益率(%)": 8.0,
                "主营业务毛利率(%)": 80.0,
                "每股收益(元)": 12.3,
            },
            {
                "日期": "2025-12-31",
                "净资产收益率(%)": 33.0,
                "主营业务毛利率(%)": 91.2,
                "每股收益(元)": 67.8,
            }
        ]

    @staticmethod
    def stock_individual_info_em(symbol: str, timeout=None):  # noqa: ARG002
        return [
            {"item": "股票简称", "value": "贵州茅台"},
            {"item": "行业", "value": "白酒"},
        ]

    @staticmethod
    def stock_board_industry_hist_em(symbol: str, start_date: str, end_date: str, period: str, adjust: str):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "开盘": 1000.0,
                "收盘": 989.5,
                "涨跌幅": -1.05,
                "成交额": 888000000.0,
            }
        ]


class _FakeAkshareModuleETFOnlySina(_FakeAkshareModule):
    @staticmethod
    def fund_etf_hist_em(symbol: str, period: str, start_date: str, end_date: str, adjust: str):  # noqa: ARG002
        raise RuntimeError("eastmoney etf hist unavailable")


class _FakeAkshareModuleWithFundDetails(_FakeAkshareModule):
    @staticmethod
    def fund_open_fund_info_em(symbol: str, indicator: str = "单位净值走势"):  # noqa: ARG002
        return [
            {"净值日期": "2026-04-21", "单位净值": 4.12, "累计净值": 4.12},
            {"净值日期": "2026-04-22", "单位净值": 4.16, "累计净值": 4.18},
        ]

    @staticmethod
    def fund_individual_detail_info_xq(symbol: str):  # noqa: ARG002
        return [
            {"item": "管理费率", "value": "0.50%"},
            {"item": "托管费率", "value": "0.10%"},
            {"item": "销售服务费率", "value": "0.00%"},
            {"item": "申购费率", "value": "0.12%"},
            {"item": "赎回费率", "value": "0.50%"},
            {"item": "申购状态", "value": "开放申购"},
            {"item": "赎回状态", "value": "开放赎回"},
            {"item": "最低申购金额", "value": "100元"},
            {"item": "赎回规则", "value": "T+1确认，T+2到账"},
            {"item": "交易规则", "value": "场内T+1交易"},
            {"item": "跟踪标的", "value": "沪深300指数"},
            {"item": "基金经理", "value": "张三"},
        ]

    @staticmethod
    def fund_etf_fund_info_em(symbol: str):  # noqa: ARG002
        return [
            {"item": "跟踪标的", "value": "沪深300指数"},
            {"item": "申购状态", "value": "开放申购"},
            {"item": "赎回状态", "value": "开放赎回"},
        ]


class _FakeAkshareModuleWithoutIndustryHistory:
    @staticmethod
    def stock_zh_a_hist(symbol: str, period: str, start_date: str, end_date: str, adjust: str, timeout=None):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "开盘": 100.0,
                "最高": 105.0,
                "最低": 99.0,
                "收盘": 103.0,
                "涨跌幅": 2.0,
                "成交量": 1000000.0,
                "成交额": 103000000.0,
            }
        ]

    @staticmethod
    def stock_individual_info_em(symbol: str, timeout=None):  # noqa: ARG002
        return [
            {"item": "股票简称", "value": "寒武纪"},
            {"item": "行业", "value": "半导体"},
        ]

    @staticmethod
    def stock_financial_analysis_indicator(symbol: str, start_year: str):  # noqa: ARG002
        return []


class _FakeAkshareModuleWithIndexDetails(_FakeAkshareModule):
    @staticmethod
    def stock_zh_index_daily(symbol: str):  # noqa: ARG002
        return [
            {
                "date": "2026-04-21",
                "open": 3920.0,
                "high": 3980.0,
                "low": 3900.0,
                "close": 3960.0,
                "volume": 120000000.0,
            },
            {
                "date": "2026-04-22",
                "open": 3960.0,
                "high": 4020.0,
                "low": 3950.0,
                "close": 4000.0,
                "volume": 130000000.0,
            },
        ]

    @staticmethod
    def stock_zh_index_value_csindex(symbol: str):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "市盈率1": 12.5,
                "市净率1": 1.35,
                "股息率1": 2.1,
                "百分位": 42.0,
            }
        ]


class _FakeAkshareModuleStockHistFails(_FakeAkshareModule):
    @staticmethod
    def stock_zh_a_hist(symbol: str, period: str, start_date: str, end_date: str, adjust: str, timeout=None):  # noqa: ARG002
        raise RuntimeError("eastmoney stock hist unavailable")


class _FakeAkshareModuleStockHistFallsBackToSina(_FakeAkshareModuleStockHistFails):
    @staticmethod
    def stock_zh_a_daily(symbol: str, start_date: str, end_date: str, adjust: str):  # noqa: ARG002
        return [
            {
                "date": "2026-04-22",
                "open": 53.0,
                "high": 54.0,
                "low": 52.8,
                "close": 53.6,
                "volume": 1200000.0,
            },
            {
                "date": "2026-04-21",
                "open": 52.0,
                "high": 53.1,
                "low": 51.8,
                "close": 53.0,
                "volume": 1100000.0,
            }
        ]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeTextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeEFinanceFundModule:
    @staticmethod
    def get_quote_history(code: str, beg: str = None, end: str = None, klt: int = 101, fqt: int = 1):  # noqa: ARG002
        return [
            {
                "日期": "2026-04-22",
                "开盘": 4.10,
                "最高": 4.18,
                "最低": 4.08,
                "收盘": 4.16,
                "涨跌幅": 0.48,
                "成交量": 900000.0,
                "成交额": 3600000.0,
            }
        ]


class _FakeSession:
    def post(self, url: str, data: dict, headers: dict, timeout: int):  # noqa: ARG002
        return _FakeResponse(
            {
                "announcements": [
                    {
                        "announcementTitle": "贵州茅台2025年年度报告公告",
                        "announcementTime": 1776842400000,
                        "adjunctUrl": "finalpage/2026-04-21/123456789.PDF",
                        "secCode": "600519",
                        "secName": "贵州茅台",
                    }
                ]
            }
        )


class _SlowAnnouncementProvider:
    timeout = 1

    def fetch_announcements(self, symbol: str, limit: int = 10):  # noqa: ARG002
        time.sleep(8)
        return []


class _FakeMixedCninfoSession:
    def post(self, url: str, data: dict, headers: dict, timeout: int):  # noqa: ARG002
        return _FakeResponse(
            {
                "announcements": [
                    {
                        "announcementTitle": "无关公司公告",
                        "announcementTime": 1776842300000,
                        "adjunctUrl": "finalpage/2026-04-21/unrelated.PDF",
                        "secCode": "301100",
                        "secName": "营口风光",
                    },
                    {
                        "announcementTitle": "寒武纪2025年年度报告公告",
                        "announcementTime": 1776842400000,
                        "adjunctUrl": "finalpage/2026-04-21/688256.PDF",
                        "secCode": "688256",
                        "secName": "寒武纪",
                    },
                ]
            }
        )


class _FakeCursor:
    def __init__(self, rows: list[dict] | list[list[dict]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, dict]] = []

    def execute(self, sql: str, params: dict) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[dict]:
        if self.rows and isinstance(self.rows[0], list):
            return self.rows.pop(0)
        return self.rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _FakeConnection:
    def __init__(self, rows: list[dict] | list[list[dict]]) -> None:
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self, row_factory=None):  # noqa: ARG002
        return self.cursor_obj


def test_tushare_market_provider_normalizes_market_and_fundamental_payloads() -> None:
    client = _FakeTushareClient()
    provider = TushareMarketProvider(client=client)

    result = provider.fetch_bundle("600519.SH", start_date="20260401", end_date="20260422")

    assert result["source_type"] == "market_api"
    assert result["source_name"] == "tushare"
    assert result["payload"]["symbol"] == "600519.SH"
    assert result["payload"]["source_name"] == "tushare"
    assert result["payload"]["pct_change_1d"] == -2.31
    assert result["payload"]["roe"] == 33.0
    assert result["fundamental_payload"]["source_name"] == "tushare"
    assert result["fundamental_payload"]["report_date"] == "2025-12-31"
    assert result["fundamental_payload"]["pe_ttm"] == 22.8
    assert result["fundamental_payload"]["pb"] == 7.4
    assert result["fundamental_payload"]["roe"] == 33.0
    assert client.last_daily_basic_trade_date == "20260422"


def test_tushare_market_provider_keeps_daily_payload_when_fundamental_permission_missing() -> None:
    provider = TushareMarketProvider(client=_FakeTushareClientNoFina())

    result = provider.fetch_bundle("600519.SH", start_date="20260401", end_date="20260422")

    assert result["payload"]["symbol"] == "600519.SH"
    assert result["payload"]["pct_change_1d"] == -2.31
    assert result["payload"]["roe"] is None
    assert result["status"] == "partial"


def test_tushare_news_provider_normalizes_major_news_items() -> None:
    provider = TushareNewsProvider(client=_FakeTushareClient())

    results = provider.fetch_news(symbol="600519.SH", canonical_name="贵州茅台", limit=5)

    assert results[0]["source_type"] == "news"
    assert results[0]["source_name"] == "新浪财经"
    assert results[0]["entity_symbols"] == ["600519.SH"]
    assert "贵州茅台" in results[0]["title"]


def test_akshare_news_provider_normalizes_news_items() -> None:
    provider = AKShareNewsProvider(ak_module=_FakeAkshareModule())

    results = provider.fetch_news(symbol="600519", canonical_name="贵州茅台", limit=5)

    assert results[0]["source_type"] == "news"
    assert results[0]["title"].startswith("贵州茅台")
    assert results[0]["entity_symbols"] == ["600519.SH"]


def test_akshare_news_provider_rejects_non_security_symbol() -> None:
    provider = AKShareNewsProvider(ak_module=_FakeAkshareModule())

    results = provider.fetch_news(symbol="科技", canonical_name="科技", limit=5)

    assert results == []


def test_akshare_market_provider_normalizes_market_industry_and_fundamental_payloads() -> None:
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModule())

    result = provider.fetch_bundle(symbol="600519.SH", canonical_name="贵州茅台", product_type="stock")

    assert result["source_type"] == "market_api"
    assert result["source_name"] == "akshare"
    assert result["payload"]["symbol"] == "600519.SH"
    assert result["payload"]["source_name"] == "akshare"
    assert result["payload"]["pct_change_1d"] == -2.31
    assert result["payload"]["industry_name"] == "白酒"
    assert result["fundamental_payload"]["source_name"] == "akshare"
    assert result["fundamental_payload"]["report_date"] == "2025-12-31"
    assert result["fundamental_payload"]["roe"] == 33.0


def test_live_market_flag_enables_related_live_document_sources_by_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("QI_USE_LIVE_MARKET", "1")
    monkeypatch.delenv("QI_USE_LIVE_NEWS", raising=False)
    monkeypatch.delenv("QI_USE_LIVE_ANNOUNCEMENT", raising=False)

    settings = Settings.from_env()

    assert settings.use_live_market is True
    assert settings.use_live_news is True
    assert settings.use_live_announcement is True


def test_live_document_sources_can_still_be_disabled_explicitly(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("QI_USE_LIVE_MARKET", "1")
    monkeypatch.setenv("QI_USE_LIVE_NEWS", "0")
    monkeypatch.setenv("QI_USE_LIVE_ANNOUNCEMENT", "0")

    settings = Settings.from_env()

    assert settings.use_live_market is True
    assert settings.use_live_news is False
    assert settings.use_live_announcement is False


def test_retrieval_pipeline_emits_industry_item_from_live_company_profile_when_board_history_missing() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithoutIndustryHistory())

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "寒武纪值得入手吗",
            "symbols": ["688256.SH"],
            "entity_names": ["寒武纪"],
            "keywords": [],
            "source_plan": ["market_api", "industry_sql", "fundamental_sql"],
            "product_type": "stock",
            "intent_labels": ["buy_sell_timing"],
            "topic_labels": ["price"],
        }
    )

    industry_item = next(item for item in structured_items if item["source_type"] == "industry_sql")
    assert industry_item["source_name"] == "akshare_company_profile"
    assert industry_item["payload"]["industry_name"] == "半导体"
    assert industry_item["payload"]["coverage_level"] == "identity_only"
    assert industry_item["payload"]["provider_endpoint"] == "akshare.stock_individual_info_em"
    assert industry_item["payload"]["query_params"] == {"symbol": "688256"}
    assert industry_item["payload"]["source_reference"].startswith("api://akshare.stock_individual_info_em?")


def test_retrieval_pipeline_live_sources_replace_seed_with_provider_metadata() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = TushareMarketProvider(client=_FakeTushareClient())
    pipeline.news_providers = [AKShareNewsProvider(ak_module=_FakeAkshareModule())]
    pipeline.announcement_provider = CninfoAnnouncementProvider(session=_FakeSession())

    query_bundle = {
        "normalized_query": "贵州茅台今天为什么跌",
        "symbols": ["600519.SH"],
        "entity_names": ["贵州茅台"],
        "keywords": [],
        "source_plan": ["market_api", "fundamental_sql", "news", "announcement"],
        "product_type": "stock",
    }

    structured_items = pipeline._fetch_structured_items(query_bundle)
    docs = pipeline._fetch_live_docs(query_bundle, top_k=5)

    market_item = next(item for item in structured_items if item["source_type"] == "market_api")
    fundamental_item = next(item for item in structured_items if item["source_type"] == "fundamental_sql")
    assert market_item["source_name"] == "tushare"
    assert market_item["payload"]["source_name"] == "tushare"
    assert fundamental_item["source_name"] == "tushare"
    assert fundamental_item["payload"]["source_name"] == "tushare"
    assert all(item["payload"].get("source_name") != "seed" for item in structured_items if item["source_type"] in {"market_api", "fundamental_sql"})
    assert any(item["source_type"] == "news" and item["source_url"] == "https://example.com/news-1" for item in docs)
    assert any(item["source_type"] == "announcement" and item["source_url"].startswith("https://static.cninfo.com.cn/") for item in docs)


def test_retrieval_pipeline_output_exposes_live_urls_and_provider_payloads() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = TushareMarketProvider(client=_FakeTushareClient())
    pipeline.news_providers = [AKShareNewsProvider(ak_module=_FakeAkshareModule())]
    pipeline.announcement_provider = CninfoAnnouncementProvider(session=_FakeSession())

    result = pipeline.run(
        {
            "query_id": "live-source-test",
            "raw_query": "贵州茅台今天为什么跌",
            "normalized_query": "贵州茅台今天为什么跌",
            "question_style": "why",
            "product_type": {"label": "stock", "score": 0.99},
            "intent_labels": [{"label": "market_explanation", "score": 0.9}, {"label": "fundamental_analysis", "score": 0.8}],
            "topic_labels": [{"label": "price", "score": 0.9}, {"label": "news", "score": 0.8}, {"label": "fundamentals", "score": 0.8}],
            "entities": [{"mention": "贵州茅台", "entity_type": "stock", "confidence": 0.99, "match_type": "alias_exact", "canonical_name": "贵州茅台", "symbol": "600519.SH"}],
            "comparison_targets": [],
            "keywords": [],
            "time_scope": "today",
            "forecast_horizon": "short_term",
            "sentiment_of_user": "neutral",
            "operation_preference": "unknown",
            "required_evidence_types": ["price", "news", "fundamentals"],
            "source_plan": ["market_api", "news", "announcement", "fundamental_sql"],
            "risk_flags": [],
            "missing_slots": [],
            "confidence": 0.9,
            "explainability": {"matched_rules": [], "top_features": []},
        },
        top_k=5,
        debug=True,
    )

    assert any(item["source_type"] == "news" and item["source_url"] == "https://example.com/news-1" for item in result["documents"])
    assert any(item["source_type"] == "announcement" and item["source_url"].startswith("https://static.cninfo.com.cn/") for item in result["documents"])
    market_item = next(item for item in result["structured_data"] if item["source_type"] == "market_api")
    fundamental_item = next(item for item in result["structured_data"] if item["source_type"] == "fundamental_sql")
    assert market_item["source_name"] == "tushare"
    assert market_item["payload"]["source_name"] == "tushare"
    assert market_item["provider_endpoint"] == "tushare.daily"
    assert market_item["query_params"]["ts_code"] == "600519.SH"
    assert market_item["source_reference"].startswith("api://tushare.daily?")
    assert fundamental_item["source_name"] == "tushare"
    assert fundamental_item["payload"]["source_name"] == "tushare"
    assert fundamental_item["provider_endpoint"] == "tushare.fina_indicator"
    assert fundamental_item["query_params"]["ts_code"] == "600519.SH"
    assert fundamental_item["source_reference"].startswith("api://tushare.fina_indicator?")


def test_retrieval_pipeline_announcement_wait_uses_provider_timeout() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.announcement_provider = _SlowAnnouncementProvider()

    query_bundle = {
        "normalized_query": "贵州茅台公告",
        "symbols": ["600519.SH"],
        "entity_names": ["贵州茅台"],
        "keywords": [],
        "source_plan": ["announcement"],
        "product_type": "stock",
    }

    started = time.time()
    docs = pipeline._fetch_live_docs(query_bundle, top_k=5)
    elapsed = time.time() - started

    assert docs == []
    assert elapsed < 10


def test_cninfo_provider_filters_announcements_to_requested_security_code() -> None:
    provider = CninfoAnnouncementProvider(session=_FakeMixedCninfoSession())

    results = provider.fetch_announcements(symbol="688256.SH", limit=5)

    assert len(results) == 1
    assert results[0]["evidence_id"] == "cninfo_688256_1"
    assert results[0]["title"] == "寒武纪2025年年度报告公告"
    assert results[0]["entity_symbols"] == ["688256.SH"]


def test_akshare_market_provider_falls_back_to_sina_for_etf_history() -> None:
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleETFOnlySina())

    result = provider.fetch_bundle(symbol="510300.SH", canonical_name="沪深300ETF", product_type="etf")

    assert result["payload"]["symbol"] == "510300.SH"
    assert result["payload"]["trade_date"] == "2026-04-22"
    assert result["payload"]["close"] == 4.16
    assert result["payload"]["pct_change_1d"] is None


def test_akshare_market_provider_returns_structured_fund_payloads_for_etf() -> None:
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithFundDetails())

    result = provider.fetch_bundle(symbol="510300.SH", canonical_name="沪深300ETF", product_type="etf")

    assert result["fund_nav_payload"]["source_name"] == "akshare"
    assert result["fund_nav_payload"]["provider"] == "akshare"
    assert result["fund_nav_payload"]["latest_nav"] == 4.16
    assert result["fund_nav_payload"]["accumulated_nav"] == 4.18
    assert result["fund_nav_payload"]["nav_date"] == "2026-04-22"
    assert result["fund_fee_payload"]["management_fee"] == "0.50%"
    assert result["fund_fee_payload"]["custodian_fee"] == "0.10%"
    assert result["fund_fee_payload"]["sales_service_fee"] == "0.00%"
    assert result["fund_fee_payload"]["purchase_fee"] == "0.12%"
    assert result["fund_fee_payload"]["redeem_fee"] == "0.50%"
    assert result["fund_redemption_payload"]["subscription_status"] == "开放申购"
    assert result["fund_redemption_payload"]["redemption_status"] == "开放赎回"
    assert result["fund_redemption_payload"]["purchase_min"] == "100元"
    assert result["fund_redemption_payload"]["redemption_rule"] == "T+1确认，T+2到账"
    assert result["fund_profile_payload"]["subscription_status"] == "开放申购"
    assert result["fund_profile_payload"]["redemption_status"] == "开放赎回"
    assert result["fund_profile_payload"]["purchase_min"] == "100元"
    assert result["fund_profile_payload"]["redemption_rule"] == "T+1确认，T+2到账"
    assert result["fund_profile_payload"]["trading_rule"] == "场内T+1交易"
    assert result["fund_profile_payload"]["tracking_index"] == "沪深300指数"
    assert result["fund_profile_payload"]["fund_manager"] == "张三"


def test_retrieval_pipeline_emits_live_structured_fund_items_for_etf() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithFundDetails())

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "沪深300ETF费率和申赎规则",
            "symbols": ["510300.SH"],
            "entity_names": ["沪深300ETF"],
            "keywords": [],
            "source_plan": ["market_api"],
            "product_type": "etf",
        }
    )

    source_types = {item["source_type"] for item in structured_items}
    assert {"market_api", "fund_nav", "fund_fee", "fund_redemption", "fund_profile"} <= source_types
    fund_nav = next(item for item in structured_items if item["source_type"] == "fund_nav")
    fund_fee = next(item for item in structured_items if item["source_type"] == "fund_fee")
    fund_redemption = next(item for item in structured_items if item["source_type"] == "fund_redemption")
    fund_profile = next(item for item in structured_items if item["source_type"] == "fund_profile")
    assert fund_nav["source_name"] == "akshare"
    assert fund_nav["provider"] == "akshare"
    assert fund_nav["payload"]["latest_nav"] == 4.16
    assert fund_fee["payload"]["management_fee"] == "0.50%"
    assert fund_redemption["payload"]["redemption_rule"] == "T+1确认，T+2到账"
    assert fund_profile["payload"]["redemption_status"] == "开放赎回"


def test_retrieval_pipeline_emits_fund_items_for_mechanism_query_without_market_api_plan() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithFundDetails())

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "沪深300ETF费率和申赎规则",
            "symbols": ["510300.SH"],
            "entity_names": ["沪深300ETF"],
            "keywords": [],
            "source_plan": ["research_note", "faq", "product_doc"],
            "product_type": "etf",
            "topic_labels": ["product_mechanism"],
            "intent_labels": ["trading_rule_fee"],
        }
    )

    source_types = {item["source_type"] for item in structured_items}
    assert {"fund_nav", "fund_fee", "fund_redemption", "fund_profile"} <= source_types


def test_akshare_market_provider_returns_index_daily_and_valuation_payloads() -> None:
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithIndexDetails())

    result = provider.fetch_bundle(symbol="000300.SH", canonical_name="沪深300", product_type="index")

    assert result["payload"]["symbol"] == "000300.SH"
    assert result["payload"]["close"] == 4000.0
    assert result["index_daily_payload"]["close"] == 4000.0
    assert result["index_daily_payload"]["pct_change_1d"] == 1.0101
    assert result["index_valuation_payload"]["pe"] == 12.5
    assert result["index_valuation_payload"]["pb"] == 1.35
    assert result["index_valuation_payload"]["dividend_yield"] == 2.1
    assert result["index_valuation_payload"]["percentile"] == 42.0


def test_retrieval_pipeline_emits_live_structured_index_items() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleWithIndexDetails())

    structured_items = pipeline._fetch_structured_items(
        {
            "normalized_query": "沪深300最近为什么跌",
            "symbols": ["000300.SH"],
            "entity_names": ["沪深300"],
            "keywords": [],
            "source_plan": ["market_api", "news", "research_note", "macro_sql"],
            "product_type": "index",
        }
    )

    source_types = {item["source_type"] for item in structured_items}
    assert {"market_api", "index_daily", "index_valuation"} <= source_types
    index_daily = next(item for item in structured_items if item["source_type"] == "index_daily")
    index_valuation = next(item for item in structured_items if item["source_type"] == "index_valuation")
    assert index_daily["payload"]["close"] == 4000.0
    assert index_valuation["payload"]["pe"] == 12.5


def test_akshare_market_provider_falls_back_to_efinance_for_stock_history(monkeypatch) -> None:  # noqa: ANN001
    def fake_get(url: str, headers: dict, timeout: int):  # noqa: ARG001
        raise RuntimeError("sina quote unavailable")

    monkeypatch.setattr("query_intelligence.integrations.akshare_market_provider.requests.get", fake_get)
    provider = AKShareMarketProvider(
        ak_module=_FakeAkshareModuleStockHistFails(),
        efinance_provider=EFinanceETFProvider(
            fund_module=_FakeEFinanceFundModule(),
            stock_module=_FakeEFinanceFundModule(),
        ),
    )

    result = provider.fetch_bundle(symbol="601318.SH", canonical_name="中国平安", product_type="stock")

    assert result["source_name"] == "efinance"
    assert result["payload"]["source_name"] == "efinance"
    assert result["payload"]["symbol"] == "601318.SH"
    assert result["payload"]["trade_date"] == "2026-04-22"
    assert result["fundamental_payload"]["source_name"] == "akshare"


def test_akshare_market_provider_falls_back_to_sina_for_stock_history() -> None:
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleStockHistFallsBackToSina())

    result = provider.fetch_bundle(symbol="601318.SH", canonical_name="中国平安", product_type="stock")

    assert result["source_name"] == "akshare_sina"
    assert result["payload"]["source_name"] == "akshare_sina"
    assert result["payload"]["trade_date"] == "2026-04-22"
    assert result["payload"]["close"] == 53.6
    assert result["payload"]["pct_change_1d"] == 1.1321


def test_akshare_market_provider_falls_back_to_direct_sina_quote(monkeypatch) -> None:  # noqa: ANN001
    def fake_get(url: str, headers: dict, timeout: int):  # noqa: ARG001
        return _FakeTextResponse(
            'var hq_str_sh601318="中国平安,57.710,57.880,57.770,57.940,57.060,57.760,57.770,26981125,1548637233.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-04-24,10:21:26,00,";'
        )

    monkeypatch.setattr("query_intelligence.integrations.akshare_market_provider.requests.get", fake_get)
    provider = AKShareMarketProvider(ak_module=_FakeAkshareModuleStockHistFails())

    result = provider.fetch_bundle(symbol="601318.SH", canonical_name="中国平安", product_type="stock")

    assert result["source_name"] == "sina_quote"
    assert result["payload"]["source_name"] == "sina_quote"
    assert result["payload"]["trade_date"] == "2026-04-24"
    assert result["payload"]["close"] == 57.77
    assert result["payload"]["pct_change_1d"] == -0.19


def test_efinance_etf_provider_normalizes_history_payloads() -> None:
    provider = EFinanceETFProvider(fund_module=_FakeEFinanceFundModule())

    result = provider.fetch_history("510300.SH")

    assert result["source_type"] == "market_api"
    assert result["source_name"] == "efinance"
    assert result["payload"]["symbol"] == "510300.SH"
    assert result["payload"]["pct_change_1d"] == 0.48


def test_cninfo_provider_normalizes_announcement_items() -> None:
    provider = CninfoAnnouncementProvider(session=_FakeSession())

    results = provider.fetch_announcements(symbol="600519.SH", limit=10)

    assert results[0]["source_type"] == "announcement"
    assert results[0]["entity_symbols"] == ["600519.SH"]
    assert results[0]["source_url"].startswith("https://static.cninfo.com.cn/")


def test_load_training_rows_and_train_product_classifier_from_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "queries.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "product_type", "intent_labels", "topic_labels"])
        writer.writeheader()
        writer.writerow({"query": "茅台今天为什么跌", "product_type": "stock", "intent_labels": "market_explanation", "topic_labels": "price|news"})
        writer.writerow({"query": "沪深300ETF适合定投吗", "product_type": "etf", "intent_labels": "product_info", "topic_labels": "product_mechanism"})

    rows = load_training_rows(dataset_path)
    classifier = ProductTypeClassifier.build_from_records(rows)

    result = classifier.predict("茅台今天为什么跌", [])
    assert rows[0]["product_type"] == "stock"
    assert result["label"] == "stock"


def test_postgres_repository_builds_fts_query_and_normalizes_rows() -> None:
    rows = [
        {
            "doc_id": 1,
            "source_type": "news",
            "source_name": "eastmoney",
            "source_url": "https://example.com/news-1",
            "title": "贵州茅台短期承压",
            "summary": "白酒板块回调",
            "body": "贵州茅台短线回调",
            "publish_time": "2026-04-22T09:30:00+08:00",
            "product_type": "stock",
            "credibility_score": 0.82,
            "entity_symbols": ["600519.SH"],
            "retrieval_score": 0.88,
        }
    ]
    connection = _FakeConnection(rows)
    repo = PostgresDocumentRepository(connection)

    result = repo.search(
        {
            "normalized_query": "贵州茅台 为什么 跌",
            "entity_names": ["贵州茅台"],
            "symbols": ["600519.SH"],
            "keywords": ["下跌", "原因"],
            "source_plan": ["news"],
        },
        top_k=5,
    )

    assert result[0]["source_type"] == "news"
    sql, params = connection.cursor_obj.executed[0]
    assert "websearch_to_tsquery" in sql
    assert params["limit"] == 5
    assert params["allowed_sources"] == ["news"]


def test_postgres_structured_repository_fetches_market_history_and_industry_and_fundamental() -> None:
    connection = _FakeConnection(
        [
            [
                {
                    "symbol": "600519.SH",
                    "trade_date": "2026-04-22",
                    "open": 1600.0,
                    "high": 1612.0,
                    "low": 1570.0,
                    "close": 1578.2,
                    "pct_change": -2.31,
                    "volume": 120000.0,
                    "amount": 186000000.0,
                }
            ],
            [
                {
                    "symbol": "600519.SH",
                    "report_date": "2025-12-31",
                    "revenue": 174120000000.0,
                    "net_profit": 85000000000.0,
                    "roe": 33.0,
                    "gross_margin": 91.2,
                    "pe_ttm": 24.6,
                    "pb": 8.1,
                }
            ],
            [{"industry_name": "白酒"}],
            [
                {
                    "industry_name": "白酒",
                    "trade_date": "2026-04-22",
                    "pct_change": -1.05,
                    "pe": 27.3,
                    "pb": 6.2,
                    "turnover": 0.84,
                }
            ],
        ]
    )
    repo = PostgresStructuredRepository(connection)

    result = repo.fetch({"symbols": ["600519.SH"], "source_plan": ["market_api", "fundamental_sql", "industry_sql"]})

    assert any(item["source_type"] == "market_api" for item in result)
    assert any(item["source_type"] == "fundamental_sql" for item in result)
    assert any(item["source_type"] == "industry_sql" for item in result)


def test_public_data_bootstrapper_writes_complete_data_package(tmp_path: Path) -> None:
    bootstrapper = PublicDataBootstrapper(
        output_dir=tmp_path,
        ak_news_provider=AKShareNewsProvider(ak_module=_FakeAkshareModule()),
        ak_market_provider=AKShareMarketProvider(ak_module=_FakeAkshareModule()),
        cninfo_provider=CninfoAnnouncementProvider(session=_FakeSession()),
        existing_documents=[
            {
                "evidence_id": "faq_001",
                "source_type": "faq",
                "source_name": "internal_faq",
                "title": "ETF 定投 FAQ",
                "summary": "FAQ",
                "body": "FAQ body",
                "publish_time": "2026-04-01T09:00:00+08:00",
                "product_type": "etf",
                "credibility_score": 0.9,
                "entity_symbols": []
            }
        ],
        existing_entities=[
            {
                "entity_id": "1",
                "canonical_name": "贵州茅台",
                "normalized_name": "贵州茅台",
                "symbol": "600519.SH",
                "market": "CN",
                "exchange": "SSE",
                "entity_type": "stock",
                "industry_name": "白酒",
                "status": "active"
            }
        ],
        existing_aliases=[
            {
                "alias_id": "1",
                "entity_id": "1",
                "alias_text": "茅台",
                "normalized_alias": "茅台",
                "alias_type": "common_alias",
                "priority": "1",
                "is_official": "false"
            }
        ],
    )

    bootstrapper.run(
        watchlist=[
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "product_type": "stock"},
            {"symbol": "510300.SH", "canonical_name": "沪深300ETF", "product_type": "etf"},
        ]
    )

    assert (tmp_path / "entity_master.csv").exists()
    assert (tmp_path / "alias_table.csv").exists()
    assert (tmp_path / "documents.json").exists()
    assert (tmp_path / "structured_data.json").exists()

    structured = json.loads((tmp_path / "structured_data.json").read_text(encoding="utf-8"))
    documents = json.loads((tmp_path / "documents.json").read_text(encoding="utf-8"))
    assert "600519.SH" in structured["market_api"]
    assert any(item["source_type"] == "news" for item in documents)
    assert any(item["source_type"] == "announcement" for item in documents)


def test_public_data_bootstrapper_prefers_tushare_primary_sources_when_available(tmp_path: Path) -> None:
    bootstrapper = PublicDataBootstrapper(
        output_dir=tmp_path,
        ak_news_provider=AKShareNewsProvider(ak_module=_FakeAkshareModule()),
        ak_market_provider=AKShareMarketProvider(ak_module=_FakeAkshareModule()),
        tushare_market_provider=TushareMarketProvider(client=_FakeTushareClient()),
        tushare_news_provider=TushareNewsProvider(client=_FakeTushareClient()),
        cninfo_provider=CninfoAnnouncementProvider(session=_FakeSession()),
        existing_documents=[],
        existing_entities=[],
        existing_aliases=[],
        existing_structured_data={"market_api": {}, "fundamental_sql": {}, "industry_sql": {}, "macro_sql": {}, "entity_to_industry": {}},
    )

    bootstrapper.run(
        watchlist=[
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "product_type": "stock"}
        ]
    )

    structured = json.loads((tmp_path / "structured_data.json").read_text(encoding="utf-8"))
    documents = json.loads((tmp_path / "documents.json").read_text(encoding="utf-8"))

    assert structured["market_api"]["600519.SH"]["source_name"] == "tushare"
    assert any(item["source_type"] == "news" and item["source_name"] == "新浪财经" for item in documents)
