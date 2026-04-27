from __future__ import annotations

import json
import time
from threading import Barrier, Thread
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from query_intelligence.api.app import create_app
from query_intelligence.nlu.source_planner import SourcePlanner
from query_intelligence.retrieval.pipeline import RetrievalPipeline
from query_intelligence.retrieval.packager import RetrievalPackager
from query_intelligence.service import QueryIntelligenceService, build_default_service, build_demo_service


ROOT = Path(__file__).resolve().parents[1]


def load_schema(name: str) -> dict:
    path = ROOT / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


class _NoopNLUPipeline:
    def run(self, **kwargs) -> dict:
        return {
            "query_id": "Q-test",
            "raw_query": kwargs["query"],
            "normalized_query": kwargs["query"],
            "question_style": "fact",
            "product_type": {"label": "stock", "score": 1.0},
            "intent_labels": [],
            "topic_labels": [],
            "entities": [],
            "comparison_targets": [],
            "keywords": [],
            "time_scope": "unspecified",
            "forecast_horizon": "unspecified",
            "sentiment_of_user": "neutral",
            "operation_preference": "unknown",
            "required_evidence_types": [],
            "source_plan": [],
            "risk_flags": [],
            "missing_slots": [],
            "confidence": 1.0,
            "explainability": {"matched_rules": [], "top_features": []},
        }


class _NoopRetrievalPipeline:
    def run(self, *, nlu_result: dict, top_k: int, debug: bool) -> dict:
        return {
            "query_id": nlu_result["query_id"],
            "nlu_snapshot": nlu_result,
            "executed_sources": [],
            "documents": [],
            "structured_data": [],
            "evidence_groups": [],
            "coverage": {},
            "coverage_detail": {},
            "warnings": [],
            "retrieval_confidence": 1.0,
            "analysis_summary": {},
            "debug_trace": {"candidate_count": top_k, "after_dedup": 0, "top_ranked": []},
        }


def test_service_rejects_top_k_outside_public_contract() -> None:
    service = QueryIntelligenceService(_NoopNLUPipeline(), _NoopRetrievalPipeline())
    nlu_result = service.analyze_query("中国平安最近为什么涨？")

    with pytest.raises(ValueError, match="top_k must be less than or equal to 100"):
        service.retrieve_evidence(nlu_result, top_k=101)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        service.run_pipeline("中国平安最近为什么涨？", top_k=0)


QUERY_CASES = [
    {
        "id": "single_stock_why_advice",
        "query": "茅台今天为什么跌？后面还值得拿吗？",
        "nlu": {
            "normalized_query_prefix": "贵州茅台",
            "product_type_label": "stock",
            "question_style": "advice",
            "intent_labels": {"market_explanation", "hold_judgment", "buy_sell_timing"},
            "topic_labels": {"price", "industry", "news"},
            "entity_symbols": ["600519.SH"],
            "time_scope": "today",
            "source_plan_prefix": ["market_api", "news", "industry_sql"],
            "risk_flags": {"investment_advice_like"},
            "required_evidence_types": {"price", "news", "industry", "fundamentals", "risk"},
            "missing_slots": [],
        },
        "retrieval": {
            "executed_sources": {"news", "research_note", "market_api", "fundamental_sql", "industry_sql"},
            "warnings": ["announcement_not_found_recent_window"],
            "coverage": {
                "price": True,
                "news": True,
                "industry": True,
                "fundamentals": True,
                "announcement": False,
                "risk": True,
            },
            "document_source_types": {"news", "research_note"},
            "structured_source_types": {"market_api", "fundamental_sql", "industry_sql"},
        },
    },
    {
        "id": "resolved_stock_comparison",
        "query": "贵州茅台和五粮液哪个好",
        "nlu": {
            "product_type_label": "stock",
            "question_style": "compare",
            "intent_labels": {"peer_compare"},
            "topic_labels": {"comparison", "valuation", "fundamentals"},
            "entity_symbols": ["600519.SH", "000858.SZ"],
            "comparison_targets": ["贵州茅台", "五粮液"],
            "source_plan": ["market_api", "fundamental_sql", "research_note"],
            "missing_slots": [],
        },
        "retrieval": {
            "executed_sources": {"market_api", "fundamental_sql", "research_note"},
            "warnings": [],
            "coverage": {
                "price": True,
                "news": False,
                "fundamentals": True,
                "comparison": True,
            },
            "structured_evidence_ids": [
                "price_600519.SH",
                "price_000858.SZ",
                "fundamental_600519.SH",
                "fundamental_000858.SZ",
            ],
            "document_source_types": {"research_note"},
        },
    },
    {
        "id": "bucket_industry_query",
        "query": "白酒板块最近还能买吗？",
        "nlu": {
            "product_type_label": "stock",
            "question_style": "advice",
            "intent_labels": {"hold_judgment", "buy_sell_timing", "market_explanation"},
            "topic_labels": {"industry", "news", "macro"},
            "entity_symbols": [],
            "source_plan_contains": {"news", "industry_sql", "macro_sql"},
            "risk_flags": {"investment_advice_like"},
            "required_evidence_types": {"macro", "industry", "news"},
            "missing_slots": [],
        },
        "retrieval": {
            "executed_sources": {"news", "research_note", "industry_sql", "macro_sql"},
            "warnings": [],
            "coverage": {
                "price": False,
                "news": True,
                "industry": True,
                "fundamentals": False,
                "macro": True,
                "risk": True,
            },
            "document_source_types": {"news", "research_note"},
            "structured_source_types": {"industry_sql", "macro_sql"},
        },
    },
    {
        "id": "missing_entity_comparison",
        "query": "510300和159915哪个好？",
        "nlu": {
            "product_type_label": "etf",
            "question_style": "compare",
            "intent_labels": {"peer_compare"},
            "topic_labels": {"comparison"},
            "entity_symbols": [],
            "comparison_targets": ["510300", "159915"],
            "source_plan": [],
            "missing_slots": ["missing_entity"],
            "risk_flags": {"entity_not_found", "clarification_required"},
        },
        "retrieval": {
            "executed_sources": set(),
            "warnings": ["clarification_required_missing_entity"],
            "coverage": {
                "price": False,
                "news": False,
                "industry": False,
                "fundamentals": False,
                "announcement": False,
                "product_mechanism": False,
                "macro": False,
                "risk": False,
                "comparison": False,
            },
            "documents_empty": True,
            "structured_empty": True,
        },
    },
]


@pytest.fixture(scope="module")
def app_client() -> TestClient:
    return TestClient(create_app(service=build_demo_service()))


def _labels(items: list[dict]) -> set[str]:
    return {item["label"] for item in items}


def _symbols(items: list[dict]) -> list[str]:
    return [item["symbol"] for item in items if item.get("symbol")]


def _source_types(items: list[dict]) -> set[str]:
    return {item["source_type"] for item in items}


def _assert_nlu_matches(actual: dict, expected: dict) -> None:
    if "normalized_query_prefix" in expected:
        assert actual["normalized_query"].startswith(expected["normalized_query_prefix"])
    if "product_type_label" in expected:
        assert actual["product_type"]["label"] == expected["product_type_label"]
    if "question_style" in expected:
        assert actual["question_style"] == expected["question_style"]
    if "intent_labels" in expected:
        assert _labels(actual["intent_labels"]) >= expected["intent_labels"]
    if "topic_labels" in expected:
        assert _labels(actual["topic_labels"]) >= expected["topic_labels"]
    if "entity_symbols" in expected:
        assert _symbols(actual["entities"]) == expected["entity_symbols"]
    if "comparison_targets" in expected:
        assert actual["comparison_targets"] == expected["comparison_targets"]
    if "time_scope" in expected:
        assert actual["time_scope"] == expected["time_scope"]
    if "source_plan" in expected:
        assert actual["source_plan"] == expected["source_plan"]
    if "source_plan_prefix" in expected:
        assert actual["source_plan"][: len(expected["source_plan_prefix"])] == expected["source_plan_prefix"]
    if "source_plan_contains" in expected:
        assert set(actual["source_plan"]) >= expected["source_plan_contains"]
    if "missing_slots" in expected:
        assert actual["missing_slots"] == expected["missing_slots"]
    if "risk_flags" in expected:
        assert set(actual["risk_flags"]) >= expected["risk_flags"]
    if "required_evidence_types" in expected:
        assert set(actual["required_evidence_types"]) >= expected["required_evidence_types"]


def _assert_retrieval_matches(actual: dict, expected: dict) -> None:
    assert actual["debug_trace"]["candidate_count"] >= actual["debug_trace"]["after_dedup"]
    if "executed_sources" in expected:
        assert set(actual["executed_sources"]) == expected["executed_sources"]
    if "warnings" in expected:
        assert actual["warnings"] == expected["warnings"]
    if "coverage" in expected:
        for key, value in expected["coverage"].items():
            assert actual["coverage"][key] is value
    if "document_source_types" in expected:
        assert _source_types(actual["documents"]) >= expected["document_source_types"]
    if "structured_source_types" in expected:
        assert _source_types(actual["structured_data"]) >= expected["structured_source_types"]
    if "structured_evidence_ids" in expected:
        assert [item["evidence_id"] for item in actual["structured_data"]] == expected["structured_evidence_ids"]
    if expected.get("documents_empty"):
        assert actual["documents"] == []
    if expected.get("structured_empty"):
        assert actual["structured_data"] == []


def test_nlu_extracts_stock_entity_intents_and_sources() -> None:
    service = build_demo_service()

    result = service.analyze_query(
        "茅台今天为什么跌？后面还值得拿吗？",
        debug=True,
    )

    assert result["normalized_query"].startswith("贵州茅台")
    assert result["product_type"]["label"] == "stock"
    assert {item["label"] for item in result["intent_labels"]} >= {
        "market_explanation",
        "hold_judgment",
    }
    assert result["entities"][0]["symbol"] == "600519.SH"
    assert result["time_scope"] == "today"
    assert result["source_plan"][:3] == ["market_api", "news", "industry_sql"]


def test_retrieval_returns_mixed_evidence_and_coverage() -> None:
    service = build_demo_service()
    nlu_result = service.analyze_query("茅台今天为什么跌？后面还值得拿吗？", debug=True)

    result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "market_api" in result["executed_sources"]
    assert "news" in result["executed_sources"]
    assert result["coverage"]["price"] is True
    assert result["coverage"]["risk"] is True
    assert any(item["source_type"] == "news" for item in result["documents"])
    assert any(item["source_type"] == "market_api" for item in result["structured_data"])
    assert "announcement_not_found_recent_window" in result["warnings"]
    assert result["debug_trace"]["candidate_count"] >= result["debug_trace"]["after_dedup"]

    summary = result.get("analysis_summary")
    assert isinstance(summary, dict)
    readiness = summary.get("data_readiness")
    assert isinstance(readiness, dict)
    assert isinstance(readiness.get("has_news"), bool)
    assert isinstance(readiness.get("has_price_data"), bool)


def test_nlu_uses_dialog_context_for_follow_up_query() -> None:
    service = build_demo_service()

    result = service.analyze_query(
        "后面还值得拿吗？",
        dialog_context=[{"role": "user", "content": "贵州茅台今天为什么跌？"}],
        debug=True,
    )

    assert _symbols(result["entities"]) == ["600519.SH"]
    assert "clarification_required" not in result["risk_flags"]
    assert result["source_plan"]


def test_nlu_uses_user_profile_entities_for_follow_up_query() -> None:
    service = build_demo_service()

    result = service.analyze_query(
        "后面还值得拿吗？",
        user_profile={"holdings": ["贵州茅台"]},
        debug=True,
    )

    assert _symbols(result["entities"]) == ["600519.SH"]
    assert "clarification_required" not in result["risk_flags"]
    assert result["source_plan"]


def test_generic_index_query_requires_clarification() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("指数最近能买吗", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "missing_entity" in nlu_result["missing_slots"]
    assert "clarification_required" in nlu_result["risk_flags"]
    assert nlu_result["source_plan"] == []
    assert retrieval_result["warnings"] == ["clarification_required_missing_entity"]


def test_explicit_why_query_is_not_misclassified_as_advice() -> None:
    service = build_demo_service()

    result = service.analyze_query("上证指数今天为什么跌", debug=True)

    assert result["question_style"] == "why"
    assert "investment_advice_like" not in result["risk_flags"]


def test_colloquial_stock_followup_does_not_require_clarification() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("茅台今天为啥跌, 后面还能拿吗", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert _symbols(nlu_result["entities"]) == ["600519.SH"]
    assert "clarification_required" not in nlu_result["risk_flags"]
    assert "missing_entity" not in nlu_result["missing_slots"]
    assert {"market_api", "news"} <= set(nlu_result["source_plan"])
    assert "news" in retrieval_result["executed_sources"]
    assert retrieval_result["coverage"]["news"] is True


def test_recent_week_why_query_with_entity_keeps_source_plan() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("贵州茅台最近一周为啥跌", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["question_style"] == "why"
    assert _symbols(nlu_result["entities"]) == ["600519.SH"]
    assert "clarification_required" not in nlu_result["risk_flags"]
    assert {"market_api", "news"} <= set(nlu_result["source_plan"])
    assert "clarification_required_missing_entity" not in retrieval_result["warnings"]


def test_same_entity_alias_comparison_is_classified_as_compare() -> None:
    service = build_demo_service()

    result = service.analyze_query("沪深300etf和300ETF有什么不同", debug=True)

    assert result["question_style"] == "compare"
    assert _symbols(result["entities"]) == ["510300.SH"]
    assert "clarification_required" not in result["risk_flags"]


def test_typo_stock_name_resolves_entity_and_news_plan() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("茅苔今天为什么跌", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "stock"
    assert _symbols(nlu_result["entities"]) == ["600519.SH"]
    assert {"market_api", "news"} <= set(nlu_result["source_plan"])
    assert "news" in retrieval_result["executed_sources"]


def test_typo_etf_query_does_not_add_spurious_second_entity() -> None:
    service = build_demo_service()

    result = service.analyze_query("創业板ETF波动大吗", debug=True)

    assert _symbols(result["entities"]) == ["159915.SZ"]


def test_split_etf_alias_normalizes_back_to_etf_entity() -> None:
    service = build_demo_service()

    result = service.analyze_query("沪深300ET F适合定投吗", debug=True)

    assert result["product_type"]["label"] == "etf"
    assert _symbols(result["entities"]) == ["510300.SH"]


def test_english_stock_why_and_hold_query_outputs_valid_json_contract() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("Why did Moutai fall today? Is it still worth holding?", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "stock"
    assert _symbols(nlu_result["entities"]) == ["600519.SH"]
    assert nlu_result["question_style"] == "advice"
    assert {"market_api", "news"} <= set(nlu_result["source_plan"])
    assert "clarification_required" not in nlu_result["risk_flags"]
    assert "news" in retrieval_result["executed_sources"]


def test_english_generic_etf_lof_difference_stays_generic() -> None:
    service = build_demo_service()

    result = service.analyze_query("What is the difference between ETF and LOF?", debug=True)

    assert result["product_type"]["label"] == "etf"
    assert result["entities"] == []
    assert result["question_style"] == "compare"
    assert set(result["source_plan"]) == {"research_note", "faq", "product_doc"}
    assert "clarification_required" not in result["risk_flags"]


def test_english_etf_comparison_resolves_both_operands() -> None:
    service = build_demo_service()

    result = service.analyze_query("Which is better, CSI 300 ETF or ChiNext ETF?", debug=True)

    assert result["product_type"]["label"] == "etf"
    assert _symbols(result["entities"]) == ["510300.SH", "159915.SZ"]
    assert result["question_style"] == "compare"
    assert "clarification_required" not in result["risk_flags"]


def test_english_macro_query_plans_macro_and_industry_sources() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("Does macro policy affect liquor?", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "macro"
    assert {"macro_sql", "industry_sql", "news"} <= set(nlu_result["source_plan"])
    assert {"macro_sql", "news"} <= set(retrieval_result["executed_sources"])


@pytest.mark.parametrize(
    "query",
    [
        "今天天气怎么样",
        "上海今天下雨吗",
        "How is the weather today?",
        "Write a Python bubble sort",
        "你好",
        "Did it sell well?",
        "I'm going to buy a CD.",
        "What programs was Trump administration planning on cutting funding to?",
    ],
)
def test_out_of_scope_queries_return_valid_empty_json(query: str) -> None:
    service = build_default_service()

    nlu_result = service.analyze_query(query, debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "out_of_scope"
    assert nlu_result["source_plan"] == []
    assert nlu_result["risk_flags"] == ["out_of_scope_query"]
    assert retrieval_result["warnings"] == ["out_of_scope_query"]
    assert retrieval_result["documents"] == []
    assert retrieval_result["structured_data"] == []


def test_training_asset_stock_alias_does_not_get_early_rejected_as_out_of_scope() -> None:
    service = build_default_service()

    result = service.analyze_query("你觉得寒武纪怎么样？", debug=True)

    assert result["product_type"]["label"] == "stock"
    assert "out_of_scope_query" not in result["risk_flags"]
    assert any(entity["symbol"] == "688256.SH" for entity in result["entities"])
    assert result["source_plan"]


def test_stock_advice_query_maps_to_investment_intents_and_stock_sources() -> None:
    service = build_default_service()

    result = service.analyze_query("你觉得寒武纪值得入手吗", debug=True)

    intent_labels = {item["label"] for item in result["intent_labels"]}
    assert {"buy_sell_timing", "hold_judgment"}.intersection(intent_labels)
    assert {"market_api", "news", "industry_sql", "fundamental_sql", "announcement", "research_note"} <= set(result["source_plan"])
    assert "faq" not in result["source_plan"]
    assert "product_doc" not in result["source_plan"]


def test_unknown_non_finance_brand_is_not_fuzzy_linked_to_stock_entity() -> None:
    service = build_default_service()

    result = service.analyze_query("你觉得中华汽车怎么样", debug=True)

    assert result["entities"] == []
    assert result["source_plan"] == []
    assert "missing_entity" in result["missing_slots"]


def test_unsupported_company_phrase_does_not_fallback_to_embedded_sector() -> None:
    service = build_default_service()

    nlu_result = service.analyze_query("腾讯科技怎么样", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["entities"] == []
    assert "missing_entity" in nlu_result["missing_slots"]
    assert nlu_result["source_plan"] == []
    assert retrieval_result["documents"] == []
    assert retrieval_result["structured_data"] == []


def test_bucket_index_query_clears_missing_entity_and_retrieves_generic_evidence() -> None:
    service = build_demo_service()

    nlu_result = service.analyze_query("成长指数后面还值得拿吗", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "missing_entity" not in nlu_result["missing_slots"]
    assert "clarification_required" not in nlu_result["risk_flags"]
    assert "announcement" not in nlu_result["source_plan"]
    assert "industry_sql" in retrieval_result["executed_sources"]
    assert "clarification_required_missing_entity" not in retrieval_result["warnings"]
    assert "announcement_not_found_recent_window" not in retrieval_result["warnings"]


def test_bucket_market_vocab_is_consistent_for_wide_index_queries() -> None:
    service = build_demo_service()

    result = service.analyze_query("宽基指数最近能买吗", debug=True)

    assert "clarification_required" not in result["risk_flags"]
    assert "announcement" not in result["source_plan"]
    assert "industry_sql" in result["source_plan"]


def test_dialog_context_skips_recent_multi_entity_turn_and_uses_older_single_entity_turn() -> None:
    service = build_demo_service()

    result = service.analyze_query(
        "后面还值得拿吗？",
        dialog_context=[
            {"role": "user", "content": "贵州茅台今天为什么跌？"},
            {"role": "assistant", "content": "贵州茅台和五粮液都受到消费板块影响。"},
        ],
        debug=True,
    )

    assert _symbols(result["entities"]) == ["600519.SH"]
    assert "clarification_required" not in result["risk_flags"]


def test_user_profile_ignores_non_semantic_metadata_strings() -> None:
    service = build_demo_service()

    result = service.analyze_query(
        "后面还值得拿吗？",
        user_profile={"session_id": "贵州茅台", "timezone": "Asia/Shanghai"},
        debug=True,
    )

    assert result["entities"] == []
    assert "clarification_required" in result["risk_flags"]


def test_retrieval_early_exit_does_not_report_unattempted_source_warnings() -> None:
    pipeline = RetrievalPipeline.build_demo()
    nlu_result = {
        "query_id": "q_missing",
        "raw_query": "这个能买吗",
        "normalized_query": "这个能买吗",
        "question_style": "advice",
        "product_type": {"label": "stock", "score": 0.4},
        "intent_labels": [{"label": "hold_judgment", "score": 0.8}],
        "topic_labels": [{"label": "risk", "score": 0.6}],
        "entities": [],
        "comparison_targets": [],
        "keywords": [],
        "time_scope": "unspecified",
        "forecast_horizon": "short_term",
        "sentiment_of_user": "neutral",
        "operation_preference": "buy",
        "required_evidence_types": ["risk"],
        "source_plan": ["announcement", "research_note"],
        "risk_flags": ["clarification_required"],
        "missing_slots": ["missing_entity"],
        "confidence": 0.5,
        "explainability": {"matched_rules": [], "top_features": []},
    }

    result = pipeline.run(nlu_result, top_k=5, debug=True)

    assert result["warnings"] == ["clarification_required_missing_entity"]
    assert result["nlu_snapshot"]["source_plan"] == []


def test_time_scope_changes_source_plan_priority() -> None:
    planner = SourcePlanner()

    today_plan = planner.plan(
        query="贵州茅台今天还值得拿吗",
        product_type="stock",
        intents=[{"label": "hold_judgment", "score": 0.8}],
        topics=[],
        time_scope="today",
    )
    long_term_plan = planner.plan(
        query="贵州茅台长期还值得拿吗",
        product_type="stock",
        intents=[{"label": "hold_judgment", "score": 0.8}],
        topics=[],
        time_scope="long_term",
    )

    assert "announcement" in today_plan["source_plan"]
    assert "announcement" not in long_term_plan["source_plan"]


class _FakeMarketProvider:
    def fetch_bundle(self, symbol: str, canonical_name: str | None = None, product_type: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:  # noqa: ARG002
        return {
            "payload": {"symbol": symbol, "industry_snapshot": {"industry_name": canonical_name or symbol}},
            "fundamental_payload": {"pe_ttm": 10.0 if symbol == "600519.SH" else 20.0},
        }


class _FailingMarketProvider:
    def fetch_bundle(self, symbol: str, canonical_name: str | None = None, product_type: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:  # noqa: ARG002
        raise ConnectionError("Remote end closed connection without response")


class _FakeNewsProvider:
    def fetch_news(self, symbol: str, canonical_name: str, limit: int) -> list[dict]:  # noqa: ARG002
        return [
            {
                "evidence_id": f"news_{symbol}",
                "source_type": "news",
                "title": canonical_name,
                "entity_symbols": [symbol],
            }
        ]


class _RecoveringAnnouncementProvider:
    timeout = 0

    def __init__(self) -> None:
        self.calls = 0

    def fetch_announcements(self, symbol: str, limit: int) -> list[dict]:  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            time.sleep(5.5)
            return []
        return [
            {
                "evidence_id": f"announcement_{symbol}",
                "source_type": "announcement",
                "title": symbol,
                "entity_symbols": [symbol],
            }
        ]


class _FakeAnnouncementProvider:
    def fetch_announcements(self, symbol: str, limit: int) -> list[dict]:  # noqa: ARG002
        return [
            {
                "evidence_id": f"announcement_{symbol}",
                "source_type": "announcement",
                "title": symbol,
                "entity_symbols": [symbol],
            }
        ]


def test_live_retrieval_fetches_all_symbols_for_comparison_queries() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = _FakeMarketProvider()
    pipeline.news_providers = [_FakeNewsProvider()]
    pipeline.announcement_provider = _FakeAnnouncementProvider()
    query_bundle = {
        "query_id": "q1",
        "normalized_query": "贵州茅台和五粮液哪个好",
        "keywords": [],
        "entity_names": ["贵州茅台", "五粮液"],
        "symbols": ["600519.SH", "000858.SZ"],
        "industry_terms": [],
        "source_plan": ["market_api", "news", "fundamental_sql", "announcement", "industry_sql"],
        "product_type": "stock",
        "intent_labels": ["peer_compare"],
        "topic_labels": ["comparison"],
        "time_scope": "unspecified",
    }

    structured_items = pipeline._fetch_structured_items(query_bundle)
    docs = pipeline._fetch_live_docs(query_bundle, top_k=5)

    assert {item["payload"]["symbol"] for item in structured_items if item["source_type"] == "market_api"} == {
        "600519.SH",
        "000858.SZ",
    }
    assert {item["payload"]["symbol"] for item in structured_items if item["source_type"] == "fundamental_sql"} == {
        "600519.SH",
        "000858.SZ",
    }
    assert {item["evidence_id"] for item in docs} == {
        "news_600519.SH",
        "news_000858.SZ",
        "announcement_600519.SH",
        "announcement_000858.SZ",
    }


def test_live_market_provider_failure_is_returned_as_structured_warning() -> None:
    pipeline = RetrievalPipeline.build_demo()
    pipeline.market_provider = _FailingMarketProvider()
    query_bundle = {
        "query_id": "q_provider_failure",
        "normalized_query": "贵州茅台今天股价",
        "keywords": [],
        "entity_names": ["贵州茅台"],
        "symbols": ["600519.SH"],
        "industry_terms": [],
        "source_plan": ["market_api"],
        "product_type": "stock",
        "intent_labels": ["price_query"],
        "topic_labels": [],
        "time_scope": "today",
    }

    structured_items = pipeline._fetch_structured_items(query_bundle)

    warning_item = next(item for item in structured_items if item["source_type"] == "provider_warning")
    assert warning_item["payload"]["symbol"] == "600519.SH"
    assert warning_item["payload"]["provider_warnings"] == [
        "market_provider_fetch_failed:600519.SH:ConnectionError:Remote end closed connection without response"
    ]


def test_announcement_timeout_does_not_permanently_disable_future_fetches() -> None:
    pipeline = RetrievalPipeline.build_demo()
    provider = _RecoveringAnnouncementProvider()
    pipeline.announcement_provider = provider
    pipeline._announcement_cooldown_seconds = 0.05

    query_bundle = {
        "query_id": "q1",
        "normalized_query": "贵州茅台公告",
        "keywords": [],
        "entity_names": ["贵州茅台"],
        "symbols": ["600519.SH"],
        "industry_terms": [],
        "source_plan": ["announcement"],
        "product_type": "stock",
    }

    docs_first = pipeline._fetch_live_docs(query_bundle, top_k=5)
    docs_during_cooldown = pipeline._fetch_live_docs(query_bundle, top_k=5)

    assert docs_first == []
    assert docs_during_cooldown == []
    assert provider.calls == 1
    assert pipeline._announcement_retry_after_monotonic > 0

    time.sleep(0.06)
    while pipeline._announcement_stuck_worker and pipeline._announcement_stuck_worker.is_alive():
        time.sleep(0.05)

    docs_after_recovery = pipeline._fetch_live_docs(query_bundle, top_k=5)

    assert provider.calls == 2
    assert {item["evidence_id"] for item in docs_after_recovery} == {"announcement_600519.SH"}


class _ConcurrentAnnouncementProvider:
    timeout = 10

    def __init__(self) -> None:
        self.calls = 0

    def fetch_announcements(self, symbol: str, limit: int) -> list[dict]:  # noqa: ARG002
        self.calls += 1
        time.sleep(0.2)
        return [
            {
                "evidence_id": f"announcement_{symbol}",
                "source_type": "announcement",
                "title": symbol,
                "entity_symbols": [symbol],
            }
        ]


def test_concurrent_requests_allow_only_one_inflight_announcement_fetch() -> None:
    pipeline = RetrievalPipeline.build_demo()
    provider = _ConcurrentAnnouncementProvider()
    pipeline.announcement_provider = provider

    query_bundle = {
        "query_id": "q_concurrent",
        "normalized_query": "贵州茅台公告",
        "keywords": [],
        "entity_names": ["贵州茅台"],
        "symbols": ["600519.SH"],
        "industry_terms": [],
        "source_plan": ["announcement"],
        "product_type": "stock",
    }

    start_barrier = Barrier(3)
    results: list[list[dict]] = []

    def _runner() -> None:
        start_barrier.wait()
        docs = pipeline._fetch_live_docs(query_bundle, top_k=5)
        results.append(docs)

    t1 = Thread(target=_runner)
    t2 = Thread(target=_runner)
    t1.start()
    t2.start()
    start_barrier.wait()
    t1.join()
    t2.join()

    assert provider.calls == 1
    assert sum(1 for docs in results if docs) == 1


@pytest.mark.parametrize("case", QUERY_CASES, ids=[case["id"] for case in QUERY_CASES])
def test_api_module_handoff_regression_matrix(app_client: TestClient, case: dict) -> None:
    nlu_response = app_client.post("/nlu/analyze", json={"query": case["query"], "debug": True})
    assert nlu_response.status_code == 200
    nlu_payload = nlu_response.json()

    jsonschema.validate(nlu_payload, load_schema("nlu_result.schema.json"))
    _assert_nlu_matches(nlu_payload, case["nlu"])

    retrieval_response = app_client.post(
        "/retrieval/search",
        json={"nlu_result": nlu_payload, "top_k": 5, "debug": True},
    )
    assert retrieval_response.status_code == 200
    retrieval_payload = retrieval_response.json()

    jsonschema.validate(retrieval_payload, load_schema("retrieval_result.schema.json"))
    assert retrieval_payload["query_id"] == nlu_payload["query_id"]
    assert retrieval_payload["nlu_snapshot"]["product_type"] == nlu_payload["product_type"]["label"]
    assert retrieval_payload["nlu_snapshot"]["intent_labels"] == [item["label"] for item in nlu_payload["intent_labels"]]
    assert retrieval_payload["nlu_snapshot"]["entities"] == _symbols(nlu_payload["entities"])
    assert retrieval_payload["nlu_snapshot"]["source_plan"] == nlu_payload["source_plan"]
    _assert_retrieval_matches(retrieval_payload, case["retrieval"])


def test_api_pipeline_response_matches_json_schemas(app_client: TestClient) -> None:
    response = app_client.post("/query/intelligence", json={"query": "茅台今天为什么跌？后面还值得拿吗？", "top_k": 5, "debug": True})

    assert response.status_code == 200
    payload = response.json()

    jsonschema.validate(payload["nlu_result"], load_schema("nlu_result.schema.json"))
    jsonschema.validate(payload["retrieval_result"], load_schema("retrieval_result.schema.json"))


def test_api_artifacts_endpoint_writes_backend_ml_files(tmp_path: Path) -> None:
    client = TestClient(create_app(service=build_demo_service(), artifact_output_dir=tmp_path))

    response = client.post(
        "/query/intelligence/artifacts",
        json={
            "query": "茅台今天为什么跌？后面还值得拿吗？",
            "session_id": "chat-session-1",
            "message_id": "message-1",
            "top_k": 5,
            "debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["query_id"] == payload["nlu_result"]["query_id"] == payload["retrieval_result"]["query_id"]
    assert payload["run_id"]

    artifact_dir = Path(payload["artifact_dir"])
    assert artifact_dir.exists()
    nlu_path = Path(payload["artifacts"]["nlu_result_path"])
    retrieval_path = Path(payload["artifacts"]["retrieval_result_path"])
    manifest_path = Path(payload["artifacts"]["manifest_path"])
    assert nlu_path.exists()
    assert retrieval_path.exists()
    assert manifest_path.exists()

    nlu_payload = json.loads(nlu_path.read_text(encoding="utf-8"))
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert nlu_payload == payload["nlu_result"]
    assert retrieval_payload == payload["retrieval_result"]
    assert manifest["query_id"] == payload["query_id"]
    assert manifest["session_id"] == "chat-session-1"
    assert manifest["message_id"] == "message-1"
    assert manifest["schema_version"] == "1.0"
    assert manifest["artifacts"]["nlu_result_path"] == str(nlu_path)
    assert manifest["artifacts"]["retrieval_result_path"] == str(retrieval_path)

    assert retrieval_payload["documents"]
    document = retrieval_payload["documents"][0]
    assert "source_name" in document
    assert "source_url" in document
    assert "text_excerpt" in document
    assert "body" in document
    assert document["body_available"] is True
    assert retrieval_payload["structured_data"]
    structured = retrieval_payload["structured_data"][0]
    assert "as_of" in structured
    assert "source_name" in structured
    assert "quality_flags" in structured


def test_research_note_without_url_gets_stable_dataset_reference() -> None:
    result = RetrievalPackager().build(
        nlu_result={
            "query_id": "source-ref-test",
            "product_type": {"label": "stock"},
            "intent_labels": [],
            "entities": [],
            "source_plan": ["research_note"],
            "risk_flags": [],
        },
        ranked_docs=[
            {
                "evidence_id": "research_note_abc",
                "doc_id": "abc",
                "source_type": "research_note",
                "source_name": "fir_bench_reports",
                "source_url": "",
                "title": "寒武纪研究报告",
                "summary": "摘要",
                "body": "正文",
                "entity_symbols": ["688256.SH"],
            }
        ],
        structured_items=[],
        groups=[],
        total_candidates=1,
        executed_sources=["research_note"],
    )

    assert result["documents"][0]["source_url"] == "dataset://fir_bench_reports/abc"


def test_retrieval_packager_exposes_live_market_provider_warnings() -> None:
    result = RetrievalPackager().build(
        nlu_result={
            "query_id": "provider-warning-test",
            "product_type": {"label": "stock"},
            "intent_labels": [],
            "entities": [],
            "source_plan": ["market_api"],
            "risk_flags": [],
        },
        ranked_docs=[],
        structured_items=[
            {
                "evidence_id": "price_601318.SH",
                "source_type": "market_api",
                "source_name": "akshare",
                "payload": {
                    "symbol": "601318.SH",
                    "source_name": "akshare",
                    "close": 57.7,
                    "provider_warnings": [
                        "akshare.stock_zh_a_hist_retry_succeeded:RemoteDisconnected:Remote end closed connection without response"
                    ],
                },
            }
        ],
        groups=[],
        total_candidates=0,
        executed_sources=["market_api"],
    )

    assert "akshare.stock_zh_a_hist_retry_succeeded:RemoteDisconnected:Remote end closed connection without response" in result["warnings"]
