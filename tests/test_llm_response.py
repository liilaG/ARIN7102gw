from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.llm_response import (
    ChatModel,
    DEFAULT_FEW_SHOT_SOURCE,
    _question_style_from_payload,
    build_record_from_query,
    build_frontend_response,
    compact_payload,
    coerce_query,
    coerce_top_k,
    extract_json_object,
    hf_token,
    load_few_shot_bank,
    ANSWER_FEW_SHOTS,
    make_next_question_language_repair_messages,
    next_questions_match_language,
    normalize_next_questions,
    resolve_model_path,
    select_few_shots,
)


def _sample_record(question_style: str = "why") -> dict:
    return {
        "status": "ok",
        "query": "国企ETF中银今天异动主要受什么影响？",
        "nlu_result": {
            "query_id": "Q1",
            "question_style": question_style,
            "product_type": {"label": "etf"},
            "intent_labels": ["market_explanation"],
            "entities": ["国企ETF中银"],
            "time_scope": "today",
            "risk_flags": ["market_volatility"],
        },
        "retrieval_result": {
            "query_id": "Q1",
            "retrieval_confidence": 0.88,
            "warnings": ["synthetic_evidence", "low_retrieval_confidence"],
            "debug_trace": {"candidate_count": 12},
            "documents": [
                {
                    "evidence_id": "D1",
                    "source_type": "news",
                    "title": "ETF movement",
                    "summary": "Policy support improved investor confidence.",
                    "body": "long body should not be copied in full",
                }
            ],
            "structured_data": [
                {
                    "evidence_id": "S1",
                    "source_type": "market_api",
                    "as_of": "2023-04-05",
                    "quality_flags": ["synthetic_source"],
                    "payload": {
                        "price": 1.25,
                        "change_1d": 0.03,
                        "history": [{"close": 1.2}],
                    },
                }
            ],
        },
        "statistical_result": {
            "statistics_version": "synthetic_statistical_v1",
            "calculation_scope": {"is_synthetic": True},
            "price_statistics": {
                "trend_signal": "positive",
                "volume_change_signal": "increase",
                "technical_summary": "Positive trend with increased volume.",
            },
            "overall_statistical_summary": {
                "data_sufficiency": "partial",
                "overall_signal": "positive",
                "answerable": True,
                "should_abstain": False,
                "summary": "ETF movement was driven by policy support and investor confidence.",
            },
        },
        "sentiment_result": {
            "sentiment_version": "synthetic_sentiment_v1",
            "overall_sentiment": {"label": "neutral", "confidence": 0.85},
            "news_sentiment": {"label": "positive", "confidence": 0.9},
        },
        "metadata": {"is_synthetic": True},
    }


def test_compact_payload_uses_evidence_summary_and_sanitizes_source_generation_terms() -> None:
    payload = compact_payload(_sample_record())
    dumped = json.dumps(payload, ensure_ascii=False).lower()

    assert "evidence_summary" in payload["retrieval_result"]
    assert "documents" not in payload["retrieval_result"]
    assert "debug_trace" not in dumped
    assert "history" not in dumped
    assert "synthetic" not in dumped
    assert payload["retrieval_result"]["warnings"] == ["low_retrieval_confidence"]


def test_compact_payload_uses_retrieval_analysis_summary_when_statistical_result_is_absent() -> None:
    record = _sample_record()
    record.pop("statistical_result")
    record["retrieval_result"]["analysis_summary"] = {
        "market_signal": {"trend_signal": "positive"},
        "fundamental_signal": {"profitability_signal": "stable"},
        "data_readiness": {"has_price_data": True, "has_news": True},
    }

    payload = compact_payload(record)

    assert payload["statistical_result"]["retrieval_analysis_summary"]["market_signal"] == {"trend_signal": "positive"}
    assert payload["statistical_result"]["retrieval_analysis_summary"]["data_readiness"]["has_price_data"] is True


def test_compact_payload_falls_back_to_nlu_raw_query() -> None:
    record = _sample_record()
    record.pop("query")
    record["nlu_result"]["raw_query"] = "中国平安最近为什么涨？"

    payload = compact_payload(record)

    assert payload["query"] == "中国平安最近为什么涨？"


def test_build_record_from_query_connects_query_intelligence_output_shape() -> None:
    class FakeService:
        def run_pipeline(self, **kwargs):
            assert kwargs["query"] == "中国平安最近为什么涨？"
            assert kwargs["top_k"] == 5
            return {
                "nlu_result": {"query_id": "Q1", "question_style": "why"},
                "retrieval_result": {"query_id": "Q1", "documents": [], "structured_data": []},
            }

    record = build_record_from_query("中国平安最近为什么涨？", top_k=5, debug=False, service=FakeService())

    assert record["status"] == "ok"
    assert record["query"] == "中国平安最近为什么涨？"
    assert record["nlu_result"]["query_id"] == "Q1"
    assert record["retrieval_result"]["query_id"] == "Q1"


def test_resolve_model_path_prefers_local_models_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models--owner--demo-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_model_path("owner/demo-model", models_dir=tmp_path)

    assert resolved == str(model_dir)


def test_resolve_model_path_uses_hf_cache_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "models--owner--demo-model"
    snapshot = cache_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (cache_dir / "refs").mkdir()
    (cache_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_model_path("owner/demo-model", models_dir=tmp_path)

    assert resolved == str(snapshot)


def test_resolve_model_path_uses_explicit_hf_cache_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "models--owner--demo-model"
    snapshot = cache_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (cache_dir / "refs").mkdir()
    (cache_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_model_path(str(cache_dir), models_dir=tmp_path / "unused")

    assert resolved == str(snapshot)


def test_hf_token_reads_standard_environment_variables(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert hf_token() is None

    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "fallback-token")
    assert hf_token() == "fallback-token"

    monkeypatch.setenv("HF_TOKEN", "primary-token")
    assert hf_token() == "primary-token"


def test_load_few_shot_bank_has_chinese_and_english_examples_for_each_question_style() -> None:
    answer_bank, next_bank = load_few_shot_bank(Path(DEFAULT_FEW_SHOT_SOURCE))

    assert set(answer_bank) == {"fact", "why", "compare", "advice", "forecast"}
    assert set(next_bank) == {"fact", "why", "compare", "advice", "forecast"}
    for style, examples in answer_bank.items():
        assert len(examples) == 2, style
        queries = [item["input"]["query"] for item in examples]
        assert any(any("\u4e00" <= char <= "\u9fff" for char in query) for query in queries)
        assert any(not any("\u4e00" <= char <= "\u9fff" for char in query) for query in queries)

    dumped = json.dumps(answer_bank, ensure_ascii=False).lower()
    assert "synthetic" not in dumped


def test_select_few_shots_uses_matching_question_style_only() -> None:
    answer_bank, _ = load_few_shot_bank(Path(DEFAULT_FEW_SHOT_SOURCE))
    payload = compact_payload(_sample_record(question_style="compare"))

    selected = select_few_shots(payload, answer_bank, [])

    assert len(selected) == 2
    assert all(_question_style_from_payload(item["input"]) == "compare" for item in selected)


def test_select_few_shots_does_not_use_mismatched_fallback_styles() -> None:
    payload = compact_payload(_sample_record(question_style="forecast"))

    selected = select_few_shots(payload, {}, [])

    assert selected == []


def test_select_few_shots_keeps_matching_fallback_styles() -> None:
    payload = compact_payload(_sample_record(question_style="why"))

    selected = select_few_shots(payload, {}, ANSWER_FEW_SHOTS)

    assert selected
    assert all(_question_style_from_payload(item["input"]) == "why" for item in selected)


def test_normalize_next_questions_returns_three_bounded_predictions() -> None:
    result = normalize_next_questions(
        {
            "predictions": [
                {"question": "A?", "score": 1.5, "reason": "x"},
                {"question": "B?", "score": -1, "reason": "y"},
            ]
        },
        "What next?",
    )

    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["score"] == 1.0
    assert result["predictions"][1]["score"] == 0.0
    assert result["predictions"][2]["reason"] == "fallback"


def test_next_questions_language_mismatch_can_be_repaired_by_model_instruction() -> None:
    output = {
        "predictions": [
            {"question": "What drove the price move?", "score": 0.9, "reason": "why_followup"},
            {"question": "What risks should I watch?", "score": 0.8, "reason": "risk_followup"},
            {"question": "What is the outlook?", "score": 0.7, "reason": "forecast_followup"},
        ]
    }

    assert next_questions_match_language(output, "中国平安最近为什么涨？") is False

    messages = make_next_question_language_repair_messages("中国平安最近为什么涨？", output)

    assert "Chinese" in messages[0]["content"]
    assert "current_output" in messages[1]["content"]


def test_normalize_next_questions_keeps_model_predictions_without_hardcoded_translation() -> None:
    result = normalize_next_questions(
        {
            "predictions": [
                {"question": "What drove the price move?", "score": 0.9, "reason": "why_followup"},
                {"question": "What risks should I watch?", "score": 0.8, "reason": "risk_followup"},
                {"question": "What is the outlook?", "score": 0.7, "reason": "forecast_followup"},
            ]
        },
        "中国平安最近为什么涨？",
    )

    assert result["predictions"][0]["question"] == "What drove the price move?"
    assert result["predictions"][0]["reason"] == "why_followup"


def test_extract_json_object_repairs_common_malformed_json() -> None:
    pytest.importorskip("json_repair")

    result = extract_json_object('```json\n{"answer": "ok", "key_points": ["a",],}\n```')

    assert result == {"answer": "ok", "key_points": ["a"]}


def test_chat_model_generate_json_retries_after_invalid_output() -> None:
    model = object.__new__(ChatModel)
    model.model_id = "fake-model"
    calls: list[list[dict[str, str]]] = []

    def fake_generate_text(messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float) -> str:
        calls.append(messages)
        if len(calls) == 1:
            return "not json"
        return '{"answer":"ok"}'

    model.generate_text = fake_generate_text

    result = model.generate_json(
        [{"role": "user", "content": "return json"}],
        max_new_tokens=10,
        temperature=0,
        json_retries=1,
    )

    assert result == {"answer": "ok"}
    assert len(calls) == 2
    assert "not valid JSON" in calls[1][-1]["content"]


def test_coerce_top_k_rejects_invalid_values() -> None:
    assert coerce_top_k(None) == 20
    assert coerce_top_k("5") == 5

    with pytest.raises(ValueError, match="top_k must be an integer"):
        coerce_top_k("abc")

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        coerce_top_k(0)


def test_coerce_query_rejects_blank_values() -> None:
    assert coerce_query("  中国平安最近为什么涨？  ") == "中国平安最近为什么涨？"

    with pytest.raises(ValueError, match="query must not be blank"):
        coerce_query("   ")


def test_cli_rejects_invalid_query_inputs() -> None:
    root = Path(__file__).resolve().parents[1]

    invalid_top_k = subprocess.run(
        [sys.executable, "scripts/llm_response.py", "--query", "中国平安最近为什么涨？", "--top-k", "0"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_top_k.returncode == 2
    assert "top_k must be greater than 0" in invalid_top_k.stderr

    blank_query = subprocess.run(
        [sys.executable, "scripts/llm_response.py", "--query", "   "],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blank_query.returncode == 2
    assert "query must not be blank" in blank_query.stderr


def test_build_frontend_response_adds_llm_sections_without_model_loading() -> None:
    response = build_frontend_response(
        _sample_record(),
        {
            "answer": "主要受政策支持和情绪改善影响。",
            "key_points": ["政策支持", "情绪改善"],
            "evidence_used": ["D1", "S1"],
            "limitations": ["数据覆盖不完整"],
            "risk_disclaimer": "不构成投资建议。",
        },
        {"predictions": [{"question": "后续政策怎么看？", "score": 0.9, "reason": "policy_followup"}]},
        answer_model="answer-model",
        next_question_model="next-model",
    )

    assert response["answer_generation"]["model_name"] == "answer-model"
    assert response["next_question_prediction"]["model_name"] == "next-model"
    assert response["request"]["query_id"] == "Q1"


def test_build_frontend_response_localizes_missing_risk_disclaimer() -> None:
    english_record = _sample_record()
    english_record["query"] = "Why has Ping An been rising recently?"
    english_record["nlu_result"]["raw_query"] = english_record["query"]

    english_response = build_frontend_response(
        english_record,
        {"answer": "It is mainly supported by policy expectations.", "key_points": [], "evidence_used": [], "limitations": []},
        {"predictions": []},
        answer_model="answer-model",
        next_question_model="next-model",
    )

    assert english_response["answer_generation"]["risk_disclaimer"] == (
        "This answer is based only on the provided evidence and is not investment advice."
    )

    chinese_response = build_frontend_response(
        _sample_record(),
        {"answer": "主要受政策预期支持。", "key_points": [], "evidence_used": [], "limitations": []},
        {"predictions": []},
        answer_model="answer-model",
        next_question_model="next-model",
    )

    assert chinese_response["answer_generation"]["risk_disclaimer"] == "以上内容仅基于给定证据生成，不构成投资建议或确定性买卖结论。"


def test_build_frontend_response_trims_overlong_answer_text() -> None:
    response = build_frontend_response(
        _sample_record(),
        {"answer": "上涨原因很多。" * 80, "key_points": [], "evidence_used": [], "limitations": []},
        {"predictions": []},
        answer_model="answer-model",
        next_question_model="next-model",
    )

    assert len(response["answer_generation"]["answer"]) <= 243


def test_build_frontend_response_caps_evidence_used() -> None:
    response = build_frontend_response(
        _sample_record(),
        {
            "answer": "主要受估值修复和行业情绪改善影响。",
            "key_points": [],
            "evidence_used": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
            "limitations": [],
        },
        {"predictions": []},
        answer_model="answer-model",
        next_question_model="next-model",
    )

    assert response["answer_generation"]["evidence_used"] == ["E1", "E2", "E3", "E4", "E5", "E6"]
