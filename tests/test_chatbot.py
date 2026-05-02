from __future__ import annotations

import json
from datetime import date as real_date
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import query_intelligence.chatbot as chatbot_module
from query_intelligence.api.app import create_app
from query_intelligence.chatbot import (
    DeepSeekClient,
    answer_matches_language,
    apply_market_freshness_guard,
    build_evidence_sources,
    build_chatbot_response,
    detect_query_language,
    load_chatbot_config,
    make_answer_language_repair_messages,
)


def _sample_pipeline_result() -> dict[str, Any]:
    nlu_result = {
        "query_id": "Q-chat",
        "raw_query": "你觉得中国平安怎么样？",
        "normalized_query": "中国平安怎么样",
        "question_style": "advice",
        "product_type": {"label": "stock", "score": 0.99},
        "intent_labels": [{"label": "hold_judgment", "score": 0.9}],
        "topic_labels": [{"label": "fundamentals", "score": 0.8}],
        "entities": [{"canonical_name": "中国平安", "symbol": "601318.SH"}],
        "risk_flags": ["investment_advice_like"],
    }
    retrieval_result = {
        "query_id": "Q-chat",
        "nlu_snapshot": nlu_result,
        "executed_sources": ["market_api", "news"],
        "documents": [
            {
                "evidence_id": "D1",
                "source_type": "news",
                "source_name": "证券时报网",
                "title": "中国平安新闻",
                "summary": "公司近期受到市场关注。",
                "source_url": "https://example.com/pingan-news",
            }
        ],
        "structured_data": [
            {
                "evidence_id": "S1",
                "source_type": "market_api",
                "payload": {"symbol": "601318.SH", "price": 42.1},
            }
        ],
        "evidence_groups": [],
        "coverage": {"price": True, "news": True},
        "coverage_detail": {},
        "warnings": [],
        "retrieval_confidence": 0.8,
        "analysis_summary": {
            "data_readiness": {"has_price_data": True, "has_news": True},
            "market_signal": {"symbol": "601318.SH", "trend_signal": "neutral"},
        },
        "debug_trace": {"candidate_count": 2, "after_dedup": 2, "top_ranked": []},
    }
    return {"nlu_result": nlu_result, "retrieval_result": retrieval_result}


def test_load_chatbot_config_applies_json_and_env_overrides(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"port": 9001},
                "ui": {"title": "Custom Title"},
                "deepseek": {"base_url": "https://example.invalid/v1", "api_key": "from-file"},
                "live_data": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    monkeypatch.setenv("CHATBOT_TITLE", "Env Title")
    monkeypatch.setenv("CHATBOT_LIVE_DATA", "true")

    config = load_chatbot_config(config_path, load_env_file=False)

    assert config["server"]["host"] == "127.0.0.1"
    assert config["server"]["port"] == 9001
    assert config["ui"]["title"] == "Env Title"
    assert config["deepseek"]["base_url"] == "https://example.invalid/v1"
    assert config["deepseek"]["api_key"] == "from-env"
    assert config["live_data"]["enabled"] is True


class _FakeHTTPResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "这是基于证据的润色回答。",
                                "key_points": ["价格数据可用", "新闻证据可用"],
                                "risk_disclaimer": "仅供参考，不构成投资建议。",
                                "evidence_used": ["S1", "D1", "NOT_ALLOWED"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class _FakeHTTPClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeHTTPResponse()


def test_deepseek_client_generates_normalized_answer() -> None:
    http_client = _FakeHTTPClient()
    client = DeepSeekClient(
        {
            "deepseek": {
                "base_url": "https://api.deepseek.test/v1",
                "chat_path": "/chat/completions",
                "model": "deepseek-chat",
                "api_key": "sk-test",
                "timeout_seconds": 5,
            }
        },
        http_client=http_client,
    )

    answer = client.generate({"query": "你觉得中国平安怎么样？", **_sample_pipeline_result()})

    assert answer["answer"] == "这是基于证据的润色回答。"
    assert answer["key_points"] == ["价格数据可用", "新闻证据可用"]
    assert answer["risk_disclaimer"] == "仅供参考，不构成投资建议。"
    assert answer["evidence_used"] == ["S1", "D1"]
    assert http_client.calls[0]["url"] == "https://api.deepseek.test/v1/chat/completions"
    assert "Chinese" in http_client.calls[0]["json"]["messages"][0]["content"]


class _SequentialHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self.payload, ensure_ascii=False),
                    }
                }
            ]
        }


class _SequentialHTTPClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _SequentialHTTPResponse(self.responses[len(self.calls) - 1])


def test_deepseek_client_repairs_answer_language_for_english_query() -> None:
    http_client = _SequentialHTTPClient(
        [
            {
                "answer": "主要受政策预期支持。",
                "key_points": ["政策支持"],
                "risk_disclaimer": "不构成投资建议。",
                "evidence_used": ["S1", "D1"],
            },
            {
                "answer": "The available evidence points to policy expectations as a possible support factor.",
                "key_points": ["Policy expectations are a possible support factor."],
                "risk_disclaimer": "This answer is based only on the provided evidence and is not investment advice.",
                "evidence_used": ["S1", "D1"],
            },
        ]
    )
    client = DeepSeekClient(
        {
            "deepseek": {
                "base_url": "https://api.deepseek.test/v1",
                "chat_path": "/chat/completions",
                "model": "deepseek-chat",
                "api_key": "sk-test",
                "timeout_seconds": 5,
            }
        },
        http_client=http_client,
    )
    record = {
        "query": "What do you think about Ping An Insurance (601318.SH)?",
        **_sample_pipeline_result(),
    }
    record["nlu_result"]["raw_query"] = record["query"]

    answer = client.generate(record)

    assert len(http_client.calls) == 2
    assert "English" in http_client.calls[0]["json"]["messages"][0]["content"]
    assert "English" in http_client.calls[1]["json"]["messages"][0]["content"]
    assert "current_output" in http_client.calls[1]["json"]["messages"][1]["content"]
    assert answer["answer"].startswith("The available evidence")
    assert answer["key_points"] == ["Policy expectations are a possible support factor."]


def test_language_detector_allows_english_query_with_chinese_entity_name() -> None:
    query = "What do you think about 中国平安?"
    output = {
        "answer": "主要受政策预期支持。",
        "key_points": ["政策支持"],
        "risk_disclaimer": "不构成投资建议。",
        "evidence_used": ["D1"],
    }

    assert detect_query_language(query) == "en"
    assert answer_matches_language(output, query) is False
    messages = make_answer_language_repair_messages(query, output)
    assert "English" in messages[0]["content"]


def test_chatbot_response_falls_back_when_deepseek_is_unconfigured() -> None:
    client = DeepSeekClient({"deepseek": {"api_key": ""}})

    response = build_chatbot_response(
        query="你觉得中国平安怎么样？",
        pipeline_result=_sample_pipeline_result(),
        deepseek_client=client,
    )

    assert response["llm"]["status"] == "fallback"
    assert "DeepSeek API 未配置" in response["answer"]
    assert response["risk_disclaimer"]
    assert response["evidence_sources"][0]["title"]
    assert response["nlu_result"]["query_id"] == "Q-chat"


def test_chatbot_response_uses_english_fallback_for_english_query() -> None:
    client = DeepSeekClient({"deepseek": {"api_key": ""}})

    response = build_chatbot_response(
        query="What do you think about Ping An Insurance (601318.SH)?",
        pipeline_result=_sample_pipeline_result(),
        deepseek_client=client,
    )

    assert response["llm"]["status"] == "fallback"
    assert "DeepSeek API is not configured" in response["answer"]
    assert response["risk_disclaimer"].startswith("This answer is based only")
    assert response["key_points"][0].startswith("Generated a structured analysis summary")


def test_evidence_sources_include_source_name_and_web_link() -> None:
    record = {"query": "你觉得中国平安怎么样？", **_sample_pipeline_result()}

    sources = build_evidence_sources(record, ["D1"])

    assert sources == [
        {
            "evidence_id": "D1",
            "source_type": "news",
            "source_name": "证券时报网",
            "title": "中国平安新闻",
            "source_url": "https://example.com/pingan-news",
        }
    ]


def test_market_freshness_guard_blocks_stale_today_answer(monkeypatch) -> None:
    class RegularTradingDate:
        @classmethod
        def today(cls):
            return real_date(2026, 4, 28)

    monkeypatch.setattr(chatbot_module, "date", RegularTradingDate)
    record = {"query": "茅台股票今天涨了吗", **_sample_pipeline_result()}
    record["retrieval_result"]["structured_data"][0]["payload"]["trade_date"] = "2026-04-22"
    answer = {
        "answer": "根据2026年4月22日行情，今日股价下跌。",
        "key_points": ["旧行情"],
        "risk_disclaimer": "风险提示",
        "evidence_used": ["S1"],
    }

    guarded = apply_market_freshness_guard(answer, record)

    assert guarded["answer"].startswith("未获取到今日")
    assert "不能据此判断今天" in guarded["answer"]
    assert guarded["evidence_used"] == ["S1"]


def test_market_freshness_guard_blocks_stale_english_today_answer(monkeypatch) -> None:
    class RegularTradingDate:
        @classmethod
        def today(cls):
            return real_date(2026, 4, 28)

    monkeypatch.setattr(chatbot_module, "date", RegularTradingDate)
    record = {"query": "Did Kweichow Moutai rise today?", **_sample_pipeline_result()}
    record["retrieval_result"]["structured_data"][0]["payload"]["trade_date"] = "2026-04-22"
    answer = {
        "answer": "Based on the April 22 quote, it fell today.",
        "key_points": ["Old quote"],
        "risk_disclaimer": "Risk note",
        "evidence_used": ["S1"],
    }

    guarded = apply_market_freshness_guard(answer, record)

    assert guarded["answer"].startswith("Unable to retrieve today's")
    assert "cannot use this data to determine" in guarded["answer"]
    assert guarded["key_points"][0].startswith("Today's quote was not retrieved")
    assert guarded["evidence_used"] == ["S1"]


def test_market_freshness_guard_uses_latest_quote_on_non_trading_day(monkeypatch) -> None:
    class WeekendDate:
        @classmethod
        def today(cls):
            return real_date(2026, 5, 2)

    monkeypatch.setattr(chatbot_module, "date", WeekendDate)
    record = {"query": "Did Ping An rise today?", **_sample_pipeline_result()}
    record["retrieval_result"]["structured_data"][0]["payload"].update(
        {"trade_date": "2026-04-30", "close": 57.63, "pct_change_1d": 0.14}
    )
    answer = {
        "answer": "It rose today.",
        "key_points": ["Original point"],
        "risk_disclaimer": "Risk note",
        "evidence_used": ["S1"],
    }

    guarded = apply_market_freshness_guard(answer, record)

    assert guarded["answer"].startswith("Today (2026-05-02) is not a regular A-share trading day")
    assert "latest available trading-day quote is from 2026-04-30" in guarded["answer"]
    assert "57.63" in guarded["answer"]
    assert guarded["key_points"][0].startswith("Today (2026-05-02) is not a regular A-share trading day")
    assert guarded["evidence_used"] == ["S1"]


class _FakeService:
    def run_pipeline(self, query: str, **kwargs) -> dict[str, Any]:
        return _sample_pipeline_result()


class _FakeDeepSeek:
    model = "fake-deepseek"

    def generate(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": f"已回答：{record['query']}",
            "key_points": ["要点一"],
            "risk_disclaimer": "风险提示",
            "evidence_used": ["S1"],
        }


def test_chat_endpoint_and_index_page() -> None:
    app = create_app(
        service=_FakeService(),
        app_config={"ui": {"title": "Financial Chatbot by Group x"}},
        deepseek_client=_FakeDeepSeek(),
    )
    client = TestClient(app)

    index = client.get("/")
    assert index.status_code == 200
    assert "Financial Chatbot by Group x" in index.text
    assert 'class="chat-messages"' in index.text
    assert "bubble-user" in index.text
    assert "typing-dots" in index.text
    assert "Key Points" in index.text
    assert "Evidence Sources" in index.text

    response = client.post("/chat", json={"query": "你觉得中国平安怎么样？"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer"] == "已回答：你觉得中国平安怎么样？"
    assert payload["key_points"] == ["要点一"]
    assert payload["evidence_used"] == ["S1"]
    assert payload["evidence_sources"][0]["source_name"] == "market_api"
    assert payload["llm"]["status"] == "ok"


def test_chat_endpoint_rejects_empty_query() -> None:
    app = create_app(service=_FakeService(), deepseek_client=_FakeDeepSeek())
    client = TestClient(app)

    response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 422
