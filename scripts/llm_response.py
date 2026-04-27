from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ANSWER_MODEL = "instruction-pretrain/finance-Llama3-8B"
DEFAULT_NEXT_QUESTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_FEW_SHOT_SOURCE = ROOT / "data" / "answer_generation_sft" / "synthetic_source_500.jsonl"
DEFAULT_LLM_MODELS_DIR = ROOT / "models" / "llm"

logger = logging.getLogger("llm_response")


def default_models_dir() -> Path:
    return Path(os.getenv("LLM_MODELS_DIR") or os.getenv("QI_LLM_MODELS_DIR") or DEFAULT_LLM_MODELS_DIR).expanduser()


def hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


ANSWER_SYSTEM_PROMPT = """You are the final answer-generation model for a financial QA system.

Use only the provided compact evidence payload:
- query
- nlu_result
- retrieval_result.evidence_summary
- statistical_result
- sentiment_result
Do not invent missing market data, fundamentals, valuation, macro data, news, or model outputs.
Do not give deterministic buy/sell instructions, guaranteed forecasts, exact target prices, or promises.
Match the user's language.

Return only one valid JSON object with exactly these keys:
{
  "answer": string,
  "key_points": [string],
  "evidence_used": [string],
  "limitations": [string],
  "risk_disclaimer": string
}

Output requirements:
- answer should be concise but complete.
- Directly answer the user's question first, then add the evidence context.
- key_points should contain 3 to 6 items.
- evidence_used should contain evidence_id values from retrieval_result.evidence_summary when available.
- limitations should mention weak, stale, missing, or low-confidence evidence.
- risk_disclaimer must say this is not investment advice.
- Do not include markdown fences.
- Do not include chain-of-thought.
"""


NEXT_QUESTION_SYSTEM_PROMPT = """You are the next-question prediction model for a financial QA frontend.

Read the user's query and the provided pipeline result.
Predict exactly 3 natural follow-up questions that the user is likely to ask next.
Questions must be useful for financial analysis, grounded in the provided entity/product/context, and in the same language as the user.
Prefer follow-ups that match the question type: fact -> detail/breakdown, why -> drivers/risks, advice -> suitability/risk horizon, compare -> peer metrics, forecast -> catalysts/downside.

Return only one valid JSON object with exactly this key:
{
  "predictions": [
    {"question": string, "score": number, "reason": string},
    {"question": string, "score": number, "reason": string},
    {"question": string, "score": number, "reason": string}
  ]
}

Output requirements:
- scores must be between 0 and 1, descending.
- reason should be short, such as "entity_risk_followup" or "valuation_followup".
- Do not include markdown fences.
- Do not include chain-of-thought.
"""


ANSWER_FEW_SHOTS: list[dict[str, Any]] = [
    {
        "input": {
            "query": "How much main capital flowed into 261102.SZ today?",
            "nlu_result": {
                "question_style": "fact",
                "product_type": {"label": "stock"},
                "intent_labels": ["capital_flow"],
                "entities": [{"entity": "261102.SZ", "type": "stock"}],
                "time_scope": "today",
            },
            "retrieval_result": {
                "retrieval_confidence": 0.92,
                "evidence_summary": [
                    {
                        "evidence_id": "D123456789",
                        "type": "news",
                        "fact": "The net inflow of main capital into 261102.SZ today reached RMB 1.2 billion.",
                    },
                    {
                        "evidence_id": "S123456789",
                        "type": "market_api",
                        "fact": "net_inflow=1200000000, total_inflow=1500000000, total_outflow=300000000, as_of=2023-04-05",
                    },
                ],
            },
            "statistical_result": {
                "overall_statistical_summary": {
                    "data_sufficiency": "high",
                    "answerable": True,
                    "should_abstain": False,
                    "summary": "The net inflow of main capital into 261102.SZ today was RMB 1.2 billion.",
                },
            },
            "sentiment_result": {
                "overall_sentiment": {"label": "neutral", "confidence": 0.85},
                "market_sentiment": {"label": "neutral", "confidence": 0.85},
            },
        },
        "output": {
            "answer": "The net inflow of main capital into 261102.SZ today was RMB 1.2 billion. The structured market data shows total inflow of RMB 1.5 billion and total outflow of RMB 300 million, which gives a net inflow of RMB 1.2 billion. The available news evidence is consistent with this figure.",
            "key_points": [
                "Main capital net inflow: RMB 1.2 billion.",
                "Total inflow was RMB 1.5 billion and total outflow was RMB 300 million.",
                "The answer is supported by both news evidence and structured market API evidence.",
                "Sentiment is neutral and the statistical result says the query is answerable.",
            ],
            "evidence_used": ["D123456789", "S123456789"],
            "limitations": ["Only today's provided capital-flow evidence was used."],
            "risk_disclaimer": "This answer is based only on the provided evidence and is not investment advice.",
        },
    },
    {
        "input": {
            "query": "国企ETF中银今天异动主要受什么影响？",
            "nlu_result": {
                "question_style": "why",
                "product_type": {"label": "etf"},
                "intent_labels": ["analyze", "cause"],
                "entities": ["国企ETF中银"],
                "time_scope": "today",
                "risk_flags": ["market_volatility"],
            },
            "retrieval_result": {
                "retrieval_confidence": 0.88,
                "evidence_summary": [
                    {
                        "evidence_id": "DOC_123456",
                        "type": "news",
                        "fact": "Policy changes favoring state-owned enterprises improved investor confidence.",
                    },
                    {
                        "evidence_id": "DOC_123457",
                        "type": "research_note",
                        "fact": "Research note links today's strength to positive SOE sentiment and policy reform.",
                    },
                    {
                        "evidence_id": "STRUCT_123456",
                        "type": "market_api",
                        "fact": "price=1.25, change_1d=0.03, change_5d=0.12, volume=15000000",
                    },
                    {
                        "evidence_id": "STRUCT_123457",
                        "type": "research_note",
                        "fact": "performance=positive, sentiment=positive, risk=moderate",
                    },
                ],
            },
            "statistical_result": {
                "price_statistics": {
                    "volume_change_signal": "increase",
                    "trend_signal": "positive",
                    "technical_summary": "Positive trend with increased volume.",
                },
                "news_statistics": {
                    "document_count": 2,
                    "dominant_news_signal": "positive",
                    "news_summary": "Positive sentiment around policy changes and investor confidence.",
                },
                "risk_statistics": {
                    "risk_level": "moderate",
                    "main_risk_factors": ["market volatility", "policy uncertainty"],
                    "risk_summary": "Moderate risk due to market volatility and policy uncertainty.",
                },
                "overall_statistical_summary": {
                    "data_sufficiency": "partial",
                    "overall_signal": "positive",
                    "answerable": True,
                    "should_abstain": False,
                    "summary": "The China SOE ETF showed a positive movement today due to policy changes and increased investor confidence.",
                },
            },
            "sentiment_result": {
                "overall_sentiment": {"label": "neutral", "score": 0.15, "confidence": 0.85},
                "news_sentiment": {"label": "positive", "score": 0.65, "confidence": 0.9},
                "market_sentiment": {"label": "positive", "score": 0.75, "confidence": 0.85},
                "risk_sentiment": {"label": "moderate", "confidence": 0.75},
            },
        },
        "output": {
            "answer": "从给定证据看，国企ETF中银今天异动主要受到政策预期改善、国企板块情绪升温和资金流入三方面影响。新闻和研报都提到近期支持国企的政策变化提升了投资者信心，统计结果也显示价格趋势为 positive、成交量信号为 increase，新闻统计的主导信号为 positive。不过风险统计为 moderate，主要风险来自市场波动和政策不确定性，因此更适合解读为事件驱动下的阶段性异动。",
            "key_points": [
                "主要驱动因素是国企相关政策变化和投资者信心改善。",
                "新闻和研报情绪均偏 positive，市场情绪也为 positive。",
                "价格统计显示 trend_signal 为 positive，成交量信号为 increase。",
                "风险水平为 moderate，需关注市场波动和政策不确定性。",
            ],
            "evidence_used": ["DOC_123456", "DOC_123457", "STRUCT_123456", "STRUCT_123457"],
            "limitations": ["当前数据覆盖为 partial", "缺少基金 NAV、费率、估值等补充数据"],
            "risk_disclaimer": "以上内容仅基于给定证据生成，不构成投资建议或确定性买卖结论。",
        },
    },
]


NEXT_QUESTION_FEW_SHOTS: list[dict[str, Any]] = [
    {
        "input": {
            "query": "How much main capital flowed into 261102.SZ today?",
            "nlu_result": {"entities": [{"entity": "261102.SZ", "type": "stock"}], "intent_labels": ["capital_flow"]},
            "statistical_result": {
                "overall_statistical_summary": {
                    "summary": "The net inflow of main capital into 261102.SZ today was RMB 1.2 billion."
                }
            },
            "sentiment_result": {"overall_sentiment": {"label": "neutral", "confidence": 0.85}},
        },
        "output": {
            "predictions": [
                {"question": "How does today's main capital inflow compare with the previous trading day?", "score": 0.92, "reason": "capital_flow_trend_followup"},
                {"question": "Did retail or institutional funds drive the move in 261102.SZ today?", "score": 0.86, "reason": "fund_flow_breakdown_followup"},
                {"question": "How did 261102.SZ's price react to the RMB 1.2 billion net inflow?", "score": 0.8, "reason": "price_reaction_followup"},
            ]
        },
    },
    {
        "input": {
            "query": "国企ETF中银今天异动主要受什么影响？",
            "nlu_result": {"entities": ["国企ETF中银"], "intent_labels": ["analyze", "cause"], "risk_flags": ["market_volatility"]},
            "statistical_result": {
                "price_statistics": {"trend_signal": "positive", "volume_change_signal": "increase"},
                "risk_statistics": {"risk_level": "moderate", "main_risk_factors": ["market volatility", "policy uncertainty"]},
                "overall_statistical_summary": {"overall_signal": "positive"},
            },
            "sentiment_result": {
                "news_sentiment": {"label": "positive", "confidence": 0.9},
                "market_sentiment": {"label": "positive", "confidence": 0.85},
            },
        },
        "output": {
            "predictions": [
                {"question": "国企ETF中银这次异动能持续多久？", "score": 0.92, "reason": "trend_continuation_followup"},
                {"question": "国企ETF中银后续最需要关注哪些政策信号？", "score": 0.86, "reason": "policy_risk_followup"},
                {"question": "国企ETF中银和同类国企ETF相比表现强在哪里？", "score": 0.8, "reason": "peer_comparison_followup"},
            ]
        },
    },
]


def strip_history(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_history(val) for key, val in value.items() if key != "history"}
    if isinstance(value, list):
        return [strip_history(item) for item in value]
    return value


def _short_text(value: Any, limit: int = 280) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _question_style_from_payload(payload: dict[str, Any]) -> str:
    nlu = payload.get("nlu_result") if isinstance(payload.get("nlu_result"), dict) else {}
    style = str(nlu.get("question_style") or "").strip().lower()
    if style in {"fact", "why", "advice", "compare", "forecast"}:
        return style
    query = str(payload.get("query") or "").lower()
    if any(term in query for term in ("compare", "which", "vs", "versus", "哪个", "对比", "相比")):
        return "compare"
    if any(term in query for term in ("why", "driven", "cause", "impact", "为什么", "原因", "影响", "异动")):
        return "why"
    if any(term in query for term in ("suitable", "should i", "advice", "适合", "能不能买", "定投")):
        return "advice"
    if any(term in query for term in ("forecast", "upside", "downside", "上涨", "下跌", "还能", "压力")):
        return "forecast"
    return "fact"


def _clean_warnings(warnings: Any) -> list[str]:
    if not isinstance(warnings, list):
        return []
    cleaned = []
    for warning in warnings:
        text = str(warning)
        if "synthetic" in text.lower():
            continue
        cleaned.append(text)
    return cleaned


def _sanitize_prompt_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_prompt_value(val)
            for key, val in value.items()
            if key not in {"metadata", "quality_flags", "is_synthetic"} and val not in (None, [], {}, "")
        }
    if isinstance(value, list):
        return [_sanitize_prompt_value(item) for item in value if item not in (None, [], {}, "")]
    if isinstance(value, str):
        text = value
        text = re.sub(r"synthetic evidence", "provided evidence", text, flags=re.IGNORECASE)
        text = re.sub(r"synthetic sources", "provided sources", text, flags=re.IGNORECASE)
        text = re.sub(r"synthetic source", "provided source", text, flags=re.IGNORECASE)
        text = re.sub(r"synthetic", "provided", text, flags=re.IGNORECASE)
        return text
    return value


def _compact_nlu(nlu_result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(nlu_result.get(key))
        for key in (
            "question_style",
            "product_type",
            "intent_labels",
            "topic_labels",
            "entities",
            "keywords",
            "time_scope",
            "forecast_horizon",
            "sentiment_of_user",
            "risk_flags",
            "missing_slots",
        )
        if nlu_result.get(key) not in (None, [], {}, "")
    }


def _compact_retrieval(retrieval_result: dict[str, Any]) -> dict[str, Any]:
    evidence_summary: list[dict[str, Any]] = []
    for doc in (retrieval_result.get("documents") or [])[:5]:
        if not isinstance(doc, dict):
            continue
        fact = doc.get("text_excerpt") or doc.get("summary") or doc.get("title") or doc.get("body")
        evidence_summary.append(
            {
                "evidence_id": doc.get("evidence_id"),
                "type": doc.get("source_type"),
                "fact": _short_text(fact),
            }
        )
    for row in (retrieval_result.get("structured_data") or [])[:5]:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        compact_payload = {
            key: value
            for key, value in payload.items()
            if key != "history" and not isinstance(value, (list, dict))
        }
        evidence_summary.append(
            {
                "evidence_id": row.get("evidence_id"),
                "type": row.get("source_type"),
                "as_of": row.get("as_of"),
                "fact": compact_payload or _short_text(row.get("source_reference")),
            }
        )
    return {
        "retrieval_confidence": retrieval_result.get("retrieval_confidence"),
        "warnings": _clean_warnings(retrieval_result.get("warnings")),
        "evidence_summary": [item for item in evidence_summary if item.get("evidence_id") or item.get("fact")],
    }


def _compact_statistical(statistical_result: dict[str, Any]) -> dict[str, Any]:
    keep_sections = (
        "retrieval_analysis_summary",
        "overall_statistical_summary",
        "price_statistics",
        "fundamental_statistics",
        "valuation_statistics",
        "news_statistics",
        "risk_statistics",
        "fund_statistics",
        "macro_statistics",
    )
    compact: dict[str, Any] = {}
    for section in keep_sections:
        value = statistical_result.get(section)
        if not isinstance(value, dict):
            continue
        compact[section] = {
            key: val
            for key, val in value.items()
            if key
            in {
                "available",
                "summary",
                "data_sufficiency",
                "overall_signal",
                "answerable",
                "should_abstain",
                "symbol",
                "close",
                "pct_change_1d",
                "pct_change_5d",
                "trend_signal",
                "volume_change_signal",
                "technical_summary",
                "roe",
                "eps",
                "grossprofit_margin",
                "profitability_signal",
                "valuation_signal",
                "valuation_summary",
                "document_count",
                "dominant_news_signal",
                "news_summary",
                "risk_level",
                "main_risk_factors",
                "risk_summary",
                "nav",
                "nav_change_1d",
                "liquidity_signal",
                "fund_summary",
                "indicator_name",
                "latest_value",
                "direction_signal",
                "market_impact_signal",
                "macro_summary",
                "market_signal",
                "fundamental_signal",
                "macro_signal",
                "data_readiness",
            }
            and val not in (None, [], {}, "")
        }
    return compact


def _compact_sentiment(sentiment_result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(sentiment_result.get(key))
        for key in (
            "overall_sentiment",
            "news_sentiment",
            "market_sentiment",
            "risk_sentiment",
            "user_sentiment",
        )
        if sentiment_result.get(key) not in (None, [], {}, "")
    }


def compact_payload(record: dict[str, Any]) -> dict[str, Any]:
    nlu_result = deepcopy(record.get("nlu_result") or {})
    retrieval_result = deepcopy(record.get("retrieval_result") or {})
    statistical_result = deepcopy(record.get("statistical_result") or {})
    if not statistical_result and isinstance(retrieval_result, dict) and retrieval_result.get("analysis_summary"):
        statistical_result = {"retrieval_analysis_summary": retrieval_result.get("analysis_summary")}
    sentiment_result = deepcopy(record.get("sentiment_result") or {})
    payload = {
        "query": record.get("query") or record.get("raw_query") or "",
        "nlu_result": _compact_nlu(nlu_result) if isinstance(nlu_result, dict) else {},
        "retrieval_result": _compact_retrieval(retrieval_result) if isinstance(retrieval_result, dict) else {},
        "statistical_result": _compact_statistical(statistical_result) if isinstance(statistical_result, dict) else {},
        "sentiment_result": _compact_sentiment(sentiment_result) if isinstance(sentiment_result, dict) else {},
    }
    return _sanitize_prompt_value(payload)


def _evidence_ids(payload: dict[str, Any], limit: int = 6) -> list[str]:
    retrieval = payload.get("retrieval_result") if isinstance(payload.get("retrieval_result"), dict) else {}
    ids: list[str] = []
    for item in retrieval.get("evidence_summary") or []:
        if isinstance(item, dict) and item.get("evidence_id"):
            ids.append(str(item["evidence_id"]))
        if len(ids) >= limit:
            break
    return ids


def _overall_summary(payload: dict[str, Any]) -> str:
    stat = payload.get("statistical_result") if isinstance(payload.get("statistical_result"), dict) else {}
    overall = stat.get("overall_statistical_summary") if isinstance(stat.get("overall_statistical_summary"), dict) else {}
    if overall.get("summary") or overall.get("overall_signal"):
        return str(overall.get("summary") or overall.get("overall_signal") or "").strip()
    retrieval_summary = stat.get("retrieval_analysis_summary") if isinstance(stat.get("retrieval_analysis_summary"), dict) else {}
    signals = [
        retrieval_summary.get("market_signal"),
        retrieval_summary.get("fundamental_signal"),
        retrieval_summary.get("macro_signal"),
    ]
    signals = [signal for signal in signals if signal not in (None, [], {}, "")]
    if signals:
        return _short_text(signals, limit=360)
    return ""


def _key_points_from_payload(payload: dict[str, Any], *, zh: bool) -> list[str]:
    stat = payload.get("statistical_result") if isinstance(payload.get("statistical_result"), dict) else {}
    sentiment = payload.get("sentiment_result") if isinstance(payload.get("sentiment_result"), dict) else {}
    points: list[str] = []
    overall = stat.get("overall_statistical_summary") if isinstance(stat.get("overall_statistical_summary"), dict) else {}
    if overall.get("overall_signal"):
        points.append(("整体统计信号：" if zh else "Overall statistical signal: ") + str(overall["overall_signal"]))
    retrieval_summary = stat.get("retrieval_analysis_summary") if isinstance(stat.get("retrieval_analysis_summary"), dict) else {}
    if retrieval_summary.get("data_readiness"):
        points.append(("数据就绪度：" if zh else "Data readiness: ") + _short_text(retrieval_summary["data_readiness"], limit=180))
    price = stat.get("price_statistics") if isinstance(stat.get("price_statistics"), dict) else {}
    if price.get("trend_signal"):
        points.append(("价格趋势信号：" if zh else "Price trend signal: ") + str(price["trend_signal"]))
    valuation = stat.get("valuation_statistics") if isinstance(stat.get("valuation_statistics"), dict) else {}
    if valuation.get("valuation_signal"):
        points.append(("估值信号：" if zh else "Valuation signal: ") + str(valuation["valuation_signal"]))
    fund = stat.get("fund_statistics") if isinstance(stat.get("fund_statistics"), dict) else {}
    if fund.get("fund_summary"):
        points.append(("基金数据：" if zh else "Fund data: ") + str(fund["fund_summary"]))
    risk = stat.get("risk_statistics") if isinstance(stat.get("risk_statistics"), dict) else {}
    if risk.get("risk_summary"):
        points.append(("风险提示：" if zh else "Risk note: ") + str(risk["risk_summary"]))
    overall_sentiment = sentiment.get("overall_sentiment")
    if isinstance(overall_sentiment, dict) and overall_sentiment.get("label"):
        points.append(("情绪信号：" if zh else "Sentiment signal: ") + str(overall_sentiment["label"]))
    return points[:5] or (["当前证据可支持初步回答。"] if zh else ["The current evidence supports a preliminary answer."])


def _limitations_from_payload(payload: dict[str, Any], *, zh: bool) -> list[str]:
    retrieval = payload.get("retrieval_result") if isinstance(payload.get("retrieval_result"), dict) else {}
    stat = payload.get("statistical_result") if isinstance(payload.get("statistical_result"), dict) else {}
    limitations = [str(item) for item in retrieval.get("warnings") or []]
    overall = stat.get("overall_statistical_summary") if isinstance(stat.get("overall_statistical_summary"), dict) else {}
    if overall.get("data_sufficiency") in {"partial", "low"}:
        limitations.append(("数据覆盖不完整" if zh else "Data coverage is incomplete"))
    if overall.get("should_abstain"):
        limitations.append(("统计结果提示应谨慎回答" if zh else "The statistical result suggests caution or abstention"))
    return limitations[:4]


def _answer_text_for_style(style: str, payload: dict[str, Any], *, zh: bool) -> str:
    query = str(payload.get("query") or "")
    summary = _overall_summary(payload)
    points = _key_points_from_payload(payload, zh=zh)
    if zh:
        if style == "why":
            return f"从给定证据看，{query} 的主要解释是：{summary}。关键依据包括：{'；'.join(points[:3])}。"
        if style == "advice":
            return f"基于当前证据，这个问题可以做条件性判断：{summary}。如果投资期限和风险承受能力匹配，可以继续评估；如果无法承受相关波动，应保持谨慎。"
        if style == "compare":
            return f"从稳定性、风险和证据摘要看，结论是：{summary}。比较类问题应优先看波动、风险事件、估值和基本面差异。"
        if style == "forecast":
            return f"基于现有统计和情绪信号，{summary}。这只能说明当前证据下的倾向，不能作为确定性涨跌预测。"
        return f"根据当前证据，{summary or '可以给出初步事实性回答'}。"
    if style == "why":
        return f"Based on the provided evidence, the latest move is mainly explained by: {summary}. Key evidence includes: {'; '.join(points[:3])}."
    if style == "advice":
        return f"Based on the current evidence, this is a conditional suitability call: {summary}. It may be suitable only if the user's horizon and risk tolerance match the product's risk profile."
    if style == "compare":
        return f"Based on stability, risk, and the evidence summary, the comparison points to: {summary}. For comparison questions, volatility, risk events, valuation, and fundamentals should carry the most weight."
    if style == "forecast":
        return f"Based on the available statistical and sentiment signals, {summary}. This is an evidence-based tendency, not a guaranteed price forecast."
    return f"Based on the current evidence, {summary or 'the question can be answered with the provided facts'}."


def _next_questions_for_style(style: str, payload: dict[str, Any], *, zh: bool) -> list[dict[str, Any]]:
    nlu = payload.get("nlu_result") if isinstance(payload.get("nlu_result"), dict) else {}
    entities = nlu.get("entities") or []
    if isinstance(entities, list) and entities:
        first = entities[0]
        if isinstance(first, dict):
            target = first.get("canonical_name") or first.get("entity") or first.get("symbol")
        else:
            target = first
    else:
        target = "这个标的" if zh else "this asset"
    target = str(target)
    if zh:
        templates = {
            "why": [f"{target}这个影响会持续多久？", f"{target}后续还要看哪些催化因素？", f"{target}当前最大的风险是什么？"],
            "advice": [f"{target}更适合什么风险偏好的投资者？", f"{target}适合长期持有还是阶段性配置？", f"{target}需要设置哪些止损或观察指标？"],
            "compare": [f"{target}和同类标的相比估值如何？", f"{target}哪个风险更低？", f"{target}后续表现差异主要看什么？"],
            "forecast": [f"{target}继续上涨/下跌需要哪些条件？", f"{target}短期最关键的压力位或风险是什么？", f"{target}基本面能支撑当前趋势吗？"],
            "fact": [f"{target}这个数据和上一期相比如何？", f"{target}相关指标的来源可靠吗？", f"{target}还需要补充哪些数据？"],
        }
    else:
        templates = {
            "why": [f"How long could this driver affect {target}?", f"What catalysts should I watch next for {target}?", f"What is the biggest risk for {target} now?"],
            "advice": [f"What risk profile is {target} suitable for?", f"Is {target} better for long-term holding or tactical allocation?", f"What stop-loss or monitoring indicators should I use for {target}?"],
            "compare": [f"How does {target} compare with peers on valuation?", f"Which option has lower risk?", f"What metrics matter most for the comparison?"],
            "forecast": [f"What conditions would support further upside or downside for {target}?", f"What is the key short-term risk for {target}?", f"Do fundamentals support the current trend in {target}?"],
            "fact": [f"How does this data compare with the previous period for {target}?", f"Which evidence source supports this number?", f"What other data should I check for {target}?"],
        }
    return [
        {"question": question, "score": round(0.92 - index * 0.06, 4), "reason": f"{style}_followup"}
        for index, question in enumerate(templates.get(style, templates["fact"]))
    ]


def _make_answer_shot_from_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = compact_payload(record)
    style = _question_style_from_payload(payload)
    zh = _is_zh(str(payload.get("query") or ""))
    output = {
        "answer": _answer_text_for_style(style, payload, zh=zh),
        "key_points": _key_points_from_payload(payload, zh=zh),
        "evidence_used": _evidence_ids(payload),
        "limitations": _limitations_from_payload(payload, zh=zh),
        "risk_disclaimer": "以上内容仅基于给定证据生成，不构成投资建议或确定性买卖结论。"
        if zh
        else "This answer is based only on the provided evidence and is not investment advice.",
    }
    return {"input": payload, "output": output}


def _make_next_question_shot_from_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = compact_payload(record)
    style = _question_style_from_payload(payload)
    zh = _is_zh(str(payload.get("query") or ""))
    return {
        "input": {
            "query": payload.get("query"),
            "nlu_result": payload.get("nlu_result"),
            "statistical_result": payload.get("statistical_result"),
            "sentiment_result": payload.get("sentiment_result"),
        },
        "output": {"predictions": _next_questions_for_style(style, payload, zh=zh)},
    }


def load_few_shot_bank(path: Path, *, per_style_per_language: int = 1) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    answer_bank: dict[str, list[dict[str, Any]]] = {}
    next_bank: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return answer_bank, next_bank
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            payload = compact_payload(record)
            style = _question_style_from_payload(payload)
            if style not in {"fact", "why", "advice", "compare", "forecast"}:
                continue
            lang = "zh" if _is_zh(str(payload.get("query") or "")) else "en"
            key = (style, lang)
            if sum(1 for item in seen if item == key) >= per_style_per_language or key in seen:
                continue
            answer_bank.setdefault(style, []).append(_make_answer_shot_from_record(record))
            next_bank.setdefault(style, []).append(_make_next_question_shot_from_record(record))
            seen.add(key)
            if len(seen) >= 10:
                break
    return answer_bank, next_bank


def select_few_shots(
    payload: dict[str, Any],
    bank: dict[str, list[dict[str, Any]]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    style = _question_style_from_payload(payload)
    selected = bank.get(style) or []
    return selected[:2] if selected else fallback[:2]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
            except ImportError as exc:
                raise ValueError(
                    "model output was not strict JSON and json-repair is not installed. "
                    "Install project requirements with `pip install -r requirements.txt`."
                ) from exc
            repaired = repair_json(candidate)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "model output could not be repaired into a valid JSON object"
                ) from exc
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"model output did not contain a valid JSON object: {text[:500]}")


def normalize_answer(output: dict[str, Any], retrieval_result: dict[str, Any]) -> dict[str, Any]:
    docs = retrieval_result.get("documents") if isinstance(retrieval_result, dict) else []
    structured = retrieval_result.get("structured_data") if isinstance(retrieval_result, dict) else []
    fallback_evidence = [
        str(item.get("evidence_id"))
        for item in list(docs or []) + list(structured or [])
        if isinstance(item, dict) and item.get("evidence_id")
    ][:6]
    answer = {
        "model_status": "real_model",
        "model_name": "",
        "answer": str(output.get("answer") or "").strip(),
        "key_points": output.get("key_points") if isinstance(output.get("key_points"), list) else [],
        "evidence_used": output.get("evidence_used") if isinstance(output.get("evidence_used"), list) else fallback_evidence,
        "limitations": output.get("limitations") if isinstance(output.get("limitations"), list) else [],
        "risk_disclaimer": str(output.get("risk_disclaimer") or "").strip(),
    }
    for key in ("key_points", "evidence_used", "limitations"):
        answer[key] = [str(value).strip() for value in answer[key] if str(value).strip()]
    if not answer["answer"]:
        raise ValueError("answer model returned empty answer")
    if not answer["risk_disclaimer"]:
        answer["risk_disclaimer"] = "以上内容仅基于给定证据生成，不构成投资建议或确定性买卖结论。"
    return answer


def normalize_next_questions(output: dict[str, Any], query: str) -> dict[str, Any]:
    predictions = output.get("predictions") if isinstance(output.get("predictions"), list) else []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(predictions):
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            score = item.get("score")
            reason = str(item.get("reason") or "model_prediction").strip()
        else:
            question = str(item).strip()
            score = 0.9 - index * 0.05
            reason = "model_prediction"
        if not question:
            continue
        try:
            score_float = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score_float = 0.9 - index * 0.05
        normalized.append({"question": question, "score": round(score_float, 4), "reason": reason})
    fallback = "还需要进一步关注哪些数据？" if re.search(r"[\u4e00-\u9fff]", query or "") else "What data should I check next?"
    while len(normalized) < 3:
        normalized.append({"question": fallback, "score": round(0.75 - len(normalized) * 0.05, 4), "reason": "fallback"})
    return {
        "model_status": "real_model",
        "model_name": "",
        "predictions": normalized[:3],
    }


def _model_cache_names(model_id: str) -> list[str]:
    normalized = model_id.strip().strip("/")
    names = [normalized]
    if "/" in normalized:
        owner, name = normalized.split("/", 1)
        names.extend(
            [
                name,
                normalized.replace("/", "__"),
                f"models--{owner}--{name}",
            ]
        )
    return list(dict.fromkeys(names))


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    direct_markers = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
    if any((path / marker).exists() for marker in direct_markers):
        return True
    refs_main = path / "refs" / "main"
    snapshots = path / "snapshots"
    return refs_main.exists() and snapshots.is_dir()


def resolve_model_path(model_id: str, *, models_dir: Path | None = None) -> str:
    direct = Path(model_id).expanduser()
    if direct.exists():
        logger.info("Using explicit local model path: %s", direct)
        return str(direct)

    search_roots: list[Path] = []
    if models_dir is not None:
        search_roots.append(models_dir.expanduser())
    for env_name in ("LLM_MODELS_DIR", "QI_LLM_MODELS_DIR"):
        value = os.getenv(env_name)
        if value:
            search_roots.append(Path(value).expanduser())
    search_roots.append(DEFAULT_LLM_MODELS_DIR)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        search_roots.append(Path(hf_home).expanduser() / "hub")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if transformers_cache:
        search_roots.append(Path(transformers_cache).expanduser())

    for root in list(dict.fromkeys(search_roots)):
        for name in _model_cache_names(model_id):
            candidate = root / name
            if _looks_like_hf_model_dir(candidate):
                logger.info("Using cached local model path for %s: %s", model_id, candidate)
                return str(candidate)

    logger.info("No local model directory found for %s; using HuggingFace model id/cache", model_id)
    return model_id


class ChatModel:
    def __init__(self, model_id: str, *, device_map: str, dtype: str, models_dir: Path | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing LLM runtime dependencies. Install the project requirements first: "
                "pip install -r requirements.txt"
            ) from exc

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.model_id = model_id
        self.resolved_model = resolve_model_path(model_id, models_dir=models_dir)
        token = hf_token()
        token_kwargs = {"token": token} if token else {}
        if token:
            logger.info("HF_TOKEN is configured; authenticated HuggingFace Hub access is enabled")
        elif self.resolved_model == model_id:
            logger.warning(
                "HF_TOKEN is not configured. HuggingFace Hub requests may be rate-limited. "
                "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN to enable authenticated downloads."
            )
        logger.info("Loading tokenizer: model=%s resolved=%s", model_id, self.resolved_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_model, trust_remote_code=True, **token_kwargs)
        logger.info("Loading model: model=%s resolved=%s device_map=%s dtype=%s", model_id, self.resolved_model, device_map, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.resolved_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **token_kwargs,
        )
        logger.info("Loaded model: model=%s resolved=%s", model_id, self.resolved_model)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        chunks: list[str] = []
        for message in messages:
            role = message["role"].upper()
            chunks.append(f"{role}:\n{message['content']}")
        chunks.append("ASSISTANT:\n")
        return "\n\n".join(chunks)

    def generate_text(self, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float) -> str:
        text = self._format_messages(messages)
        logger.info(
            "Generating response: model=%s messages=%d prompt_chars=%d max_new_tokens=%d temperature=%.3f",
            self.model_id,
            len(messages),
            len(text),
            max_new_tokens,
            temperature,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        json_retries: int = 1,
    ) -> dict[str, Any]:
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for attempt in range(json_retries + 1):
            generated_text = self.generate_text(
                attempt_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            try:
                parsed = extract_json_object(generated_text)
                logger.info("Generated valid JSON: model=%s keys=%s attempt=%d", self.model_id, sorted(parsed.keys()), attempt + 1)
                return parsed
            except ValueError as exc:
                last_error = exc
                logger.warning("Invalid JSON from model=%s attempt=%d/%d: %s", self.model_id, attempt + 1, json_retries + 1, exc)
                if attempt >= json_retries:
                    break
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": generated_text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not valid JSON. "
                            "Return only one strict JSON object matching the requested schema. "
                            "Do not include markdown fences, explanations, or extra text."
                        ),
                    },
                ]
        raise ValueError(f"model failed to produce valid JSON after {json_retries + 1} attempts") from last_error


def make_messages(system_prompt: str, few_shots: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for shot in few_shots:
        messages.append({"role": "user", "content": json.dumps(shot["input"], ensure_ascii=False, separators=(",", ":"))})
        messages.append({"role": "assistant", "content": json.dumps(shot["output"], ensure_ascii=False, separators=(",", ":"))})
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})
    return messages


def build_frontend_response(
    record: dict[str, Any],
    answer_output: dict[str, Any],
    next_question_output: dict[str, Any],
    *,
    answer_model: str,
    next_question_model: str,
) -> dict[str, Any]:
    result = deepcopy(record)
    query = result.get("query") or result.get("raw_query") or (result.get("nlu_result") or {}).get("raw_query") or ""
    retrieval_result = result.get("retrieval_result") if isinstance(result.get("retrieval_result"), dict) else {}
    answer = normalize_answer(answer_output, retrieval_result)
    answer["model_name"] = answer_model
    next_questions = normalize_next_questions(next_question_output, query)
    next_questions["model_name"] = next_question_model

    result["schema_version"] = result.get("schema_version") or "frontend_pipeline_response_v1"
    result["status"] = result.get("status") or "ok"
    result["request"] = result.get("request") or {
        "query_id": (result.get("nlu_result") or {}).get("query_id") or retrieval_result.get("query_id"),
        "raw_query": query,
        "language": "zh" if re.search(r"[\u4e00-\u9fff]", query or "") else "en",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "real_model_frontend_integration",
    }
    result["answer_generation"] = answer
    result["next_question_prediction"] = next_questions
    return result


class LLMResponseRuntime:
    def __init__(
        self,
        *,
        answer_model: str,
        next_question_model: str,
        models_dir: Path,
        few_shot_source: Path,
        device_map: str,
        dtype: str,
        temperature: float,
        answer_max_new_tokens: int,
        next_max_new_tokens: int,
        json_retries: int,
    ) -> None:
        logger.info("Initializing LLM response runtime")
        self.answer_model_name = answer_model
        self.next_question_model_name = next_question_model
        self.temperature = temperature
        self.answer_max_new_tokens = answer_max_new_tokens
        self.next_max_new_tokens = next_max_new_tokens
        self.json_retries = json_retries
        self.answer_few_shot_bank, self.next_question_few_shot_bank = load_few_shot_bank(few_shot_source)
        logger.info("Loaded few-shot bank: source=%s styles=%s", few_shot_source, sorted(self.answer_few_shot_bank.keys()))
        self.answer_model = ChatModel(answer_model, device_map=device_map, dtype=dtype, models_dir=models_dir)
        self.next_question_model = ChatModel(next_question_model, device_map=device_map, dtype=dtype, models_dir=models_dir)
        logger.info("LLM response runtime ready")

    def generate(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = compact_payload(record)
        style = _question_style_from_payload(payload)
        logger.info(
            "Prepared compact payload: style=%s evidence_items=%d has_statistical=%s has_sentiment=%s",
            style,
            len((payload.get("retrieval_result") or {}).get("evidence_summary") or []),
            bool(payload.get("statistical_result")),
            bool(payload.get("sentiment_result")),
        )
        answer_few_shots = select_few_shots(payload, self.answer_few_shot_bank, ANSWER_FEW_SHOTS)
        next_question_few_shots = select_few_shots(payload, self.next_question_few_shot_bank, NEXT_QUESTION_FEW_SHOTS)
        logger.info(
            "Selected few-shots: style=%s answer=%d next_question=%d",
            style,
            len(answer_few_shots),
            len(next_question_few_shots),
        )
        answer_output = self.answer_model.generate_json(
            make_messages(ANSWER_SYSTEM_PROMPT, answer_few_shots, payload),
            max_new_tokens=self.answer_max_new_tokens,
            temperature=self.temperature,
            json_retries=self.json_retries,
        )
        next_question_output = self.next_question_model.generate_json(
            make_messages(NEXT_QUESTION_SYSTEM_PROMPT, next_question_few_shots, payload),
            max_new_tokens=self.next_max_new_tokens,
            temperature=self.temperature,
            json_retries=self.json_retries,
        )
        return build_frontend_response(
            record,
            answer_output,
            next_question_output,
            answer_model=self.answer_model_name,
            next_question_model=self.next_question_model_name,
        )


def build_record_from_query(
    query: str,
    *,
    top_k: int,
    debug: bool,
    user_profile: dict[str, Any] | None = None,
    dialog_context: list[dict[str, Any]] | None = None,
    service: Any | None = None,
) -> dict[str, Any]:
    if service is None:
        from query_intelligence.service import build_default_service

        logger.info("Building default Query Intelligence service")
        service = build_default_service()
    logger.info("Running Query Intelligence pipeline: top_k=%d debug=%s query=%s", top_k, debug, query)
    result = service.run_pipeline(
        query=query,
        user_profile=user_profile or {},
        dialog_context=dialog_context or [],
        top_k=top_k,
        debug=debug,
    )
    logger.info(
        "Query Intelligence completed: query_id=%s documents=%d structured_data=%d",
        (result.get("nlu_result") or {}).get("query_id"),
        len((result.get("retrieval_result") or {}).get("documents") or []),
        len((result.get("retrieval_result") or {}).get("structured_data") or []),
    )
    return {
        "status": "ok",
        "query": query,
        "nlu_result": result["nlu_result"],
        "retrieval_result": result["retrieval_result"],
    }


def make_runtime(args: argparse.Namespace) -> LLMResponseRuntime:
    return LLMResponseRuntime(
        answer_model=args.answer_model,
        next_question_model=args.next_question_model,
        models_dir=args.models_dir,
        few_shot_source=args.few_shot_source,
        device_map=args.device_map,
        dtype=args.dtype,
        temperature=args.temperature,
        answer_max_new_tokens=args.answer_max_new_tokens,
        next_max_new_tokens=args.next_max_new_tokens,
        json_retries=args.json_retries,
    )


def run_service(args: argparse.Namespace) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise SystemExit(
            "Missing API runtime dependencies. Install the project requirements first: "
            "pip install -r requirements.txt"
        ) from exc

    runtime = make_runtime(args)
    app = FastAPI(title="Query Intelligence LLM Response Service")

    class RespondRequest(BaseModel):
        query: str | None = Field(default=None, description="Raw query. The service will run Query Intelligence first.")
        pipeline_result: dict[str, Any] | None = Field(
            default=None,
            description="Already computed Query Intelligence pipeline result.",
        )
        top_k: int = 20
        debug: bool = False
        user_profile: dict[str, Any] = Field(default_factory=dict)
        dialog_context: list[dict[str, Any]] = Field(default_factory=list)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "answer_model": args.answer_model,
            "next_question_model": args.next_question_model,
        }

    @app.post("/respond")
    def respond(request: RespondRequest) -> dict[str, Any]:
        try:
            if request.pipeline_result is not None:
                record = request.pipeline_result
            elif request.query:
                record = build_record_from_query(
                    request.query.strip(),
                    top_k=request.top_k,
                    debug=request.debug,
                    user_profile=request.user_profile,
                    dialog_context=request.dialog_context,
                )
            else:
                raise HTTPException(status_code=422, detail="Provide either query or pipeline_result")
            return runtime.generate(record)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("LLM response request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Starting LLM response service: host=%s port=%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


def load_record(path: Path) -> dict[str, Any]:
    if str(path) == "-":
        logger.info("Reading pipeline input from stdin")
        text = sys.stdin.read().strip()
        if not text:
            raise ValueError("stdin input is empty")
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            item = json.loads(first_line)
        if not isinstance(item, dict):
            raise ValueError("stdin input must be a JSON object")
        logger.info("Loaded pipeline input from stdin")
        return item
    logger.info("Reading pipeline input from file: %s", path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty input file: {path}")
    if path.suffix == ".jsonl":
        first = text.splitlines()[0]
        item = json.loads(first)
    else:
        item = json.loads(text)
    if not isinstance(item, dict):
        raise ValueError("input must be a JSON object or JSONL object per line")
    logger.info("Loaded pipeline input from file: %s", path)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-model frontend answer and next-question JSON.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--input", type=Path, help="Pipeline JSON/JSONL path. Use '-' to read one JSON object from stdin.")
    source.add_argument("--query", help="Run Query Intelligence first, then generate LLM response from that pipeline result.")
    parser.add_argument("--serve", action="store_true", help="Run a persistent FastAPI service and load models once at startup.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --serve mode.")
    parser.add_argument("--port", type=int, default=8010, help="Port for --serve mode.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path. Omit to print JSON to stdout.")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--next-question-model", default=DEFAULT_NEXT_QUESTION_MODEL)
    parser.add_argument("--models-dir", type=Path, default=default_models_dir(), help="Local LLM model directory checked before HuggingFace model IDs.")
    parser.add_argument("--few-shot-source", type=Path, default=DEFAULT_FEW_SHOT_SOURCE)
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval top-k when --query is used.")
    parser.add_argument("--debug", action="store_true", help="Enable Query Intelligence debug mode when --query is used.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--answer-max-new-tokens", type=int, default=700)
    parser.add_argument("--next-max-new-tokens", type=int, default=260)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--json-retries", type=int, default=1, help="Strict JSON retry count after repair/parsing failure.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.json_retries < 0:
        parser.error("--json-retries must be >= 0")
    if not args.serve and not (args.input or args.query):
        parser.error("one of --input, --query, or --serve is required")
    if args.serve and (args.input or args.query or args.output):
        parser.error("--serve cannot be combined with --input, --query, or --output")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting LLM response runner")

    if args.serve:
        run_service(args)
        return

    if args.query:
        record = build_record_from_query(args.query.strip(), top_k=args.top_k, debug=args.debug)
    else:
        record = load_record(args.input)
    response = make_runtime(args).generate(record)

    output_text = json.dumps(response, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        logger.info("Wrote LLM response JSON: %s", args.output)
    else:
        logger.info("Writing LLM response JSON to stdout")
        print(output_text)


if __name__ == "__main__":
    main()
