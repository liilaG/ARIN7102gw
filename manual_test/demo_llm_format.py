"""Demo: 从 RetrievalResult 生成 LLM 可消费的格式化内容。

用法:
    python manual_test/demo_llm_format.py
    python manual_test/demo_llm_format.py --query "茅台今天股价多少"
    python manual_test/demo_llm_format.py --input manual_test/output/20260426-XXXX-茅台今天股价多少/retrieval_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import build_default_service, clear_service_caches


# ── helpers ──────────────────────────────────────────────────────────
def _load_retrieval_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_output_dir() -> Path | None:
    output_dir = Path(__file__).resolve().parent / "output"
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "retrieval_result.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_pipeline(query: str) -> dict:
    clear_service_caches()
    svc = build_default_service(
        use_live_market=True,
        use_live_macro=True,
        use_live_news=True,
        use_live_announcement=False,
    )
    nlu = svc.analyze_query(query, debug=False)
    return svc.retrieve_evidence(nlu, debug=False)


# ── formatters ──────────────────────────────────────────────────────
def fmt_market_signal(ms: dict | list[dict] | None) -> str:
    if not ms:
        return "  （无市场技术信号 — 可能缺少历史数据）"
    if isinstance(ms, list):
        if not ms:
            return "  （无市场技术信号 — 可能缺少历史数据）"
        sections = []
        for item in ms:
            symbol = item.get("symbol") or item.get("canonical_name") or "未知标的"
            sections.append(f"  [{symbol}]")
            sections.append(fmt_market_signal(item))
        return "\n".join(sections)
    lines = [
        f"  趋势判断: {ms.get('trend_signal', 'N/A')}",
        f"  RSI(14): {ms.get('rsi_14', 'N/A')}",
        f"  MA5={ms.get('ma5', 'N/A')}  MA20={ms.get('ma20', 'N/A')}",
    ]
    macd = ms.get("macd") or {}
    if macd:
        lines.append(f"  MACD: DIF={macd.get('macd_line')}  DEA={macd.get('signal_line')}  柱={macd.get('histogram')}")
    boll = ms.get("bollinger") or {}
    if boll:
        lines.append(f"  布林带: 上轨={boll.get('upper')}  中轨={boll.get('middle')}  下轨={boll.get('lower')}  bandwidth={boll.get('bandwidth')}")
    if ms.get("volatility_20d") is not None:
        lines.append(f"  20日波动率: {ms['volatility_20d']:.2f}%")
    pnd = ms.get("pct_change_nd") or {}
    if pnd:
        parts = [f"{k}={v}" for k, v in pnd.items() if v is not None]
        lines.append(f"  多日涨跌: {', '.join(parts)}")
    pma = ms.get("price_vs_ma") or {}
    if pma:
        lines.append(f"  价格位置: MA5上方={pma.get('above_ma5')}  MA20上方={pma.get('above_ma20')}")
    return "\n".join(lines)


def fmt_fundamental_signal(fs: dict | list[dict] | None) -> str:
    if not fs:
        return "  （无基本面信号）"
    if isinstance(fs, list):
        if not fs:
            return "  （无基本面信号）"
        sections = []
        for item in fs:
            symbol = item.get("symbol") or "未知标的"
            sections.append(f"  [{symbol}]")
            sections.append(fmt_fundamental_signal(item))
        return "\n".join(sections)
    lines = [
        f"  PE(TTM)={fs.get('pe_ttm', 'N/A')}  PB={fs.get('pb', 'N/A')}  ROE={fs.get('roe', 'N/A')}",
        f"  估值评估: {fs.get('valuation_assessment', 'N/A')}",
    ]
    return "\n".join(lines)


def fmt_macro_signal(ms: dict | None) -> str:
    if not ms:
        return "  （无宏观信号）"
    lines = [f"  综合判断: {ms.get('overall', 'N/A')}"]
    for ind in ms.get("indicators") or []:
        name = ind.get("indicator_name") or ind.get("name", "?")
        value = ind.get("metric_value") if ind.get("metric_value") is not None else ind.get("value", "?")
        unit = ind.get("unit", "")
        direction = ind.get("direction", "?")
        lines.append(f"  - {name}: {value}{unit} → {direction}")
    return "\n".join(lines)


def fmt_data_readiness(dr: dict) -> str:
    flags = {k: ("✅" if v else "❌") for k, v in dr.items() if k.startswith("has_")}
    parts = [f"{k.replace('has_', '')}={v}" for k, v in flags.items()]
    return "  " + "  ".join(parts)


def fmt_news(documents: list[dict], limit: int = 5) -> str:
    news = [d for d in documents if d.get("source_type") == "news"]
    if not news:
        return "  （无新闻）"
    lines = []
    for i, doc in enumerate(news[:limit], 1):
        title = doc.get("title", "无标题")
        summary = (doc.get("summary") or "")[:80]
        lines.append(f"  {i}. {title}")
        if summary:
            lines.append(f"     摘要: {summary}")
    return "\n".join(lines)


def fmt_structured_data(items: list[dict]) -> str:
    if not items:
        return "  （无结构化数据）"
    lines = []
    for item in items:
        st = item.get("source_type", "?")
        p = item.get("payload", {})
        if st == "market_api":
            pct = p.get('pct_change_1d')
            pct_str = f"{pct}%" if pct is not None else "N/A"
            lines.append(f"  行情: {p.get('symbol')} 收盘={p.get('close')} 涨跌幅={pct_str}")
        elif st == "fundamental_sql":
            lines.append(f"  基本面: {p.get('symbol')} ROE={p.get('roe')} PE={p.get('pe_ttm')} PB={p.get('pb')}")
        elif st == "industry_sql":
            ind_pct = p.get('pct_change')
            ind_pct_str = f"{ind_pct}%" if ind_pct is not None else "N/A"
            lines.append(f"  行业: {p.get('industry_name')} 涨跌幅={ind_pct_str} PE={p.get('pe')}")
        elif st in ("macro_indicator", "macro_sql"):
            lines.append(f"  宏观: {p.get('indicator_name')}={p.get('metric_value')}{p.get('unit', '')} ({p.get('metric_date', p.get('period', ''))})")
    return "\n".join(lines)


# ── main builder ────────────────────────────────────────────────────
def build_llm_context(rr: dict) -> str:
    """将 RetrievalResult 转换为 LLM 可消费的格式化文本。"""
    nlu_snap = rr.get("nlu_snapshot") or {}
    query = nlu_snap.get("raw_query") or nlu_snap.get("normalized_query") or rr.get("query_id", "?")
    summary = rr.get("analysis_summary") or {}
    readiness = summary.get("data_readiness") or {}
    documents = rr.get("documents") or []
    structured = rr.get("structured_data") or []
    coverage = rr.get("coverage") or {}

    sections = []

    # ── 1. 用户问题 & 数据可用性 ──
    sections.append("## 用户问题")
    sections.append(f"  {query}")
    sections.append("")
    sections.append("## 数据可用性")
    sections.append(fmt_data_readiness(readiness))
    sections.append(f"  覆盖维度: {', '.join(k for k, v in coverage.items() if v) or '无'}")
    sections.append("")

    # ── 2. 市场技术信号 ──
    if readiness.get("has_technical_indicators") or summary.get("market_signal"):
        sections.append("## 市场技术信号")
        sections.append(fmt_market_signal(summary.get("market_signal")))
        sections.append("")

    # ── 3. 基本面信号 ──
    if readiness.get("has_fundamentals") or summary.get("fundamental_signal"):
        sections.append("## 基本面信号")
        sections.append(fmt_fundamental_signal(summary.get("fundamental_signal")))
        sections.append("")

    # ── 4. 宏观信号 ──
    if readiness.get("has_macro") or summary.get("macro_signal"):
        sections.append("## 宏观经济信号")
        sections.append(fmt_macro_signal(summary.get("macro_signal")))
        sections.append("")

    # ── 5. 相关新闻 ──
    if readiness.get("has_news"):
        sections.append("## 相关新闻")
        sections.append(fmt_news(documents))
        sections.append("")

    # ── 6. 结构化数据摘要 ──
    sections.append("## 结构化数据摘要")
    sections.append(fmt_structured_data(structured))
    sections.append("")

    return "\n".join(sections)


def build_llm_prompt(rr: dict) -> str:
    """生成完整的 LLM prompt（含 system instruction + context）。"""
    context = build_llm_context(rr)
    nlu_snap = rr.get("nlu_snapshot") or {}
    query = nlu_snap.get("raw_query") or nlu_snap.get("normalized_query") or rr.get("query_id", "?")
    readiness = (rr.get("analysis_summary") or {}).get("data_readiness") or {}

    prompt = f"""你是一个专业的金融投资分析助手。请根据以下数据回答用户的问题。

要求:
- 优先使用「市场技术信号」和「基本面信号」中的预计算数据
- 如果 data_readiness 中某项为 false，请明确告知用户该维度数据暂不可用
- 不要编造数据，只使用下方提供的信息
- 回答应结构化，包含: 行情概况、技术面分析、基本面分析(如有)、宏观环境(如有)、风险提示

{context}

用户问题: {query}

请回答:"""
    return prompt


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Demo: 从 RetrievalResult 生成 LLM 格式化内容")
    parser.add_argument("--query", type=str, help="直接运行 pipeline 并格式化输出")
    parser.add_argument("--input", type=str, help="指定 retrieval_result.json 路径")
    parser.add_argument("--prompt", action="store_true", help="输出完整 LLM prompt（含 system instruction）")
    args = parser.parse_args()

    if args.input:
        rr = _load_retrieval_result(args.input)
    elif args.query:
        print(f"正在运行 pipeline: {args.query}\n")
        rr = _run_pipeline(args.query)
    else:
        latest = _latest_output_dir()
        if not latest:
            print("未找到输出目录，请使用 --query 或 --input 参数")
            return
        rr_path = latest / "retrieval_result.json"
        if not rr_path.exists():
            print(f"未找到 {rr_path}")
            return
        print(f"使用最近的输出: {latest.name}\n")
        rr = _load_retrieval_result(str(rr_path))

    if args.prompt:
        print(build_llm_prompt(rr))
    else:
        print(build_llm_context(rr))


if __name__ == "__main__":
    main()
