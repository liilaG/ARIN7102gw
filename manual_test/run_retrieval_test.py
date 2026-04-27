from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import build_default_service, clear_service_caches


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    args = _build_parser().parse_args()
    query = args.query.strip() if args.query else _prompt_query()
    if not query:
        raise SystemExit("query is empty")

    clear_service_caches()
    service = build_default_service(
        use_live_market=args.live_market,
        use_live_macro=args.live_macro,
        use_live_news=args.live_news,
        use_live_announcement=args.live_announcement,
    )
    print(f"  Live providers: market={args.live_market}  macro={args.live_macro}  news={args.live_news}  announcement={args.live_announcement}")

    # ---- Step 1: NLU ----
    nlu_result = service.analyze_query(query, debug=True)

    # ---- Step 2: Retrieval + Analysis ----
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=args.top_k, debug=True)

    # ---- Save outputs ----
    run_dir = _prepare_run_dir(query)
    (run_dir / "query.txt").write_text(query + "\n", encoding="utf-8")
    (run_dir / "nlu_result.json").write_text(json.dumps(nlu_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrieval_result.json").write_text(
        json.dumps(retrieval_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- Print summary ----
    _print_summary(query, nlu_result, retrieval_result, run_dir)


def _print_summary(query: str, nlu_result: dict, retrieval_result: dict, run_dir: Path) -> None:
    nlu = nlu_result if isinstance(nlu_result, dict) else nlu_result
    rr = retrieval_result if isinstance(retrieval_result, dict) else retrieval_result

    print("\n" + "=" * 60)
    print(f"  查询: {query}")
    print("=" * 60)

    # NLU summary
    print("\n【第1步 NLU】")
    print(f"  question_style : {nlu.get('question_style')}")
    print(f"  product_type   : {nlu.get('product_type', {}).get('label') if isinstance(nlu.get('product_type'), dict) else nlu.get('product_type')}")
    print(f"  intent_labels  : {[x['label'] for x in nlu.get('intent_labels', [])]}")
    print(f"  topic_labels   : {[x['label'] for x in nlu.get('topic_labels', [])]}")
    entities_str = ", ".join(f"{e.get('mention')}({e.get('symbol')})" for e in nlu.get('entities', []))
    print(f"  entities       : [{entities_str}]")
    print(f"  source_plan    : {nlu.get('source_plan')}")
    print(f"  time_scope     : {nlu.get('time_scope')}")

    # Retrieval summary
    print("\n【第2步 数据检索】")
    coverage = rr.get("coverage", {})
    print(f"  executed_sources : {rr.get('executed_sources')}")
    print(f"  coverage         : {json.dumps(coverage, ensure_ascii=False)}")
    print(f"  documents        : {len(rr.get('documents', []))} 条")
    print(f"  structured_data  : {len(rr.get('structured_data', []))} 项")

    # Analysis summary
    summary = rr.get("analysis_summary", {})
    if summary:
        print("\n【分析信号】")

        ms = summary.get("market_signal")
        if isinstance(ms, list):
            for idx, item in enumerate(ms, 1):
                if not isinstance(item, dict):
                    continue
                print(f"  📈 市场信号[{idx}]:")
                print(f"     symbol        : {item.get('symbol')} ({item.get('canonical_name')})")
                print(f"     close         : {item.get('close')}")
                print(f"     pct_change_1d : {item.get('pct_change_1d')}%")
                print(f"     trend_signal  : {item.get('trend_signal')}")
                print(f"     RSI_14        : {item.get('rsi_14')}")
                print(f"     MA5 / MA20    : {item.get('ma5')} / {item.get('ma20')}")
                macd = item.get("macd")
                if macd:
                    print(f"     MACD          : line={macd.get('macd_line')}, signal={macd.get('signal_line')}, hist={macd.get('histogram')}")
                bb = item.get("bollinger")
                if bb:
                    print(f"     Bollinger     : upper={bb.get('upper')}, mid={bb.get('middle')}, lower={bb.get('lower')}, bw={bb.get('bandwidth')}%")
                print(f"     volatility    : {item.get('volatility_20d')}%")
                pct = item.get("pct_change_nd")
                if pct:
                    print(f"     多日涨跌幅    : 3d={pct.get('pct_3d')}%, 5d={pct.get('pct_5d')}%, 10d={pct.get('pct_10d')}%, 20d={pct.get('pct_20d')}%")
                pvma = item.get("price_vs_ma")
                if pvma:
                    print(f"     price_vs_ma   : above_ma5={pvma.get('above_ma5')}, above_ma20={pvma.get('above_ma20')}")
        elif isinstance(ms, dict) and ms:
            print(f"  📈 市场信号:")
            print(f"     symbol        : {ms.get('symbol')} ({ms.get('canonical_name')})")
            print(f"     close         : {ms.get('close')}")
            print(f"     pct_change_1d : {ms.get('pct_change_1d')}%")
            print(f"     trend_signal  : {ms.get('trend_signal')}")
            print(f"     RSI_14        : {ms.get('rsi_14')}")
            print(f"     MA5 / MA20    : {ms.get('ma5')} / {ms.get('ma20')}")
            macd = ms.get("macd")
            if macd:
                print(f"     MACD          : line={macd.get('macd_line')}, signal={macd.get('signal_line')}, hist={macd.get('histogram')}")
            bb = ms.get("bollinger")
            if bb:
                print(f"     Bollinger     : upper={bb.get('upper')}, mid={bb.get('middle')}, lower={bb.get('lower')}, bw={bb.get('bandwidth')}%")
            print(f"     volatility    : {ms.get('volatility_20d')}%")
            pct = ms.get("pct_change_nd")
            if pct:
                print(f"     多日涨跌幅    : 3d={pct.get('pct_3d')}%, 5d={pct.get('pct_5d')}%, 10d={pct.get('pct_10d')}%, 20d={pct.get('pct_20d')}%")
            pvma = ms.get("price_vs_ma")
            if pvma:
                print(f"     price_vs_ma   : above_ma5={pvma.get('above_ma5')}, above_ma20={pvma.get('above_ma20')}")

        fs = summary.get("fundamental_signal")
        if isinstance(fs, list):
            for idx, item in enumerate(fs, 1):
                if not isinstance(item, dict):
                    continue
                print(f"  📊 基本面信号[{idx}]:")
                print(f"     PE/PB/ROE     : {item.get('pe_ttm')} / {item.get('pb')} / {item.get('roe')}")
                print(f"     估值评估      : {item.get('valuation_assessment')}")
        elif isinstance(fs, dict) and fs:
            print(f"  📊 基本面信号:")
            print(f"     PE/PB/ROE     : {fs.get('pe_ttm')} / {fs.get('pb')} / {fs.get('roe')}")
            print(f"     估值评估      : {fs.get('valuation_assessment')}")

        mcs = summary.get("macro_signal")
        if mcs:
            print(f"  🌐 宏观信号:")
            for ind in mcs.get("indicators", []):
                print(f"     {ind.get('indicator_name')} = {ind.get('metric_value')} → {ind.get('direction')}")
            print(f"     综合方向      : {mcs.get('overall')}")

        readiness = summary.get("data_readiness")
        if readiness:
            print(f"  📋 数据就绪      : {json.dumps(readiness, ensure_ascii=False)}")

    # Output paths
    print("\n" + "=" * 60)
    print(f"  输出目录: {run_dir}")
    print(f"  NLU结果 : {run_dir / 'nlu_result.json'}")
    print(f"  检索结果: {run_dir / 'retrieval_result.json'}")
    print("=" * 60)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one manual Query Intelligence test and write NLU/Retrieval JSON outputs.")
    parser.add_argument("--query", type=str, default="", help="Question to evaluate. If omitted, the script prompts in the terminal.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k passed to the retrieval module.")
    # Live data toggles (default: market/macro/news on, announcement off)
    parser.add_argument("--live-market", dest="live_market", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live market data (default: on). Use --no-live-market to disable.")
    parser.add_argument("--live-macro", dest="live_macro", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live macro data (default: on). Use --no-live-macro to disable.")
    parser.add_argument("--live-news", dest="live_news", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live news data (default: on). Use --no-live-news to disable.")
    parser.add_argument("--live-announcement", dest="live_announcement", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable live announcement data (default: off — cninfo often hangs from overseas). Use --live-announcement to enable.")
    return parser


def _prompt_query() -> str:
    sys.stdout.write("请输入问题: ")
    sys.stdout.flush()
    stdin_buffer = getattr(sys.stdin, "buffer", None)
    if stdin_buffer is None:
        return sys.stdin.readline().strip()
    return _decode_stdin_bytes(stdin_buffer.readline()).strip()


def _decode_stdin_bytes(raw: bytes) -> str:
    encodings = [
        sys.stdin.encoding,
        sys.getfilesystemencoding(),
        "utf-8",
        "gb18030",
        "gbk",
        "big5",
    ]
    tried: set[str] = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _prepare_run_dir(query: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(query)[:48] or "manual-query"
    run_dir = OUTPUT_DIR / f"{timestamp}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-_")


if __name__ == "__main__":
    main()
