from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("QI_USE_LIVE_MARKET", "1")
os.environ.setdefault("QI_USE_LIVE_MACRO", "1")
os.environ.setdefault("QI_USE_LIVE_NEWS", "1")
os.environ.setdefault("QI_USE_LIVE_ANNOUNCEMENT", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import build_default_service, clear_service_caches

from sentiment import Preprocessor, SentimentClassifier

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ── Mock inference for sentiment ──────────────────────────────────────

_POSITIVE_KW = [
    "增长", "revenue grew", "record", "龙头", "超预期", "创历史",
    "看好", "新高", "突破", "里程碑", "upgrade", "overweight", "beat",
]
_NEGATIVE_KW = [
    "下降", "decline", "loss", "下跌", "担忧", "放缓", "miss",
    "downgrade", "sell", "collapse", "headwind", "蒸发", "压力", "下调",
]


def _mock_infer(text: str, language: str) -> dict[str, float]:
    text_lower = text.casefold()
    pos = any(kw in text_lower for kw in _POSITIVE_KW)
    neg = any(kw in text_lower for kw in _NEGATIVE_KW)
    if pos and not neg:
        return {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
    elif neg and not pos:
        return {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
    elif pos and neg:
        return {"positive": 0.30, "neutral": 0.50, "negative": 0.20}
    return {"positive": 0.10, "neutral": 0.80, "negative": 0.10}


# ── Sentiment runner ──────────────────────────────────────────────────


def _run_sentiment(
    nlu_result: dict,
    retrieval_result: dict,
    use_real_models: bool,
) -> list[dict]:
    """Run the Preprocessor → SentimentClassifier pipeline."""
    preprocessor = Preprocessor()
    classifier = SentimentClassifier(device="cpu")

    if not use_real_models:
        patcher = patch.object(classifier, "_infer_single", side_effect=_mock_infer)
        patcher.start()

    skip_reason, docs, meta = preprocessor.process_query(nlu_result, retrieval_result)
    results = classifier.analyze_documents(docs)

    if skip_reason:
        print(f"  [sentiment] query skipped: {skip_reason}")

    return [r.model_dump() for r in results]


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    args = _build_parser().parse_args()
    query = args.query.strip() if args.query else _prompt_query()
    if not query:
        raise SystemExit("query is empty")

    clear_service_caches()
    service = build_default_service()

    nlu_result = service.analyze_query(query, debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=args.top_k, debug=True)

    run_dir = _prepare_run_dir(query)
    (run_dir / "query.txt").write_text(query + "\n", encoding="utf-8")
    (run_dir / "nlu_result.json").write_text(
        json.dumps(nlu_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "retrieval_result.json").write_text(
        json.dumps(retrieval_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = {
        "query": query,
        "output_dir": str(run_dir),
        "nlu_result": str(run_dir / "nlu_result.json"),
        "retrieval_result": str(run_dir / "retrieval_result.json"),
    }

    if args.sentiment:
        print("  [sentiment] running document sentiment analysis...")
        sentiment_results = _run_sentiment(nlu_result, retrieval_result, args.real_models)
        (run_dir / "sentiment_results.json").write_text(
            json.dumps(sentiment_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output["sentiment_results"] = str(run_dir / "sentiment_results.json")
        if sentiment_results:
            labels = [r["sentiment_label"] for r in sentiment_results]
            pos = sum(1 for l in labels if l == "positive")
            neg = sum(1 for l in labels if l == "negative")
            neu = sum(1 for l in labels if l == "neutral")
            print(f"  [sentiment] {len(sentiment_results)} docs: "
                  f"positive={pos} negative={neg} neutral={neu}")
        else:
            print(f"  [sentiment] no documents to analyze")

    print(json.dumps(output, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one manual Query Intelligence test and write NLU/Retrieval JSON outputs.")
    parser.add_argument("--query", type=str, default="", help="Question to evaluate. If omitted, the script prompts in the terminal.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k passed to the retrieval module.")
    parser.add_argument("--sentiment", action="store_true", default=False, help="Run document sentiment analysis on retrieved documents.")
    parser.add_argument("--real-models", action="store_true", default=False, help="Use real FinBERT models (not mock) for sentiment analysis.")
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
