"""Coverage test: verify document sentiment analysis behavior across diverse inputs.

Follows the same pattern as test_coverage.py (Query Intelligence coverage):
pre-defined test cases covering source types, languages, skip logic,
sentiment polarity, and batch processing, each with expected outcomes.

Usage:
    python manual_test/test_coverage_sentiment.py
    python manual_test/test_coverage_sentiment.py --quick  # core cases only
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment import Preprocessor, SentimentClassifier
from sentiment.schemas import SKIP_PRODUCT_TYPES, SUPPORTED_SOURCE_TYPES


# ── Mock inference ────────────────────────────────────────────────────

_MOCK_RESPONSES: dict[str, dict[str, float]] = {
    "positive": {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    "negative": {"positive": 0.05, "neutral": 0.15, "negative": 0.80},
    "neutral":  {"positive": 0.10, "neutral": 0.80, "negative": 0.10},
}


def _mock_infer(text: str, language: str) -> dict[str, float]:
    positive_kw = ["增长", "record", "超预期", "创新", "突破", "beat", "upgrade", "overweight"]
    negative_kw = ["下降", "decline", "miss", "downgrade", "sell", "下跌", "担忧", "放缓"]
    t = text.casefold()
    pos = any(kw in t for kw in positive_kw)
    neg = any(kw in t for kw in negative_kw)
    if pos and not neg:
        return _MOCK_RESPONSES["positive"]
    elif neg and not pos:
        return _MOCK_RESPONSES["negative"]
    return _MOCK_RESPONSES["neutral"]


# ── Test case definition ──────────────────────────────────────────────
# (case_name, nlu, documents, expected_checks)  where expected_checks is a dict

CASE_NLU = {
    "stock": {
        "product_type": {"label": "stock"},
        "entities": [{"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"}],
    },
    "out_of_scope": {"product_type": {"label": "out_of_scope"}, "entities": []},
    "product_info": {"product_type": {"label": "product_info"}, "entities": []},
    "trading_rule": {"product_type": {"label": "trading_rule_fee"}, "entities": []},
    "two_entities": {
        "product_type": {"label": "stock"},
        "entities": [
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
            {"symbol": "000858.SZ", "canonical_name": "五粮液", "mention": "五粮液"},
        ],
    },
    "no_entities": {"product_type": {"label": "stock"}, "entities": []},
}

TEST_CASES: list[tuple] = [
    # ── Source type coverage ──
    (
        "source/news",
        CASE_NLU["stock"],
        [{"evidence_id": "n_001", "source_type": "news", "title": "新闻",
          "body": "公司营收增长超预期。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_skipped": 0, "doc_skipped": False},
    ),
    (
        "source/announcement",
        CASE_NLU["stock"],
        [{"evidence_id": "a_001", "source_type": "announcement", "title": "公告",
          "body": "董事会审议通过相关议案。", "body_available": True, "rank_score": 0.8}],
        {"expected_analyzed": 1, "expected_skipped": 0, "doc_skipped": False},
    ),
    (
        "source/research_note",
        CASE_NLU["stock"],
        [{"evidence_id": "r_001", "source_type": "research_note", "title": "研报",
          "body": "Company beat earnings estimates. Upgrade to overweight.", "body_available": True, "rank_score": 0.7}],
        {"expected_analyzed": 1, "expected_skipped": 0, "doc_skipped": False},
    ),
    (
        "source/product_doc",
        CASE_NLU["stock"],
        [{"evidence_id": "p_001", "source_type": "product_doc", "title": "产品书",
          "body": "该基金跟踪沪深300指数。", "body_available": True, "rank_score": 0.5}],
        {"expected_analyzed": 1, "expected_skipped": 0, "doc_skipped": False},
    ),
    (
        "source/unsupported_faq",
        CASE_NLU["stock"],
        [{"evidence_id": "f_001", "source_type": "faq", "title": "FAQ"}],
        {"expected_analyzed": 0, "expected_skipped": 1, "doc_skipped": True},
    ),
    (
        "source/unsupported_chat",
        CASE_NLU["stock"],
        [{"evidence_id": "c_001", "source_type": "chat", "title": "聊天"}],
        {"expected_analyzed": 0, "expected_skipped": 1, "doc_skipped": True},
    ),

    # ── Language coverage ──
    (
        "lang/zh",
        CASE_NLU["stock"],
        [{"evidence_id": "lz_001", "source_type": "news", "title": "中文",
          "body": "公司营收增长显著，净利润创历史新高。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_skip_reason": None, "lang": "zh"},
    ),
    (
        "lang/en",
        CASE_NLU["stock"],
        [{"evidence_id": "le_001", "source_type": "research_note", "title": "English",
          "body": "Apple reported record quarterly revenue and beat estimates.", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_skip_reason": None, "lang": "en"},
    ),

    # ── Text level coverage ──
    (
        "text_level/full",
        CASE_NLU["stock"],
        [{"evidence_id": "tf_001", "source_type": "news", "title": "全文",
          "body": "公司昨日发布公告。营收增长超预期。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "text_level": "full"},
    ),
    (
        "text_level/short",
        CASE_NLU["stock"],
        [{"evidence_id": "ts_001", "source_type": "news", "title": "标题摘要",
          "summary": "业绩增长超预期", "body_available": False, "rank_score": 0.9}],
        {"expected_analyzed": 1, "text_level": "short"},
    ),
    (
        "text_level/empty_body_available",
        CASE_NLU["stock"],
        [{"evidence_id": "te_001", "source_type": "news", "title": "空正文",
          "body": "", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 0, "expected_skipped": 1, "doc_skipped": True},
    ),
    (
        "text_level/no_text",
        CASE_NLU["stock"],
        [{"evidence_id": "tn_001", "source_type": "news", "body_available": False, "rank_score": 0.9}],
        {"expected_analyzed": 0, "expected_skipped": 1, "doc_skipped": True},
    ),

    # ── Query-level skip ──
    (
        "skip/out_of_scope",
        CASE_NLU["out_of_scope"],
        [{"evidence_id": "os_001", "source_type": "news", "title": "无关",
          "body": "一些内容。", "body_available": True}],
        {"expected_skip_reason": "product_type=out_of_scope", "expected_analyzed": 0, "expected_docs_count": 0},
    ),
    (
        "skip/product_info",
        CASE_NLU["product_info"],
        [{"evidence_id": "pi_001", "source_type": "news", "title": "产品",
          "body": "一些内容。", "body_available": True}],
        {"expected_skip_reason": "product_type=product_info", "expected_analyzed": 0, "expected_docs_count": 0},
    ),
    (
        "skip/trading_rule",
        CASE_NLU["trading_rule"],
        [{"evidence_id": "tr_001", "source_type": "news", "title": "规则",
          "body": "一些内容。", "body_available": True}],
        {"expected_skip_reason": "product_type=trading_rule_fee", "expected_analyzed": 0, "expected_docs_count": 0},
    ),

    # ── Entity coverage ──
    (
        "entity/single",
        CASE_NLU["stock"],
        [{"evidence_id": "es_001", "source_type": "news", "title": "茅台新闻",
          "body": "贵州茅台发布最新财报。营收增长超预期。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_entity_hits": ["600519.SH"]},
    ),
    (
        "entity/two_entities",
        CASE_NLU["two_entities"],
        [{"evidence_id": "et_001", "source_type": "news", "title": "两家公司",
          "body": "贵州茅台发布财报。五粮液也有不错表现。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_entity_hits": ["600519.SH", "000858.SZ"]},
    ),
    (
        "entity/no_entities",
        CASE_NLU["no_entities"],
        [{"evidence_id": "en_001", "source_type": "news", "title": "无实体",
          "body": "公司发布最新财报。营收增长超预期。", "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "expected_entity_hits": []},
    ),

    # ── Sentiment polarity (mock should produce consistent results) ──
    (
        "sentiment/positive_zh",
        CASE_NLU["stock"],
        [{"evidence_id": "spz_001", "source_type": "news", "title": "利好",
          "body": "公司营收大幅增长，净利润创历史新高。业绩超预期，多家机构上调评级。",
          "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "sentiment_label": "positive"},
    ),
    (
        "sentiment/negative_zh",
        CASE_NLU["stock"],
        [{"evidence_id": "snz_001", "source_type": "news", "title": "利空",
          "body": "公司营收同比下降。净利润出现亏损。股价持续下跌。市场普遍担忧。",
          "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "sentiment_label": "negative"},
    ),
    (
        "sentiment/neutral_zh",
        CASE_NLU["stock"],
        [{"evidence_id": "snt_001", "source_type": "news", "title": "中性",
          "body": "公司发布第三季度财务报告。董事会审议通过相关议案。",
          "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "sentiment_label": "neutral"},
    ),
    (
        "sentiment/positive_en",
        CASE_NLU["stock"],
        [{"evidence_id": "spe_001", "source_type": "research_note", "title": "Bullish",
          "body": "The company beat earnings by a wide margin. We upgrade to overweight.",
          "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "sentiment_label": "positive"},
    ),
    (
        "sentiment/negative_en",
        CASE_NLU["stock"],
        [{"evidence_id": "sne_001", "source_type": "research_note", "title": "Bearish",
          "body": "Revenue missed estimates. We downgrade the stock to sell.",
          "body_available": True, "rank_score": 0.9}],
        {"expected_analyzed": 1, "sentiment_label": "negative"},
    ),

    # ── Batch coverage ──
    (
        "batch/mixed_supported",
        CASE_NLU["stock"],
        [
            {"evidence_id": "b_001", "source_type": "news", "title": "新闻",
             "body": "营收增长超预期。", "body_available": True, "rank_score": 0.9},
            {"evidence_id": "b_002", "source_type": "announcement", "title": "公告",
             "body": "董事会审议通过。", "body_available": True, "rank_score": 0.8},
            {"evidence_id": "b_003", "source_type": "research_note", "title": "研报",
             "body": "Company beat estimates. Raise price target.", "body_available": True, "rank_score": 0.7},
        ],
        {"expected_analyzed": 3, "expected_skipped": 0, "expected_results": 3},
    ),
    (
        "batch/all_skipped",
        CASE_NLU["stock"],
        [
            {"evidence_id": "bs_001", "source_type": "faq", "title": "FAQ"},
            {"evidence_id": "bs_002", "source_type": "chat", "title": "聊天"},
        ],
        {"expected_analyzed": 0, "expected_skipped": 2, "expected_results": 0},
    ),
    (
        "batch/mixed_skipped_and_ok",
        CASE_NLU["stock"],
        [
            {"evidence_id": "mx_001", "source_type": "news", "title": "新闻",
             "body": "营收增长。", "body_available": True, "rank_score": 0.9},
            {"evidence_id": "mx_002", "source_type": "faq", "title": "FAQ"},
            {"evidence_id": "mx_003", "source_type": "research_note", "title": "研报",
             "body": "Beat estimates.", "body_available": True, "rank_score": 0.7},
        ],
        {"expected_analyzed": 2, "expected_skipped": 1, "expected_results": 2},
    ),
    (
        "batch/empty",
        CASE_NLU["stock"],
        [],
        {"expected_analyzed": 0, "expected_skipped": 0, "expected_results": 0},
    ),

    # ── Score/confidence bounds ──
    (
        "bounds/all_positive",
        CASE_NLU["stock"],
        [{"evidence_id": "bp_001", "source_type": "news", "title": "全正面",
          "body": "营收增长超预期。净利润创新高。业绩突破历史纪录。",
          "body_available": True, "rank_score": 0.9}],
        {"expected_score_range": (0.7, 1.0), "expected_conf_range": (0.6, 1.0)},
    ),
]

# Core cases for --quick
QUICK_CASES = [
    tc for tc in TEST_CASES
    if tc[0] in [
        "source/news", "source/unsupported_faq",
        "lang/zh", "lang/en",
        "text_level/full", "text_level/short",
        "skip/out_of_scope",
        "sentiment/positive_zh", "sentiment/negative_zh", "sentiment/neutral_zh",
        "batch/mixed_supported", "batch/mixed_skipped_and_ok", "batch/empty",
    ]
]


# ── Test runner ───────────────────────────────────────────────────────


def _patch_classifier(clf):
    patcher = patch.object(clf, "_infer_single", side_effect=_mock_infer)
    patcher.start()
    return patcher


def run_tests(cases: list[tuple]) -> list[dict]:
    preprocessor = Preprocessor()
    classifier = SentimentClassifier(device="cpu")
    _patch_classifier(classifier)

    results = []

    for i, (name, nlu, documents, expected) in enumerate(cases, 1):
        t0 = time.time()
        retrieval = {"documents": documents}

        try:
            skip_reason, docs, meta = preprocessor.process_query(nlu, retrieval)
            sentiment_results = classifier.analyze_documents(docs)

            elapsed = time.time() - t0

            checks: dict[str, bool] = {}
            messages: list[str] = []

            # Check skip_reason
            exp_skip = expected.get("expected_skip_reason")
            if exp_skip is not None:
                checks["skip_reason_match"] = skip_reason == exp_skip
                if not checks["skip_reason_match"]:
                    messages.append(f"skip_reason: expected={exp_skip}, actual={skip_reason}")
            else:
                checks["no_skip_reason"] = skip_reason is None
                if not checks["no_skip_reason"]:
                    messages.append(f"unexpected skip_reason={skip_reason}")

            # Check counts
            exp_analyzed = expected.get("expected_analyzed")
            if exp_analyzed is not None:
                checks["analyzed_count"] = meta.analyzed_docs_count == exp_analyzed
                if not checks["analyzed_count"]:
                    messages.append(f"analyzed_count: expected={exp_analyzed}, actual={meta.analyzed_docs_count}")

            exp_skipped = expected.get("expected_skipped")
            if exp_skipped is not None:
                checks["skipped_count"] = meta.skipped_docs_count == exp_skipped
                if not checks["skipped_count"]:
                    messages.append(f"skipped_count: expected={exp_skipped}, actual={meta.skipped_docs_count}")

            # Check docs count
            exp_docs = expected.get("expected_docs_count")
            if exp_docs is not None:
                checks["docs_count"] = len(docs) == exp_docs
            else:
                checks["docs_match_meta"] = len(docs) == meta.analyzed_docs_count + meta.skipped_docs_count

            # Check doc-level fields
            if "doc_skipped" in expected:
                for d in docs:
                    checks[f"doc_{d.evidence_id}_skipped"] = d.skipped == expected["doc_skipped"]
                    if d.skipped != expected["doc_skipped"]:
                        messages.append(f"doc_{d.evidence_id}: skipped expected={expected['doc_skipped']}, actual={d.skipped}")

            if "lang" in expected and docs and not docs[0].skipped:
                checks["language_match"] = docs[0].language == expected["lang"]
                if not checks["language_match"]:
                    messages.append(f"language: expected={expected['lang']}, actual={docs[0].language}")

            if "text_level" in expected and docs and not docs[0].skipped:
                checks["text_level_match"] = docs[0].text_level == expected["text_level"]
                if not checks["text_level_match"]:
                    messages.append(f"text_level: expected={expected['text_level']}, actual={docs[0].text_level}")

            if "expected_entity_hits" in expected and docs and not docs[0].skipped:
                expected_hits = set(expected["expected_entity_hits"])
                actual_hits = set(docs[0].entity_hits)
                checks["entity_hits_match"] = actual_hits == expected_hits
                if not checks["entity_hits_match"]:
                    messages.append(f"entity_hits: expected={expected_hits}, actual={actual_hits}")

            # Check results
            exp_results = expected.get("expected_results")
            if exp_results is not None:
                checks["results_count"] = len(sentiment_results) == exp_results
                if not checks["results_count"]:
                    messages.append(f"results_count: expected={exp_results}, actual={len(sentiment_results)}")

            # Check sentiment label
            exp_label = expected.get("sentiment_label")
            if exp_label is not None and sentiment_results:
                checks["sentiment_label_match"] = sentiment_results[0].sentiment_label == exp_label
                if not checks["sentiment_label_match"]:
                    messages.append(f"sentiment_label: expected={exp_label}, actual={sentiment_results[0].sentiment_label}")

            # Check score bounds
            score_range = expected.get("expected_score_range")
            if score_range is not None and sentiment_results:
                lo, hi = score_range
                for r in sentiment_results:
                    checks[f"score_{r.evidence_id}_in_range"] = lo <= r.sentiment_score <= hi
                    if not checks[f"score_{r.evidence_id}_in_range"]:
                        messages.append(f"score: expected [{lo}, {hi}], actual={r.sentiment_score}")

            conf_range = expected.get("expected_conf_range")
            if conf_range is not None and sentiment_results:
                lo, hi = conf_range
                for r in sentiment_results:
                    checks[f"conf_{r.evidence_id}_in_range"] = lo <= r.confidence <= hi

            all_pass = all(checks.values())
            status = "PASS" if all_pass else "FAIL"

        except Exception as exc:
            status = "ERROR"
            all_pass = False
            checks = {}
            messages = [str(exc)[:200]]
            elapsed = time.time() - t0

        results.append({
            "idx": i,
            "name": name,
            "status": status,
            "all_pass": all_pass,
            "checks": checks,
            "messages": messages,
            "elapsed": round(elapsed, 3),
            "nlu_product_type": nlu.get("product_type", {}).get("label", "?") if isinstance(nlu.get("product_type"), dict) else "?",
            "num_docs": len(documents),
            "analyzed": meta.analyzed_docs_count if 'meta' in dir() else -1,
            "skipped": meta.skipped_docs_count if 'meta' in dir() else -1,
            "results_count": len(sentiment_results) if 'sentiment_results' in dir() else -1,
        })

    return results


# ── Report printer ────────────────────────────────────────────────────


def print_report(results: list[dict]):
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\n{'=' * 70}")
    print(f"  Coverage Test: {passed}/{total} PASS  |  {failed} FAIL  |  {errors} ERROR")
    print(f"  Pass rate: {passed / max(total, 1) * 100:.0f}%")
    print(f"{'=' * 70}")

    # By category
    print(f"\n## 按类别统计\n")
    cats: dict[str, list[dict]] = {}
    for r in results:
        cat = r["name"].split("/")[0]
        cats.setdefault(cat, []).append(r)
    for cat, items in sorted(cats.items()):
        p = sum(1 for r in items if r["status"] == "PASS")
        f = sum(1 for r in items if r["status"] == "FAIL")
        e = sum(1 for r in items if r["status"] == "ERROR")
        print(f"    {cat:<20} PASS={p:<3} FAIL={f:<3} ERROR={e:<3} total={len(items)}")

    # Details
    print(f"\n## 逐条结果\n")
    for r in results:
        icon = {"PASS": "  OK", "FAIL": "  NG", "ERROR": "  !! "}.get(r["status"], " ? ")
        msg = "; ".join(r.get("messages", []))[:120]
        line = f"  {icon} [{r['idx']:>2}] {r['name']:<30} {r['status']:<5}"
        if msg:
            line += f"\n         {msg}"
        print(line)

    # Timing
    times = [r["elapsed"] for r in results if r["status"] != "ERROR"]
    if times:
        print(f"\n## 耗时统计\n")
        print(f"  平均: {sum(times)/len(times):.3f}s  最快: {min(times):.3f}s  最慢: {max(times):.3f}s  总计: {sum(times):.3f}s")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment analysis coverage test")
    parser.add_argument("--quick", action="store_true", help="Run core cases only")
    args = parser.parse_args()

    cases = QUICK_CASES if args.quick else TEST_CASES
    print(f"Running {len(cases)} coverage test cases...")
    results = run_tests(cases)
    print_report(results)

    # Save JSON report
    report_dir = ROOT / "manual_test" / "output"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "coverage_sentiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
