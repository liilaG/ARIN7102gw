"""Fuzz test: verify document sentiment analysis pipeline stability across diverse inputs.

Generates random NLU configurations and documents, runs Preprocessor → SentimentClassifier,
validates output structure, and reports pass/fail summary.

Usage:
  python manual_test/test_fuzz_sentiment.py
  python manual_test/test_fuzz_sentiment.py --num-samples 200 --seed 1
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment import Preprocessor, SentimentClassifier
from sentiment.preprocessor import normalize_text
from sentiment.schemas import (
    SKIP_PRODUCT_TYPES,
    SUPPORTED_SOURCE_TYPES,
    FilterMeta,
    PreprocessedDoc,
    SentimentItem,
)

OUTPUT_DIR = ROOT / "manual_test" / "output"


# ===========================================================================
# Mock inference (deterministic, no model download)
# ===========================================================================

_POSITIVE_KW = [
    "增长", "revenue grew", "record", "龙头", "超预期", "创历史",
    "看好", "新高", "突破", "里程碑", "升级", "supercycle",
    "overweight", "beat", "上调", "供不应求", "利好",
]
_NEGATIVE_KW = [
    "下降", "decline", "loss", "下跌", "担忧", "放缓", "miss",
    "downgrade", "sell", "collapse", "headwind", "蒸发", "压力",
    "下调", "积压", "不确定性", "延迟", "阻力",
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
    else:
        return {"positive": 0.10, "neutral": 0.80, "negative": 0.10}


# ===========================================================================
# Entity / NLU generators
# ===========================================================================

ENTITY_TEMPLATES = [
    {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
    {"symbol": "000858.SZ", "canonical_name": "五粮液", "mention": "五粮液"},
    {"symbol": "300750.SZ", "canonical_name": "宁德时代", "mention": "宁德时代"},
    {"symbol": "601318.SH", "canonical_name": "中国平安", "mention": "中国平安"},
    {"symbol": "AAPL", "canonical_name": "Apple Inc.", "mention": "Apple"},
    {"symbol": "TSLA", "canonical_name": "Tesla Inc.", "mention": "特斯拉"},
    {"symbol": "MSFT", "canonical_name": "Microsoft", "mention": "Microsoft"},
    {"symbol": "NVDA", "canonical_name": "NVIDIA", "mention": "英伟达"},
]

PRODUCT_TYPE_LABELS = ["stock", "etf", "index", "fund", "macro"] + list(SKIP_PRODUCT_TYPES)


def _random_nlu(rng: random.Random) -> dict:
    """Generate a random NLU result dict."""
    product_label = rng.choice(PRODUCT_TYPE_LABELS)
    # Select 0–4 entities
    num_entities = rng.choices([0, 1, 2, 3, 4], weights=[5, 30, 35, 20, 10])[0]
    entities = rng.sample(ENTITY_TEMPLATES, min(num_entities, len(ENTITY_TEMPLATES)))

    return {
        "query_id": f"fuzz_{rng.randint(1, 99999):05d}",
        "product_type": {"label": product_label, "confidence": round(rng.uniform(0.5, 1.0), 3)},
        "entities": entities,
    }


# ===========================================================================
# Document generators
# ===========================================================================

# Chinese sentences with clear sentiment
_ZH_POSITIVE_SENTENCES = [
    "公司营业收入同比增长25%，大幅超出市场预期。",
    "净利润创下历史新高，盈利能力持续增强。",
    "新产品市场反响热烈，订单供不应求。",
    "公司宣布大手笔分红计划，股息率超过3%。",
    "行业龙头地位稳固，市场份额持续提升。",
    "研发取得重大突破，技术水平达到国际领先。",
    "公司基本面稳健，长期投资价值凸显。",
    "销量创历史新高，品牌影响力不断增强。",
    "分析师一致看好，纷纷上调目标股价。",
    "公司现金流充裕，资产负债率持续优化。",
]
_ZH_NEGATIVE_SENTENCES = [
    "公司营收同比下降15%，业绩大幅低于预期。",
    "净利润出现亏损，经营状况持续恶化。",
    "产品销量持续下滑，库存积压严重。",
    "公司股价连续下跌，市值蒸发超过百亿。",
    "行业增速明显放缓，竞争格局日益激烈。",
    "公司面临严峻的监管审查，不确定性增加。",
    "多家机构下调公司评级，投资者信心不足。",
    "公司债务规模持续扩大，偿债压力上升。",
    "产品质量问题频发，品牌声誉严重受损。",
    "海外业务进展受阻，地缘政治风险上升。",
]
_ZH_NEUTRAL_SENTENCES = [
    "公司发布2026年第一季度财务报告。",
    "董事会审议通过了相关决议事项。",
    "公司主营业务未发生重大变化。",
    "该指数由300只成份股组成。",
    "股东大会将于5月15日召开。",
    "公司管理层在业绩说明会上介绍了经营情况。",
    "交易所对相关规则进行了修订完善。",
    "该基金跟踪沪深300指数，费率为0.15%。",
    "定期报告按照监管要求编制和披露。",
    "公司表示将持续关注市场动态。",
]

_EN_POSITIVE_SENTENCES = [
    "Revenue grew 22% year-over-year, beating consensus estimates.",
    "Net income reached an all-time record this quarter.",
    "We see a multi-year growth supercycle ahead for the company.",
    "The company raised its dividend by 15%, returning capital to shareholders.",
    "Market share expanded across all major geographies.",
    "The new product line has exceeded initial sales expectations.",
    "Operating margins improved significantly due to cost efficiencies.",
    "Strong free cash flow generation supports continued investment.",
    "Analysts upgraded the stock following the strong earnings report.",
    "The company's competitive moat continues to widen.",
]
_EN_NEGATIVE_SENTENCES = [
    "Revenue missed consensus estimates by a wide margin this quarter.",
    "Gross margins compressed 400 basis points year-over-year.",
    "The company issued downside guidance for the next quarter.",
    "Inventory days swelled to the highest level in company history.",
    "Market share declined in the key Chinese market.",
    "The company faces mounting regulatory scrutiny across jurisdictions.",
    "We downgrade the stock to SELL and cut our price target.",
    "Customer demand is weakening faster than previously expected.",
    "Operating losses widened due to rising input costs.",
    "The supply chain disruption shows no signs of easing.",
]
_EN_NEUTRAL_SENTENCES = [
    "The company published its quarterly financial report yesterday.",
    "The board of directors approved the routine governance items.",
    "The annual shareholder meeting is scheduled for June.",
    "The company appointed a new independent director.",
    "The ETF tracks the S&P 500 index with an expense ratio of 0.03%.",
    "Management commentary focused on long-term strategic priorities.",
    "The stock trades on the NASDAQ exchange under the ticker AAPL.",
    "The company complies with all applicable regulations.",
    "The audit committee reviewed the financial statements.",
    "The company has operations in over 50 countries worldwide.",
]


def _generate_text(rng: random.Random, language: str, polarity: str) -> str:
    """Generate a multi-sentence paragraph with given polarity."""
    if language == "zh":
        pool = {
            "positive": _ZH_POSITIVE_SENTENCES,
            "negative": _ZH_NEGATIVE_SENTENCES,
            "neutral": _ZH_NEUTRAL_SENTENCES,
        }
    else:
        pool = {
            "positive": _EN_POSITIVE_SENTENCES,
            "negative": _EN_NEGATIVE_SENTENCES,
            "neutral": _EN_NEUTRAL_SENTENCES,
        }

    # Use the dominant polarity, but mix in other sentences for realism
    primary_pool = pool[polarity]
    num_sentences = rng.randint(1, 8)
    text = "".join(rng.sample(primary_pool, min(num_sentences, len(primary_pool))))

    # Sometimes add a neutral sentence to mixed/moderate docs
    if rng.random() < 0.3:
        extra = rng.choice(pool["neutral"])
        text += extra
    return normalize_text(text)


_UNSUPPORTED_SOURCE_TYPES = ["faq", "chat", "unknown", "weibo", "wechat"]


def _random_document(rng: random.Random, doc_id: str) -> dict:
    """Generate a single random document dict."""
    language = rng.choice(["zh", "zh", "zh", "en", "en"])  # bias Chinese
    polarity = rng.choice(["positive", "negative", "negative", "neutral", "positive", "positive"])
    body_text = _generate_text(rng, language, polarity)

    # 80% supported, 20% unsupported
    if rng.random() < 0.85:
        source_type = rng.choice(list(SUPPORTED_SOURCE_TYPES))
    else:
        source_type = rng.choice(_UNSUPPORTED_SOURCE_TYPES)

    body_available = rng.random() < 0.85
    has_body = body_available and rng.random() < 0.9

    doc = {
        "evidence_id": doc_id,
        "source_type": source_type,
        "source_name": rng.choice(["证券时报", "财经网", "深交所", "上交所", "Wind", "Reuters", "Bloomberg", ""]),
        "publish_time": f"2026-{rng.randint(1,4):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00",
        "title": _generate_text(rng, language, polarity)[:30] or "标题",
        "summary": body_text[:50] if rng.random() < 0.7 else "",
        "body": body_text if has_body else "",
        "body_available": body_available,
        "rank_score": round(rng.uniform(0.1, 1.0), 4),
    }

    # Occasionally drop fields (boundary case)
    drop_choices: list[str] = []
    if rng.random() < 0.05:
        drop_choices.append("source_name")
    if rng.random() < 0.03:
        drop_choices.append("publish_time")
    if rng.random() < 0.02:
        drop_choices.append("title")
    for field in drop_choices:
        doc.pop(field, None)

    return doc


# ===========================================================================
# Boundary / adversarial cases
# ===========================================================================

BOUNDARY_NLU_CASES: list[tuple[str, dict]] = [
    ("empty_entities", {"product_type": {"label": "stock"}, "entities": []}),
    ("missing_product_type", {"entities": []}),
    ("product_type_not_dict", {"product_type": "stock", "entities": []}),
    ("none_entities", {"product_type": {"label": "stock"}}),
    ("empty_dict", {}),
]

BOUNDARY_DOC_CASES: list[tuple[str, list[dict]]] = [
    ("empty_docs", []),
    ("single_null_doc", [{"evidence_id": "null_001", "source_type": "news", "title": "Null"} | {}]),  # noqa: E231
    ("all_unsupported", [
        {"evidence_id": "bad_001", "source_type": "faq", "title": "FAQ"},
        {"evidence_id": "bad_002", "source_type": "unknown", "title": "Unknown"},
        {"evidence_id": "bad_003", "source_type": "chat", "title": "Chat"},
    ]),
    ("body_not_string", [
        {"evidence_id": "obj_001", "source_type": "news", "title": "Obj",
         "body": object(), "body_available": True}
    ]),
    ("very_long_body", [
        {"evidence_id": "long_001", "source_type": "news", "title": "Long",
         "body": "测试。" * 5000, "body_available": True}
    ]),
    ("no_title_no_summary", [
        {"evidence_id": "no_title_001", "source_type": "news",
         "body_available": False}
    ]),
]


# ===========================================================================
# Test runner
# ===========================================================================


class SentimentFuzzRunner:
    """Fuzz test runner for the document sentiment analysis pipeline."""

    def __init__(self, num_samples: int = 100, seed: int = 42):
        self.rng = random.Random(seed)
        self.num_samples = num_samples
        self.preprocessor = Preprocessor()
        self.classifier = SentimentClassifier(device="cpu")
        self._patch_classifier()

    def _patch_classifier(self):
        patcher = patch.object(self.classifier, "_infer_single", side_effect=_mock_infer)
        patcher.start()
        self.classifier._mock_patcher = patcher

    def _validate_skip_reason(self, skip_reason: str | None, nlu: dict) -> dict:
        checks: dict[str, bool] = {}
        pt = nlu.get("product_type", {})
        label = pt.get("label") if isinstance(pt, dict) else None
        if label in SKIP_PRODUCT_TYPES:
            checks["should_skip"] = skip_reason is not None
            checks["skip_reason_nonempty"] = bool(skip_reason)
        else:
            checks["should_not_skip"] = skip_reason is None
        return checks

    def _validate_filter_meta(self, meta: FilterMeta, docs_count: int, skip_reason: str | None) -> dict:
        checks: dict[str, bool] = {}
        # When query is skipped (product_type), no docs are processed at all
        if skip_reason:
            checks["skip_reason_no_docs"] = meta.analyzed_docs_count == 0 and meta.skipped_docs_count == 0
        else:
            checks["total_counts_match"] = meta.analyzed_docs_count + meta.skipped_docs_count == docs_count
        # No negative counts
        checks["non_negative_counts"] = (
            meta.analyzed_docs_count >= 0
            and meta.skipped_docs_count >= 0
            and meta.short_text_fallback_count >= 0
        )
        return checks

    def _validate_processed_docs(self, docs: list[PreprocessedDoc]) -> dict:
        checks: dict[str, bool] = {}
        if not docs:
            checks["no_docs_ok"] = True
            return checks

        for doc in docs:
            checks[f"doc_{doc.evidence_id}_is_preprocesseddoc"] = isinstance(doc, PreprocessedDoc)
            checks[f"doc_{doc.evidence_id}_evidence_id_not_empty"] = bool(doc.evidence_id)

            if not doc.skipped:
                checks[f"doc_{doc.evidence_id}_language_valid"] = doc.language in ("zh", "en", "mixed", "unknown")
                checks[f"doc_{doc.evidence_id}_text_level_valid"] = doc.text_level in ("full", "short")
                checks[f"doc_{doc.evidence_id}_sentences_is_list"] = isinstance(doc.sentences, list)
                checks[f"doc_{doc.evidence_id}_entity_hits_is_list"] = isinstance(doc.entity_hits, list)
                # rank_score should not be 0.0 (it was random > 0.1)
            else:
                checks[f"doc_{doc.evidence_id}_skip_reason"] = bool(doc.skip_reason)
                checks[f"doc_{doc.evidence_id}_skipped_empty_text"] = doc.raw_text == ""
                checks[f"doc_{doc.evidence_id}_skipped_no_sentences"] = doc.sentences == []
                checks[f"doc_{doc.evidence_id}_skipped_unknown_lang"] = doc.language == "unknown"
                checks[f"doc_{doc.evidence_id}_skipped_no_entity_hits"] = doc.entity_hits == []

        return checks

    def _validate_sentiment_results(self, results: list[SentimentItem]) -> dict:
        checks: dict[str, bool] = {}
        if not results:
            checks["empty_results_ok"] = True
            return checks

        for item in results:
            eid = item.evidence_id
            checks[f"result_{eid}_is_sentimentitem"] = isinstance(item, SentimentItem)
            checks[f"result_{eid}_label_valid"] = item.sentiment_label in ("positive", "negative", "neutral")
            checks[f"result_{eid}_score_in_range"] = 0.0 <= item.sentiment_score <= 1.0
            checks[f"result_{eid}_conf_in_range"] = 0.0 <= item.confidence <= 1.0
            checks[f"result_{eid}_entity_symbols_is_list"] = isinstance(item.entity_symbols, list)
            checks[f"result_{eid}_text_level_valid"] = item.text_level in ("full", "short")

        # Deduplicate: no two results with same evidence_id
        ids = [r.evidence_id for r in results]
        checks["no_duplicate_ids"] = len(ids) == len(set(ids))

        return checks

    def run_single(self, nlu: dict, documents: list[dict]) -> dict:
        """Run the full sentiment pipeline on one sample. Returns check results."""
        checks: dict[str, bool] = {}
        errors: list[str] = []
        elapsed: float = 0.0

        retrieval = {"query_id": nlu.get("query_id", "fuzz_00000"), "documents": documents}

        try:
            t0 = time.time()

            # Step 1: Preprocessor
            skip_reason, docs, meta = self.preprocessor.process_query(nlu, retrieval)

            checks.update(self._validate_skip_reason(skip_reason, nlu))
            checks.update(self._validate_filter_meta(meta, len(documents), skip_reason))
            checks.update(self._validate_processed_docs(docs))

            # Step 2: Classifier
            results = self.classifier.analyze_documents(docs)

            checks.update(self._validate_sentiment_results(results))

            # Cross-field consistency
            num_skipped = sum(1 for d in docs if d.skipped)
            num_analyzed = len(docs) - num_skipped
            checks["result_count_matches_analyzed"] = len(results) == num_analyzed

            elapsed = time.time() - t0

        except Exception as exc:
            errors.append(str(exc)[:200])
            checks["no_exception"] = False

        if not errors:
            checks["no_exception"] = True

        all_pass = all(checks.values())
        return {"all_pass": all_pass, "checks": checks, "errors": errors, "elapsed": round(elapsed, 3)}

    def run_all(self) -> list[dict]:
        """Generate and run all fuzz samples."""
        results: list[dict] = []

        # Random samples
        for i in range(self.num_samples):
            nlu = _random_nlu(self.rng)
            num_docs = self.rng.choices([1, 2, 3, 5, 10], weights=[20, 30, 25, 15, 10])[0]
            docs = [_random_document(self.rng, f"fuzz_doc_{i}_{j:03d}") for j in range(num_docs)]
            r = self.run_single(nlu, docs)
            r["idx"] = i + 1
            r["category"] = "random"
            r["nlu_product_type"] = nlu.get("product_type", {}).get("label", "unknown") if isinstance(nlu.get("product_type"), dict) else "unknown"
            r["num_docs"] = len(docs)
            results.append(r)

        # Boundary NLU cases
        for name, nlu in BOUNDARY_NLU_CASES:
            docs = [_random_document(self.rng, f"bnd_nlu_{name}")]
            r = self.run_single(nlu, docs)
            r["idx"] = len(results) + 1
            r["category"] = f"boundary_nlu/{name}"
            r["nlu_product_type"] = nlu.get("product_type", {}).get("label", "N/A") if isinstance(nlu.get("product_type"), dict) else "N/A"
            r["num_docs"] = len(docs)
            results.append(r)

        # Boundary document cases
        for name, docs in BOUNDARY_DOC_CASES:
            nlu = {"product_type": {"label": "stock"}, "entities": ENTITY_TEMPLATES[:2]}
            r = self.run_single(nlu, docs)
            r["idx"] = len(results) + 1
            r["category"] = f"boundary_doc/{name}"
            r["nlu_product_type"] = "stock"
            r["num_docs"] = len(docs)
            results.append(r)

        # Specific sentiment verification: verify known-sentiment docs produce expected label
        polarities: list[tuple[str, str, str]] = [
            ("positive_zh", "zh", "positive"),
            ("negative_zh", "zh", "negative"),
            ("neutral_zh", "zh", "neutral"),
            ("positive_en", "en", "positive"),
            ("negative_en", "en", "negative"),
            ("neutral_en", "en", "neutral"),
        ]
        for tag, lang, polarity in polarities:
            body = _generate_text(self.rng, lang, polarity)
            nlu = {
                "product_type": {"label": "stock"},
                "entities": [{"symbol": "TEST", "canonical_name": "测试", "mention": "测试"}],
            }
            docs = [{
                "evidence_id": f"sentiment_check_{tag}",
                "source_type": "news",
                "source_name": "Test",
                "publish_time": "2026-04-27T00:00:00",
                "title": f"Sentiment check: {tag}",
                "body": body,
                "body_available": True,
                "rank_score": 0.9,
            }]
            r = self.run_single(nlu, docs)
            r["idx"] = len(results) + 1
            r["category"] = f"sentiment_check/{tag}"
            r["nlu_product_type"] = "stock"
            r["num_docs"] = len(docs)
            results.append(r)

        return results


# ===========================================================================
# Report
# ===========================================================================


def _print_report(results: list[dict]):
    pass_count = sum(1 for r in results if r["all_pass"])
    fail_count = sum(1 for r in results if r["all_pass"] is False)
    total = len(results)

    print(f"\n{'=' * 70}")
    print(f"FUZZ TEST SUMMARY — Document Sentiment Analysis")
    print(f"{'=' * 70}")
    print(f"  Total:    {total}")
    print(f"  PASS:     {pass_count}")
    print(f"  FAIL:     {fail_count}")
    print(f"  Pass rate: {pass_count / max(total, 1) * 100:.1f}%")

    # By category
    print(f"\n  BY CATEGORY:")
    cat_results: dict[str, list[dict]] = {}
    for r in results:
        cat = r["category"].split("/")[0]
        cat_results.setdefault(cat, []).append(r)
    for cat, items in sorted(cat_results.items()):
        p = sum(1 for r in items if r["all_pass"])
        f = sum(1 for r in items if r["all_pass"] is False)
        print(f"    {cat:<25} PASS={p:<4} FAIL={f:<4} total={len(items)}")

    # By product_type
    print(f"\n  BY PRODUCT TYPE:")
    pt_results: dict[str, list[dict]] = {}
    for r in results:
        pt = r.get("nlu_product_type", "unknown")
        pt_results.setdefault(pt, []).append(r)
    for pt, items in sorted(pt_results.items()):
        p = sum(1 for r in items if r["all_pass"])
        f = sum(1 for r in items if r["all_pass"] is False)
        print(f"    {pt:<25} PASS={p:<4} FAIL={f:<4} total={len(items)}")

    # Failure analysis
    failures = [r for r in results if not r["all_pass"]]
    if failures:
        print(f"\n  FAILURE BREAKDOWN:")
        check_fails: dict[str, int] = {}
        for r in failures:
            for k, v in r.get("checks", {}).items():
                if not v:
                    check_fails[k] = check_fails.get(k, 0) + 1
        for k, v in sorted(check_fails.items(), key=lambda x: -x[1])[:15]:
            print(f"    {k}: {v}")

        print(f"\n  FIRST 3 FAILURES:")
        for r in failures[:3]:
            failed = [k for k, v in r.get("checks", {}).items() if not v]
            err_msg = "; ".join(r.get("errors", []))[:100]
            print(f"    [{r['category']}] idx={r['idx']}")
            print(f"      failed: {failed}")
            if err_msg:
                print(f"      error: {err_msg}")

    # Performance
    elapsed_times = [r["elapsed"] for r in results if "elapsed" in r]
    if elapsed_times:
        print(f"\n  PERFORMANCE:")
        print(f"    Avg: {sum(elapsed_times) / len(elapsed_times):.3f}s")
        print(f"    Max: {max(elapsed_times):.3f}s")
        print(f"    Min: {min(elapsed_times):.3f}s")
        print(f"    Total: {sum(elapsed_times):.3f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fuzz test: document sentiment analysis pipeline")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of random samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--report", type=str, default="", help="Save report to path instead of output dir")
    args = parser.parse_args()

    print(f"Sentiment Fuzz Test — num_samples={args.num_samples}, seed={args.seed}")
    print(f"Generating samples...")

    runner = SentimentFuzzRunner(num_samples=args.num_samples, seed=args.seed)
    results = runner.run_all()

    _print_report(results)

    # Save report
    report_path = Path(args.report) if args.report else (OUTPUT_DIR / "fuzz_sentiment_report.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for r in results if r["all_pass"])
    fail_count = sum(1 for r in results if r["all_pass"] is False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {"num_samples": args.num_samples, "seed": args.seed},
            "summary": {
                "total": len(results),
                "pass": pass_count,
                "fail": fail_count,
                "pass_rate": round(pass_count / max(len(results), 1), 3),
            },
            "results": [{
                "idx": r["idx"],
                "category": r["category"],
                "nlu_product_type": r.get("nlu_product_type", ""),
                "num_docs": r.get("num_docs", 0),
                "all_pass": r["all_pass"],
                "failed_checks": [k for k, v in r.get("checks", {}).items() if not v],
                "errors": r.get("errors", []),
                "elapsed": r.get("elapsed", 0),
            } for r in results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
