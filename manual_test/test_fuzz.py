"""Fuzz test: generate boundary queries from runtime entity data and verify pipeline stability.

Reads entity_master.csv and alias_table.csv from data/runtime/ to build
diverse edge-case queries, then runs the full retrieval pipeline on each
and checks that:
  1. No crash / exception
  2. Output is valid RetrievalResult (schema + Pydantic)
  3. analysis_summary exists (even if empty)
  4. Key fields are present and have correct types
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import QueryIntelligenceService, build_default_service

OUTPUT_DIR = ROOT / "manual_test" / "output"
SCHEMA_PATH = ROOT / "schemas" / "retrieval_result.schema.json"

# Load JSON schema for validation
_SCHEMA = None
if jsonschema is not None and SCHEMA_PATH.exists():
    with SCHEMA_PATH.open("r", encoding="utf-8") as _f:
        _SCHEMA = json.load(_f)

# ── Load runtime entity data ──────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_entities() -> list[dict]:
    return _load_csv(ROOT / "data" / "runtime" / "entity_master.csv")


def _load_aliases() -> list[dict]:
    return _load_csv(ROOT / "data" / "runtime" / "alias_table.csv")


# ── Query generators ────────────────────────────────────────────────────

TEMPLATES_STOCK = [
    "{name}今天股价多少",
    "{name}怎么样",
    "{name}能买吗",
    "{name}为什么跌",
    "{name}估值高不高",
    "{name}基本面怎么样",
    "{name}后面还值得拿吗",
    "{code}行情",
    "{name}最近涨了多少",
    "{name}风险大吗",
]

TEMPLATES_ETF = [
    "{name}最近怎么样",
    "{name}适合定投吗",
    "{name}费率高吗",
    "{code}ETF走势",
    "{name}和沪深300ETF哪个好",
]

TEMPLATES_INDEX = [
    "{name}今天涨了吗",
    "{name}走势怎么样",
    "{name}为什么跌",
    "{code}指数行情",
]

TEMPLATES_FUND = [
    "{name}净值多少",
    "{name}怎么样",
    "{name}费率高吗",
    "{code}基金怎么样",
]

TEMPLATES_MACRO = [
    "CPI对市场有什么影响",
    "PMI最近怎么样",
    "M2增速多少",
    "降息对股市有什么影响",
    "宏观政策会影响{industry}吗",
    "10年期国债收益率走势",
]

TEMPLATES_BOUNDARY = [
    "123456",               # pure digits
    "ABCDEF",               # pure english
    "!!!???###",            # special chars
    "茅台茅台茅台茅台茅台",  # repeated
    "茅台12345ABC!@#",      # mixed
    "不知道问什么",          # vague
    "你好",                 # greeting
    "今天天气怎么样",        # off-topic
    "帮我翻译一下hello",    # utility
    "茅台和五粮液和泸州老窖哪个好",  # multi-compare
]


def _build_queries(entities: list[dict], aliases: list[dict], max_per_type: int = 50) -> list[dict]:
    """Generate fuzz queries from entity data."""
    random.seed(42)
    queries: list[dict] = []
    alias_map: dict[str, list[str]] = {}
    for a in aliases:
        eid = a.get("entity_id", "")
        alias_map.setdefault(eid, []).append(a.get("alias_text", ""))

    # Group entities by type
    by_type: dict[str, list[dict]] = {}
    for e in entities:
        etype = e.get("entity_type", "unknown")
        by_type.setdefault(etype, []).append(e)

    # Stock queries
    template_map = {
        "stock": TEMPLATES_STOCK,
        "etf": TEMPLATES_ETF,
        "index": TEMPLATES_INDEX,
        "fund": TEMPLATES_FUND,
    }

    for etype, templates in template_map.items():
        pool = by_type.get(etype, [])
        if not pool:
            continue
        sample = random.sample(pool, min(len(pool), max_per_type))
        for e in sample:
            tmpl = random.choice(templates)
            name = e.get("canonical_name", "")
            code = e.get("symbol", "").split(".")[0]
            industry = e.get("industry_name", "")
            q = tmpl.format(name=name, code=code, industry=industry)
            queries.append({
                "query": q,
                "category": f"fuzz_{etype}",
                "entity_type": etype,
                "symbol": e.get("symbol", ""),
            })

    # Alias-based queries (use alias names instead of canonical)
    stock_entities = by_type.get("stock", [])[:20]
    for e in stock_entities:
        eid = e.get("entity_id", "")
        names = alias_map.get(eid, [])
        if names:
            alias_name = random.choice(names)
            q = f"{alias_name}怎么样"
            queries.append({
                "query": q,
                "category": "fuzz_alias",
                "entity_type": "stock",
                "symbol": e.get("symbol", ""),
            })

    # Macro queries
    for tmpl in TEMPLATES_MACRO:
        industries = [e.get("industry_name", "") for e in by_type.get("stock", []) if e.get("industry_name")]
        industry = random.choice(industries) if industries else "白酒"
        q = tmpl.format(industry=industry)
        queries.append({
            "query": q,
            "category": "fuzz_macro",
            "entity_type": "macro",
            "symbol": "",
        })

    # Boundary / adversarial queries
    for q in TEMPLATES_BOUNDARY:
        queries.append({
            "query": q,
            "category": "fuzz_boundary",
            "entity_type": "unknown",
            "symbol": "",
        })

    return queries


# ── Validation ──────────────────────────────────────────────────────────

def _validate_result(result: dict, query_info: dict) -> dict:
    """Check a RetrievalResult for structural validity. Returns check dict."""
    checks: dict[str, bool] = {}

    # Must have top-level keys
    for key in ["query_id", "nlu_snapshot", "executed_sources", "documents",
                "structured_data", "coverage", "warnings", "retrieval_confidence",
                "analysis_summary", "debug_trace"]:
        checks[f"has_{key}"] = key in result

    # analysis_summary must be a dict
    summary = result.get("analysis_summary", {})
    checks["analysis_summary_is_dict"] = isinstance(summary, dict)

    # data_readiness must be a dict if present
    if "data_readiness" in summary:
        checks["data_readiness_is_dict"] = isinstance(summary["data_readiness"], dict)

    # retrieval_confidence in [0, 1]
    conf = result.get("retrieval_confidence")
    checks["confidence_in_range"] = isinstance(conf, (int, float)) and 0 <= conf <= 1

    # nlu_snapshot must have product_type
    nlu = result.get("nlu_snapshot", {})
    checks["nlu_has_product_type"] = "product_type" in nlu

    # For non-boundary queries, product_type should not be empty
    if query_info["category"] != "fuzz_boundary" and query_info["query"].strip():
        checks["product_type_not_empty"] = bool(nlu.get("product_type"))

    # JSON Schema validation (if jsonschema available)
    if _SCHEMA is not None:
        try:
            jsonschema.validate(instance=result, schema=_SCHEMA)
            checks["json_schema_valid"] = True
        except jsonschema.ValidationError:
            checks["json_schema_valid"] = False

    all_pass = all(checks.values())
    return {"all_pass": all_pass, "checks": checks}


# ── Main ────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fuzz test: verify pipeline stability across diverse queries")
    parser.add_argument("--max-per-type", type=int, default=10, help="Max entities per type to sample")
    parser.add_argument("--live-market", dest="live_market", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live market data (default: on)")
    parser.add_argument("--live-macro", dest="live_macro", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live macro data (default: on)")
    parser.add_argument("--live-news", dest="live_news", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable live news data (default: on)")
    parser.add_argument("--live-announcement", dest="live_announcement", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable live announcement data (default: off — cninfo often hangs from overseas)")
    return parser.parse_args()


def main():
    args = _parse_args()

    print("Loading entity data...")
    entities = _load_entities()
    aliases = _load_aliases()
    print(f"  entities: {len(entities)}, aliases: {len(aliases)}")

    print("Generating fuzz queries...")
    queries = _build_queries(entities, aliases, max_per_type=args.max_per_type)
    print(f"  generated {len(queries)} queries")

    print("Initializing service...")
    print(f"  Live providers: market={args.live_market}  macro={args.live_macro}  news={args.live_news}  announcement={args.live_announcement}")
    service = build_default_service(
        use_live_market=args.live_market,
        use_live_macro=args.live_macro,
        use_live_news=args.live_news,
        use_live_announcement=args.live_announcement,
    )

    results: list[dict] = []
    exception_count = 0
    pass_count = 0
    fail_count = 0

    for i, qi in enumerate(queries):
        query = qi["query"]
        cat = qi["category"]
        label = f"[{i+1}/{len(queries)}] ({cat})"

        try:
            t0 = time.time()
            combined = service.run_pipeline(query)
            rr = combined.get("retrieval_result", {})
            elapsed = time.time() - t0
            validation = _validate_result(rr, qi)
            validation["elapsed"] = round(elapsed, 1)
        except Exception as exc:
            rr = {}
            validation = {"all_pass": False, "checks": {"exception": False}, "error": str(exc)[:200]}
            exception_count += 1
            elapsed = -1

        status = "PASS" if validation["all_pass"] else "FAIL"
        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        result_row = {
            "idx": i + 1,
            "query": query[:80],
            "category": cat,
            "entity_type": qi["entity_type"],
            "symbol": qi.get("symbol", ""),
            "status": status,
            "elapsed": validation.get("elapsed", -1),
            "checks": validation.get("checks", {}),
        }
        if "error" in validation:
            result_row["error"] = validation["error"]

        results.append(result_row)

        # Print progress
        failed_checks = [k for k, v in validation.get("checks", {}).items() if not v]
        check_str = "OK" if not failed_checks else f"FAIL:{','.join(failed_checks)}"
        print(f"  {label} {query[:40]:<40} {status} ({check_str}) {validation.get('elapsed', '?')}s")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FUZZ TEST SUMMARY")
    print(f"  Total:   {len(queries)}")
    print(f"  PASS:    {pass_count}")
    print(f"  FAIL:    {fail_count}")
    print(f"  EXCEPTION: {exception_count}")
    print(f"  Pass rate: {pass_count / max(len(queries), 1) * 100:.1f}%")

    # By category
    print("\n  BY CATEGORY:")
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, items in sorted(cats.items()):
        p = sum(1 for r in items if r["status"] == "PASS")
        f = sum(1 for r in items if r["status"] == "FAIL")
        print(f"    {cat:<20} PASS={p} FAIL={f} total={len(items)}")

    # Common failure patterns
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print("\n  FAILURE BREAKDOWN:")
        check_fails: dict[str, int] = {}
        for r in failures:
            for k, v in r.get("checks", {}).items():
                if not v:
                    check_fails[k] = check_fails.get(k, 0) + 1
        for k, v in sorted(check_fails.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "fuzz_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(queries),
                "pass": pass_count,
                "fail": fail_count,
                "crash": exception_count,
                "pass_rate": round(pass_count / max(len(queries), 1), 3),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
