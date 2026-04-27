"""全面有效性测试：覆盖所有问题类别、关键词多样性、边界情况。

用法:
    python manual_test/test_coverage.py
    python manual_test/test_coverage.py --quick   # 只跑核心用例
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QI_USE_LIVE_MARKET", "1")
os.environ.setdefault("QI_USE_LIVE_MACRO", "1")
os.environ.setdefault("QI_USE_LIVE_NEWS", "1")
os.environ.setdefault("QI_USE_LIVE_ANNOUNCEMENT", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import build_default_service, clear_service_caches


# ── test cases ─────────────────────────────────────────────────────
# (query, expected_product_type, expected_signals, category_label)

TEST_CASES = [
    # ─── 个股行情 ───
    ("茅台今天股价多少", "stock", ["market_signal", "fundamental_signal"], "个股/价格查询"),
    ("贵州茅台现在多少钱", "stock", ["market_signal", "fundamental_signal"], "个股/别名+价格"),
    ("600519今天收盘价", "stock", ["market_signal"], "个股/代码查询"),
    ("宁德时代最近涨了多少", "stock", ["market_signal", "fundamental_signal"], "个股/涨幅查询"),
    ("比亚迪股价走势怎么样", "stock", ["market_signal", "fundamental_signal"], "个股/走势查询"),
    ("中国平安为什么跌", "stock", ["market_signal", "fundamental_signal"], "个股/下跌原因"),

    # ─── 基本面/估值 ───
    ("茅台的PE高不高", "stock", ["fundamental_signal"], "个股/估值查询"),
    ("比亚迪基本面怎么样", "stock", ["fundamental_signal", "market_signal"], "个股/基本面"),
    ("宁德时代ROE多少", "stock", ["fundamental_signal"], "个股/财务指标"),
    ("五粮液估值合理吗", "stock", ["fundamental_signal"], "个股/估值判断"),

    # ─── 买卖建议 ───
    ("茅台现在能买吗", "stock", ["market_signal", "fundamental_signal"], "个股/买入时机"),
    ("中国平安该不该卖", "stock", ["market_signal", "fundamental_signal"], "个股/卖出时机"),
    ("宁德时代还能持有吗", "stock", ["market_signal", "fundamental_signal"], "个股/持有判断"),

    # ─── 风险 ───
    ("茅台有什么风险", "stock", ["market_signal", "fundamental_signal"], "个股/风险分析"),
    ("比亚迪下跌风险大吗", "stock", ["market_signal"], "个股/下跌风险"),

    # ─── 行业 ───
    ("白酒行业最近怎么样", "stock", ["market_signal"], "行业/板块行情"),
    ("半导体板块走势", "stock", ["market_signal"], "行业/板块走势"),
    ("新能源车行业估值", "stock", ["fundamental_signal"], "行业/板块估值"),

    # ─── ETF ───
    ("沪深300ETF净值多少", "etf", ["market_signal"], "ETF/净值查询"),
    ("510300今天涨了吗", "etf", ["market_signal"], "ETF/代码查询"),
    ("中证500ETF值得买吗", "etf", ["market_signal"], "ETF/买入建议"),

    # ─── 基金 ───
    ("易方达蓝筹精选基金怎么样", "fund", ["fundamental_signal"], "基金/产品信息"),
    ("005827基金净值", "fund", ["fundamental_signal"], "基金/代码查询"),

    # ─── 指数 ───
    ("创业板指最近走势", "index", ["market_signal"], "指数/走势查询"),
    ("上证指数今天多少点", "index", ["market_signal"], "指数/点位查询"),
    ("科创50估值高吗", "index", ["market_signal"], "指数/估值查询"),
    ("沪深300指数行情", "index", ["market_signal"], "指数/行情查询"),

    # ─── 宏观 ───
    ("最近CPI和PMI怎么样", "macro", ["macro_signal"], "宏观/指标查询"),
    ("M2增速多少", "macro", ["macro_signal"], "宏观/货币供应"),
    ("中国10年期国债收益率走势", "macro", ["macro_signal"], "宏观/利率"),
    ("最近有什么宏观政策", "macro", ["macro_signal"], "宏观/政策"),

    # ─── 新闻/事件 ───
    ("茅台最近有什么新闻", "stock", ["market_signal"], "个股/新闻查询"),
    ("宁德时代最新消息", "stock", ["market_signal"], "个股/事件查询"),

    # ─── 比较 ───
    ("茅台和五粮液哪个好", "stock", ["market_signal", "fundamental_signal"], "个股/比较"),
    ("创业板和科创板哪个更值得投", "index", ["market_signal"], "指数/比较"),

    # ─── 边界/模糊 ───
    ("最近股市怎么样", "generic_market", ["market_signal"], "泛市场/大盘"),
    ("A股行情", "generic_market", ["market_signal"], "泛市场/A股"),
    ("今天市场怎么样", "generic_market", [], "泛市场/笼统"),
]

QUICK_CASES = [
    ("茅台今天股价多少", "stock", ["market_signal", "fundamental_signal"], "个股/价格"),
    ("最近CPI和PMI怎么样", "macro", ["macro_signal"], "宏观/指标"),
    ("创业板指最近走势", "index", ["market_signal"], "指数/走势"),
    ("沪深300ETF净值多少", "etf", ["market_signal"], "ETF/净值"),
    ("比亚迪基本面怎么样", "stock", ["fundamental_signal", "market_signal"], "个股/基本面"),
    ("茅台和五粮液哪个好", "stock", ["market_signal", "fundamental_signal"], "个股/比较"),
    ("白酒行业最近怎么样", "stock", ["market_signal"], "行业/板块"),
]


# ── runner ──────────────────────────────────────────────────────────
def run_tests(cases: list[tuple], verbose: bool = False) -> list[dict]:
    clear_service_caches()
    svc = build_default_service()
    results = []

    for i, (query, expected_type, expected_signals, label) in enumerate(cases, 1):
        t0 = time.time()
        try:
            nlu = svc.analyze_query(query, debug=False)
            rr = svc.retrieve_evidence(nlu, debug=False)
            elapsed = time.time() - t0

            summary = rr.get("analysis_summary") or {}
            readiness = summary.get("data_readiness") or {}
            nlu_snap = rr.get("nlu_snapshot") or {}
            actual_type = nlu_snap.get("product_type", "?")

            # Check signals
            signal_results = {}
            for sig in ["market_signal", "fundamental_signal", "macro_signal"]:
                present = summary.get(sig) is not None
                signal_results[sig] = present

            # Check data_readiness
            readiness_flags = {k: v for k, v in readiness.items() if k.startswith("has_")}

            # Determine pass/fail
            type_ok = actual_type == expected_type
            signals_ok = all(signal_results.get(s, False) for s in expected_signals)

            status = "PASS" if (type_ok and signals_ok) else "FAIL"

            result = {
                "idx": i,
                "query": query,
                "label": label,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "type_ok": type_ok,
                "expected_signals": expected_signals,
                "signal_results": signal_results,
                "signals_ok": signals_ok,
                "readiness": readiness_flags,
                "status": status,
                "elapsed": round(elapsed, 1),
            }

            if verbose:
                # Show detail for each signal
                detail_parts = []
                ms = summary.get("market_signal")
                if isinstance(ms, list):
                    ms = ms[0] if ms and isinstance(ms[0], dict) else None
                if isinstance(ms, dict):
                    detail_parts.append(f"trend={ms.get('trend_signal')} RSI={ms.get('rsi_14')}")
                fs = summary.get("fundamental_signal")
                if isinstance(fs, list):
                    fs = fs[0] if fs and isinstance(fs[0], dict) else None
                if isinstance(fs, dict):
                    detail_parts.append(f"PE={fs.get('pe_ttm')} PB={fs.get('pb')} ROE={fs.get('roe')} val={fs.get('valuation_assessment')}")
                mcs = summary.get("macro_signal")
                if mcs:
                    detail_parts.append(f"overall={mcs.get('overall')}")
                result["detail"] = " | ".join(detail_parts) if detail_parts else "—"

            results.append(result)

        except Exception as exc:
            results.append({
                "idx": i,
                "query": query,
                "label": label,
                "expected_type": expected_type,
                "status": "ERROR",
                "error": str(exc)[:80],
                "elapsed": round(time.time() - t0, 1),
            })

    return results


def print_report(results: list[dict], verbose: bool = False):
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print("\n" + "=" * 70)
    print(f"  测试结果: {passed}/{total} PASS  |  {failed} FAIL  |  {errors} ERROR")
    print("=" * 70)

    # ── By category ──
    categories: dict[str, list] = {}
    for r in results:
        cat = r["label"].split("/")[0]
        categories.setdefault(cat, []).append(r)

    print("\n## 按类别统计\n")
    print(f"  {'类别':<10} {'通过':>4} {'失败':>4} {'错误':>4} {'总计':>4}")
    print(f"  {'─'*10} {'─'*4} {'─'*4} {'─'*4} {'─'*4}")
    for cat, items in sorted(categories.items()):
        p = sum(1 for r in items if r["status"] == "PASS")
        f = sum(1 for r in items if r["status"] == "FAIL")
        e = sum(1 for r in items if r["status"] == "ERROR")
        print(f"  {cat:<10} {p:>4} {f:>4} {e:>4} {len(items):>4}")

    # ── Signal coverage ──
    print("\n## 信号覆盖统计\n")
    for sig in ["market_signal", "fundamental_signal", "macro_signal"]:
        triggered = sum(1 for r in results if r.get("signal_results", {}).get(sig))
        total_applicable = sum(1 for r in results if sig in r.get("expected_signals", []))
        print(f"  {sig:<22} 触发 {triggered:>2}/{total_applicable:>2} 次")

    # ── Data readiness coverage ──
    print("\n## data_readiness 覆盖统计\n")
    readiness_keys = ["has_price_data", "has_fundamentals", "has_macro", "has_news", "has_technical_indicators"]
    for key in readiness_keys:
        true_count = sum(1 for r in results if r.get("readiness", {}).get(key))
        print(f"  {key:<28} true={true_count:>2}/{total}")

    # ── Product type accuracy ──
    print("\n## product_type 准确率\n")
    type_results: dict[str, dict] = {}
    for r in results:
        exp = r.get("expected_type", "?")
        if exp not in type_results:
            type_results[exp] = {"correct": 0, "total": 0}
        type_results[exp]["total"] += 1
        if r.get("type_ok"):
            type_results[exp]["correct"] += 1
    for t, v in sorted(type_results.items()):
        pct = v["correct"] / v["total"] * 100 if v["total"] else 0
        print(f"  {t:<18} {v['correct']}/{v['total']}  ({pct:.0f}%)")

    # ── Detail per case ──
    print("\n## 逐条结果\n")
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}.get(r["status"], "?")
        line = f"  {icon} [{r['idx']:>2}] {r['label']:<12} | {r['query'][:25]:<25} | {r.get('actual_type', '?'):<8} | {r['elapsed']:>4.1f}s"
        if verbose and r.get("detail"):
            line += f"\n       {r['detail']}"
        if r["status"] == "FAIL":
            exp_sigs = ", ".join(r.get("expected_signals", []))
            actual_sigs = {k for k, v in r.get("signal_results", {}).items() if v}
            line += f"\n       ⚠ expected=[{exp_sigs}] actual=[{', '.join(sorted(actual_sigs)) or 'none'}]"
        if r["status"] == "ERROR":
            line += f"\n       ⚠ {r.get('error', '')}"
        print(line)

    # ── Timing ──
    times = [r["elapsed"] for r in results if r["status"] != "ERROR"]
    if times:
        print(f"\n## 耗时统计\n")
        print(f"  平均: {sum(times)/len(times):.1f}s  最快: {min(times):.1f}s  最慢: {max(times):.1f}s  总计: {sum(times):.0f}s")


def main():
    parser = argparse.ArgumentParser(description="全面有效性测试")
    parser.add_argument("--quick", action="store_true", help="只跑核心用例(7条)")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示信号详情")
    args = parser.parse_args()

    cases = QUICK_CASES if args.quick else TEST_CASES
    print(f"运行 {len(cases)} 条测试用例...\n")
    results = run_tests(cases, verbose=args.verbose)
    print_report(results, verbose=args.verbose)

    # Save JSON report
    report_dir = Path(__file__).resolve().parent / "output"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
