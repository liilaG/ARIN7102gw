"""Pytest tests for the retrieval pipeline analysis_summary feature."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QI_USE_LIVE_MARKET", "0")
os.environ.setdefault("QI_USE_LIVE_MACRO", "0")
os.environ.setdefault("QI_USE_LIVE_NEWS", "0")

from query_intelligence.contracts import RetrievalResult
from query_intelligence.service import build_default_service, clear_service_caches


def _require_integration_tests_enabled() -> None:
    if os.environ.get("QI_INTEGRATION_TESTS") != "1":
        pytest.skip("live integration tests disabled; set QI_INTEGRATION_TESTS=1 to enable")


def _as_signal_item(signal: object) -> dict | None:
    if isinstance(signal, dict):
        return signal
    if isinstance(signal, list):
        for item in signal:
            if isinstance(item, dict):
                return item
    return None


TEST_CASES = [
    {
        "query": "茅台今天股价多少",
        "desc": "个股行情查询",
        "expect": {"market_signal": True, "fundamental_signal": True, "macro_signal": False},
    },
    {
        "query": "最近CPI和PMI怎么样",
        "desc": "宏观指标查询",
        "expect": {"market_signal": False, "fundamental_signal": False, "macro_signal": True},
    },
    {
        "query": "沪深300ETF今天净值多少",
        "desc": "ETF净值查询",
        "expect": {"market_signal": True, "fundamental_signal": False, "macro_signal": False},
    },
    {
        "query": "比亚迪基本面怎么样",
        "desc": "基本面查询",
        "expect": {"market_signal": True, "fundamental_signal": True, "macro_signal": False},
    },
    {
        "query": "创业板指最近走势",
        "desc": "指数走势查询",
        "expect": {"market_signal": True, "fundamental_signal": False, "macro_signal": False},
    },
]


@pytest.fixture(scope="module")
def service():
    _require_integration_tests_enabled()
    clear_service_caches()
    return build_default_service()


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_analysis_summary_exists(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    assert summary, "analysis_summary should not be empty"


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_expected_signals_present(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    readiness = summary.get("data_readiness", {})
    for sig_name, expected in tc["expect"].items():
        if not expected:
            continue
        if sig_name == "market_signal":
            if not readiness.get("has_price_data"):
                pytest.skip("price data unavailable for this query")
            if readiness.get("has_technical_indicators"):
                assert _as_signal_item(summary.get("market_signal")) is not None
        elif sig_name == "fundamental_signal":
            if not readiness.get("has_fundamentals"):
                pytest.skip("fundamental data unavailable for this query")
            assert _as_signal_item(summary.get("fundamental_signal")) is not None
        elif sig_name == "macro_signal":
            if not readiness.get("has_macro"):
                pytest.skip("macro data unavailable for this query")
            assert _as_signal_item(summary.get("macro_signal")) is not None


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_market_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    ms = _as_signal_item(summary.get("market_signal"))
    if not ms:
        pytest.skip("no market_signal for this query")

    assert ms["trend_signal"] in ("bullish", "bearish", "neutral")
    rsi = ms.get("rsi_14")
    assert rsi is None or 0 <= rsi <= 100
    macd = ms.get("macd")
    assert macd is None or all(k in macd for k in ("macd_line", "signal_line", "histogram"))
    bb = ms.get("bollinger")
    assert bb is None or all(k in bb for k in ("upper", "middle", "lower"))
    pct = ms.get("pct_change_nd")
    assert pct is None or any(k in pct for k in ("pct_3d", "pct_5d", "pct_10d", "pct_20d"))
    vol = ms.get("volatility_20d")
    assert vol is None or vol >= 0


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_macro_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    mcs = _as_signal_item(summary.get("macro_signal"))
    if not mcs:
        pytest.skip("no macro_signal for this query")

    assert len(mcs.get("indicators", [])) > 0
    assert mcs["overall"] in ("expansionary", "contractionary", "mixed")
    for ind in mcs["indicators"]:
        assert ind.get("direction") != "unknown" or ind.get("metric_value") is None


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_fundamental_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    fs = _as_signal_item(summary.get("fundamental_signal"))
    if not fs:
        pytest.skip("no fundamental_signal for this query")

    assert fs["valuation_assessment"] in (
        "potentially_undervalued",
        "fair_range",
        "potentially_overvalued",
        "unknown",
    )


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_data_readiness(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    readiness = summary.get("data_readiness", {})
    assert readiness, "data_readiness should exist"
    assert isinstance(readiness.get("has_price_data"), bool)
    assert isinstance(readiness.get("has_news"), bool)


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_coverage_consistency(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    coverage = rr.get("coverage", {})

    ms = summary.get("market_signal")
    if ms:
        assert coverage.get("price"), "market_signal exists but coverage.price is False"

    mcs = summary.get("macro_signal")
    if mcs:
        assert coverage.get("macro"), "macro_signal exists but coverage.macro is False"


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_retrieval_result_pydantic(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    RetrievalResult(**rr)  # should not raise
