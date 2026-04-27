from __future__ import annotations

from query_intelligence.retrieval.market_analyzer import MarketAnalyzer


def test_rsi_flat_window_is_neutral() -> None:
    closes = [100.0] * 15

    rsi = MarketAnalyzer._rsi(closes, period=14)

    assert rsi == 50.0


def test_rsi_all_gains_is_100() -> None:
    closes = [float(v) for v in range(1, 16)]

    rsi = MarketAnalyzer._rsi(closes, period=14)

    assert rsi == 100.0


def test_market_signal_deduplicates_same_symbol_and_prefers_market_api() -> None:
    analyzer = MarketAnalyzer()
    structured_items = [
        {
            "source_type": "index_daily",
            "payload": {
                "symbol": "000300.SH",
                "canonical_name": "沪深300",
                "close": 3999.0,
                "_market_analysis": {"trend_signal": "neutral", "rsi_14": 48.0},
            },
        },
        {
            "source_type": "market_api",
            "payload": {
                "symbol": "000300.SH",
                "canonical_name": "沪深300",
                "close": 4001.0,
                "_market_analysis": {"trend_signal": "bullish", "rsi_14": 56.0},
            },
        },
    ]

    summary = analyzer.build_analysis_summary(structured_items, nlu_result={"intent_labels": [], "topic_labels": []}, documents=[])

    assert isinstance(summary["market_signal"], dict)
    assert summary["market_signal"]["symbol"] == "000300.SH"
    assert summary["market_signal"]["trend_signal"] == "bullish"
    assert summary["market_signal"]["close"] == 4001.0


def test_fundamental_signal_parses_string_numeric_fields() -> None:
    analyzer = MarketAnalyzer()
    structured_items = [
        {
            "source_type": "fundamental_sql",
            "payload": {
                "symbol": "600519.SH",
                "report_date": "2025-12-31",
                "pe_ttm": "22.8",
                "pb": "7.4",
                "roe": "33.0%",
            },
        }
    ]

    summary = analyzer.build_analysis_summary(structured_items, nlu_result={"intent_labels": [], "topic_labels": []}, documents=[])

    assert isinstance(summary["fundamental_signal"], dict)
    assert summary["fundamental_signal"]["symbol"] == "600519.SH"
    assert summary["fundamental_signal"]["pe_ttm"] == 22.8
    assert summary["fundamental_signal"]["pb"] == 7.4
    assert summary["fundamental_signal"]["roe"] == 33.0
    assert summary["fundamental_signal"]["valuation_assessment"] == "fair_range"
