from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Computes technical indicators and trend signals from market history data."""

    def enrich_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add computed analysis fields to a market_api payload in-place and return it."""
        history = payload.get("history") or []
        if len(history) < 2:
            return payload

        closes = [row.get("close") for row in history if row.get("close") is not None]
        if len(closes) < 2:
            return payload

        # Providers return history in latest-first order; reverse so
        # closes[-1] is the most recent close and tail slices are recent.
        closes = list(reversed(closes))

        analysis: dict[str, Any] = {}

        # Multi-day returns
        analysis["pct_change_nd"] = self._multi_day_returns(closes)

        # Moving averages
        analysis["ma5"] = self._sma(closes, 5)
        analysis["ma20"] = self._sma(closes, 20)

        # RSI(14)
        analysis["rsi_14"] = self._rsi(closes, 14)

        # MACD
        macd_result = self._macd(closes)
        if macd_result:
            analysis["macd"] = macd_result

        # Volatility (20-day annualized)
        analysis["volatility_20d"] = self._volatility(closes, 20)

        # Bollinger Bands (20,2)
        bb = self._bollinger_bands(closes, 20, 2)
        if bb:
            analysis["bollinger"] = bb

        # Trend signal
        analysis["trend_signal"] = self._trend_signal(closes, analysis)

        # Price position relative to MA
        latest_close = closes[-1]
        ma5 = analysis.get("ma5")
        ma20 = analysis.get("ma20")
        analysis["price_vs_ma"] = {
            "above_ma5": latest_close > ma5 if ma5 is not None else None,
            "above_ma20": latest_close > ma20 if ma20 is not None else None,
        }

        payload["_market_analysis"] = analysis
        return payload

    def build_analysis_summary(
        self,
        structured_items: list[dict],
        nlu_result: dict,
        documents: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Build a top-level analysis_summary for downstream sentiment analysis module."""
        summary: dict[str, Any] = {
            "market_signal": None,
            "fundamental_signal": None,
            "macro_signal": None,
            "data_readiness": {},
        }

        market_signal_by_key: dict[str, dict[str, Any]] = {}
        fundamental_signals: list[dict] = []

        for item in structured_items:
            source_type = item.get("source_type", "")
            payload = item.get("payload", {})
            analysis = payload.get("_market_analysis")

            if source_type in {"market_api", "index_daily"} and analysis:
                signal = self._summarize_market_signal(payload, analysis)
                signal_key = str(signal.get("symbol") or signal.get("canonical_name") or "")
                if not signal_key:
                    signal_key = f"__market_signal_{len(market_signal_by_key)}"
                existing = market_signal_by_key.get(signal_key)
                if existing is None or source_type == "market_api":
                    market_signal_by_key[signal_key] = signal
            elif source_type == "fundamental_sql":
                fundamental_signals.append(self._summarize_fundamental_signal(payload))
            elif source_type in {"macro_sql", "macro_indicator"}:
                if summary["macro_signal"] is None:
                    summary["macro_signal"] = {"indicators": [], "overall": None}
                indicator = self._summarize_macro_indicator(payload)
                if indicator:
                    summary["macro_signal"]["indicators"].append(indicator)

        # Compute macro overall direction
        if summary["macro_signal"] and summary["macro_signal"]["indicators"]:
            summary["macro_signal"]["overall"] = self._macro_overall(summary["macro_signal"]["indicators"])

        market_signals = list(market_signal_by_key.values())

        # Assign market/fundamental signals (single → object, multi → list)
        if len(market_signals) == 1:
            summary["market_signal"] = market_signals[0]
        elif market_signals:
            summary["market_signal"] = market_signals
        if len(fundamental_signals) == 1:
            summary["fundamental_signal"] = fundamental_signals[0]
        elif fundamental_signals:
            summary["fundamental_signal"] = fundamental_signals

        # Data readiness
        intent_labels = {item["label"] for item in nlu_result.get("intent_labels", [])}
        topic_labels = {item["label"] for item in nlu_result.get("topic_labels", [])}
        source_types = {item.get("source_type", "") for item in structured_items}
        summary["data_readiness"] = {
            "has_price_data": bool(source_types.intersection({"market_api", "index_daily", "fund_nav"})),
            "has_fundamentals": "fundamental_sql" in source_types,
            "has_macro": bool(source_types.intersection({"macro_sql", "macro_indicator"})),
            "has_news": any(doc.get("source_type") == "news" for doc in (documents or [])),
            "has_technical_indicators": any(
                item.get("payload", {}).get("_market_analysis") is not None
                for item in structured_items
                if item.get("source_type") in {"market_api", "index_daily"}
            ),
            "relevant_intents": sorted(intent_labels),
            "relevant_topics": sorted(topic_labels),
        }

        return summary

    # --- Technical Indicators ---

    @staticmethod
    def _sma(closes: list[float], period: int) -> float | None:
        if len(closes) < period:
            return None
        return round(sum(closes[-period:]) / period, 4)

    @staticmethod
    def _multi_day_returns(closes: list[float]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for n, label in [(3, "pct_3d"), (5, "pct_5d"), (10, "pct_10d"), (20, "pct_20d")]:
            if len(closes) > n and closes[-n - 1] != 0:
                result[label] = round((closes[-1] - closes[-n - 1]) / abs(closes[-n - 1]) * 100, 4)
            else:
                result[label] = None
        return result

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 50.0 if avg_gain == 0 else 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    @staticmethod
    def _macd(
        closes: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, float] | None:
        if len(closes) < slow:
            return None
        ema_fast = MarketAnalyzer._ema(closes, fast)
        ema_slow = MarketAnalyzer._ema(closes, slow)
        if ema_fast is None or ema_slow is None:
            return None
        macd_line = round(ema_fast - ema_slow, 4)

        # Compute MACD values at each trailing window for signal line
        macd_values: list[float] = []
        for offset in range(min(signal, len(closes) - slow + 1)):
            window = closes[:len(closes) - offset] if offset > 0 else closes
            ef = MarketAnalyzer._ema(window, fast)
            es = MarketAnalyzer._ema(window, slow)
            if ef is not None and es is not None:
                macd_values.append(ef - es)
        if macd_values:
            signal_line = round(sum(macd_values) / len(macd_values), 4)
        else:
            signal_line = macd_line

        histogram = round(macd_line - signal_line, 4)
        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def _ema(data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return round(ema, 4)

    @staticmethod
    def _volatility(closes: list[float], period: int = 20) -> float | None:
        if len(closes) < period + 1:
            return None
        returns = [
            (closes[i] - closes[i - 1]) / abs(closes[i - 1])
            for i in range(len(closes) - period, len(closes))
            if closes[i - 1] != 0
        ]
        if len(returns) < 2:
            return None
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        # Annualize (252 trading days)
        return round(variance**0.5 * (252**0.5) * 100, 2)

    @staticmethod
    def _bollinger_bands(closes: list[float], period: int = 20, num_std: float = 2.0) -> dict[str, float] | None:
        if len(closes) < period:
            return None
        slice_closes = closes[-period:]
        sma = sum(slice_closes) / period
        variance = sum((c - sma) ** 2 for c in slice_closes) / period
        std = variance**0.5
        return {
            "upper": round(sma + num_std * std, 4),
            "middle": round(sma, 4),
            "lower": round(sma - num_std * std, 4),
            "bandwidth": round(2 * num_std * std / sma * 100, 4) if sma != 0 else None,
        }

    @staticmethod
    def _trend_signal(closes: list[float], analysis: dict) -> str:
        """Determine trend: bullish / bearish / neutral based on MA cross, RSI, and price action."""
        latest = closes[-1]
        ma5 = analysis.get("ma5")
        ma20 = analysis.get("ma20")
        rsi = analysis.get("rsi_14")
        pct_5d = (analysis.get("pct_change_nd") or {}).get("pct_5d")

        bullish_signals = 0
        bearish_signals = 0

        # Price vs MA
        if ma5 is not None:
            if latest > ma5:
                bullish_signals += 1
            elif latest < ma5:
                bearish_signals += 1
        if ma20 is not None:
            if latest > ma20:
                bullish_signals += 1
            elif latest < ma20:
                bearish_signals += 1

        # MA cross
        if ma5 is not None and ma20 is not None:
            if ma5 > ma20:
                bullish_signals += 1
            elif ma5 < ma20:
                bearish_signals += 1

        # RSI
        if rsi is not None:
            if rsi > 60:
                bullish_signals += 1
            elif rsi < 40:
                bearish_signals += 1

        # 5-day momentum
        if pct_5d is not None:
            if pct_5d > 2:
                bullish_signals += 1
            elif pct_5d < -2:
                bearish_signals += 1

        if bullish_signals >= 3:
            return "bullish"
        if bearish_signals >= 3:
            return "bearish"
        return "neutral"

    # --- Signal Summaries for Downstream ---

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1]
            try:
                return float(text)
            except ValueError:
                return None
        return None

    @staticmethod
    def _summarize_market_signal(payload: dict, analysis: dict) -> dict[str, Any]:
        return {
            "symbol": payload.get("symbol"),
            "canonical_name": payload.get("canonical_name"),
            "trade_date": payload.get("trade_date"),
            "close": payload.get("close"),
            "pct_change_1d": payload.get("pct_change_1d"),
            "trend_signal": analysis.get("trend_signal"),
            "rsi_14": analysis.get("rsi_14"),
            "ma5": analysis.get("ma5"),
            "ma20": analysis.get("ma20"),
            "pct_change_nd": analysis.get("pct_change_nd"),
            "volatility_20d": analysis.get("volatility_20d"),
            "macd": analysis.get("macd"),
            "bollinger": analysis.get("bollinger"),
            "price_vs_ma": analysis.get("price_vs_ma"),
        }

    @staticmethod
    def _summarize_fundamental_signal(payload: dict) -> dict[str, Any]:
        pe_ttm = MarketAnalyzer._to_float(payload.get("pe_ttm"))
        pb = MarketAnalyzer._to_float(payload.get("pb"))
        roe = MarketAnalyzer._to_float(payload.get("roe"))
        # Simple valuation assessment
        valuation = "unknown"
        if pe_ttm is not None:
            if pe_ttm <= 0:
                valuation = "unknown"  # negative/zero PE → loss-making or invalid
            elif pe_ttm < 15:
                valuation = "potentially_undervalued"
            elif pe_ttm > 40:
                valuation = "potentially_overvalued"
            else:
                valuation = "fair_range"
        return {
            "symbol": payload.get("symbol"),
            "report_date": payload.get("report_date"),
            "pe_ttm": pe_ttm,
            "pb": pb,
            "roe": roe,
            "valuation_assessment": valuation,
        }

    @staticmethod
    def _summarize_macro_indicator(payload: dict) -> dict[str, Any] | None:
        code = payload.get("indicator_code")
        value = payload.get("metric_value")
        if not code:
            return None
        direction = "unknown"
        if value is not None:
            if code == "CPI_CN":
                direction = "inflationary" if value > 2.5 else ("deflationary" if value < 0.5 else "stable")
            elif code == "PMI_CN":
                direction = "expansion" if value > 50 else "contraction"
            elif code == "M2_CN":
                direction = "expansionary" if value > 10 else ("tightening" if value < 8 else "moderate")
            elif code == "CN10Y":
                direction = "rising" if value > 3 else ("falling" if value < 2.5 else "stable")
        return {
            "indicator_code": code,
            "indicator_name": payload.get("indicator_name"),
            "metric_date": payload.get("metric_date"),
            "metric_value": value,
            "unit": payload.get("unit"),
            "direction": direction,
        }

    @staticmethod
    def _macro_overall(indicators: list[dict]) -> str:
        expansion_count = sum(1 for ind in indicators if ind.get("direction") in {"expansion", "expansionary", "rising"})
        contraction_count = sum(1 for ind in indicators if ind.get("direction") in {"contraction", "tightening", "deflationary", "falling"})
        # "stable" and "moderate" count as neither → lean toward "mixed"
        if expansion_count > contraction_count:
            return "expansionary"
        if contraction_count > expansion_count:
            return "contractionary"
        return "mixed"
