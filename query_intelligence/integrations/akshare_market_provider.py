from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urlencode

import requests

from .efinance_provider import EFinanceETFProvider


logger = logging.getLogger(__name__)


def _rows_to_records(rows):
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return []


@dataclass
class AKShareMarketProvider:
    ak_module: object
    efinance_provider: EFinanceETFProvider | None = None
    timeout: int = 15
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25

    @classmethod
    def from_import(cls, *, timeout: int = 15, max_retries: int = 1, retry_backoff_seconds: float = 0.25) -> "AKShareMarketProvider":
        import akshare as ak

        return cls(
            ak_module=ak,
            efinance_provider=EFinanceETFProvider.from_import(),
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    def fetch_bundle(self, symbol: str, canonical_name: str, product_type: str, start_date: str = "20250101", end_date: str = "20261231") -> dict:
        plain_symbol = symbol.split(".")[0]
        market_rows, market_source_name, provider_warnings = self._fetch_market_rows(plain_symbol, product_type, start_date, end_date)
        latest = market_rows[0] if market_rows else {}
        is_fund_product = product_type in {"etf", "fund"}
        is_index_product = product_type == "index"
        company_info = self._safe_fetch_company_info(plain_symbol) if product_type == "stock" else {}
        industry_name = company_info.get("industry_name")
        industry_payload = self._safe_fetch_industry_snapshot(industry_name) if industry_name else None
        if industry_name and not industry_payload:
            industry_payload = self._identity_industry_snapshot(plain_symbol, industry_name)
        fundamental_payload = self._safe_fetch_fundamental_payload(plain_symbol) if product_type == "stock" else {}
        fund_payloads = self._safe_fetch_fund_payloads(plain_symbol) if is_fund_product else {}
        index_payloads = self._safe_fetch_index_payloads(plain_symbol, market_rows) if is_index_product else {}
        market_trace = self._market_api_trace(plain_symbol, product_type, market_source_name, start_date, end_date)

        payload = {
            "symbol": symbol,
            "source_name": market_source_name,
            **market_trace,
            "canonical_name": canonical_name,
            "trade_date": latest.get("trade_date"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "pct_change_1d": latest.get("pct_change_1d"),
            "volume": latest.get("volume"),
            "amount": latest.get("amount"),
            "history": market_rows[:30],
            "industry_name": industry_name,
        }
        if provider_warnings:
            payload["provider_warnings"] = provider_warnings
        if industry_payload:
            payload["industry_snapshot"] = industry_payload

        bundle = {
            "source_type": "market_api",
            "source_name": market_source_name,
            "payload": payload,
            "fundamental_payload": fundamental_payload,
            "provider_warnings": provider_warnings,
            "request_trace": {
                "symbol": symbol,
                "product_type": product_type,
                "start_date": start_date,
                "end_date": end_date,
            },
            "status": "degraded" if provider_warnings or not market_rows else "ok",
        }
        bundle.update(fund_payloads)
        bundle.update(index_payloads)
        return bundle

    def _api_trace(self, endpoint: str, query_params: dict) -> dict:
        encoded_params = urlencode(query_params, doseq=True)
        return {
            "provider": self._provider_name_for_endpoint(endpoint),
            "provider_endpoint": endpoint,
            "query_params": query_params,
            "source_reference": f"api://{endpoint}?{encoded_params}" if encoded_params else f"api://{endpoint}",
        }

    def _market_api_trace(
        self,
        symbol: str,
        product_type: str,
        source_name: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        endpoint = self._market_endpoint(symbol, product_type, source_name)
        query_params: dict[str, str] = {"symbol": symbol}
        if endpoint in {"akshare.stock_zh_a_hist", "akshare.fund_etf_hist_em", "akshare.stock_zh_a_daily"}:
            query_params.update({"period": "daily", "start_date": start_date, "end_date": end_date, "adjust": ""})
        elif endpoint == "akshare.stock_zh_index_daily":
            query_params = {"symbol": self._prefixed_index_symbol(symbol)}
        elif endpoint == "akshare.fund_open_fund_info_em":
            query_params["indicator"] = "单位净值走势"
        elif endpoint == "sina.hq_sinajs_cn":
            query_params = {"list": f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"}
        elif endpoint == "efinance.stock.get_quote_history":
            query_params.update({"beg": start_date, "end": end_date})
        return self._api_trace(endpoint, query_params)

    def _market_endpoint(self, symbol: str, product_type: str, source_name: str) -> str:  # noqa: ARG002
        if source_name == "akshare_sina":
            return "akshare.stock_zh_a_daily"
        if source_name == "sina_quote":
            return "sina.hq_sinajs_cn"
        if source_name == "efinance":
            return "efinance.stock.get_quote_history"
        if product_type == "etf":
            return "akshare.fund_etf_hist_em"
        if product_type == "fund":
            return "akshare.fund_open_fund_info_em"
        if product_type == "index":
            return "akshare.stock_zh_index_daily"
        return "akshare.stock_zh_a_hist"

    def _provider_name_for_endpoint(self, endpoint: str) -> str:
        if endpoint.startswith("efinance."):
            return "efinance"
        if endpoint.startswith("sina."):
            return "sina"
        return "akshare"

    def _fetch_market_rows(self, symbol: str, product_type: str, start_date: str, end_date: str) -> tuple[list[dict], str, list[str]]:
        provider_warnings: list[str] = []
        if product_type == "etf":
            rows = self._fetch_etf_rows(symbol, start_date, end_date, provider_warnings)
            source_name = "akshare"
        elif product_type == "fund":
            rows = self._fetch_fund_rows(symbol, provider_warnings)
            source_name = "akshare"
        elif product_type == "index":
            rows = self._fetch_index_rows(symbol, start_date, end_date, provider_warnings)
            source_name = "akshare"
        else:
            try:
                rows = self._call_akshare(
                    "akshare.stock_zh_a_hist",
                    "stock_zh_a_hist",
                    provider_warnings,
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
                source_name = "akshare"
            except Exception:
                rows, source_name = self._fetch_stock_rows_fallback(symbol, start_date, end_date, provider_warnings)
        records = _rows_to_records(rows)
        normalized = []
        for row in records:
            normalized.append(
                {
                    "trade_date": self._normalize_date(row.get("日期") or row.get("date") or row.get("trade_date")),
                    "open": row.get("开盘") if row.get("开盘") is not None else row.get("open"),
                    "high": row.get("最高") if row.get("最高") is not None else row.get("high"),
                    "low": row.get("最低") if row.get("最低") is not None else row.get("low"),
                    "close": row.get("收盘") if row.get("收盘") is not None else row.get("close"),
                    "pct_change_1d": row.get("涨跌幅") if row.get("涨跌幅") is not None else row.get("pct_change_1d"),
                    "volume": row.get("成交量") if row.get("成交量") is not None else row.get("volume"),
                    "amount": row.get("成交额") if row.get("成交额") is not None else row.get("amount"),
                }
            )
            if normalized[-1]["trade_date"] is None:
                normalized[-1]["trade_date"] = self._normalize_date(row.get("净值日期") or row.get("nav_date"))
            if normalized[-1]["close"] is None:
                normalized[-1]["close"] = row.get("单位净值") if row.get("单位净值") is not None else row.get("latest_nav")
            if normalized[-1]["pct_change_1d"] is None:
                normalized[-1]["pct_change_1d"] = row.get("日增长率") if row.get("日增长率") is not None else row.get("pct_change")
        normalized.sort(key=lambda item: item.get("trade_date") or "", reverse=True)
        self._fill_missing_pct_change(normalized)
        if not normalized:
            provider_warnings.append(f"market_provider_empty_rows:{source_name}:{symbol}")
        return normalized, source_name, list(dict.fromkeys(provider_warnings))

    def _call_akshare(self, endpoint: str, method_name: str, provider_warnings: list[str], **kwargs):
        fn = getattr(self.ak_module, method_name)
        call_kwargs = dict(kwargs)
        if self._accepts_timeout(fn):
            call_kwargs["timeout"] = self.timeout
        return self._call_with_retry(endpoint, lambda: fn(**call_kwargs), provider_warnings)

    def _call_with_retry(self, endpoint: str, operation, provider_warnings: list[str]):
        attempts = max(1, self.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                result = operation()
                if attempt > 0 and last_error is not None:
                    provider_warnings.append(f"{endpoint}_retry_succeeded:{self._error_summary(last_error)}")
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts - 1:
                    provider_warnings.append(f"{endpoint}_failed:{self._error_summary(exc)}")
                    logger.warning("Live provider endpoint failed: endpoint=%s error=%s", endpoint, exc)
                    raise
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"{endpoint} failed without error detail")

    def _accepts_timeout(self, fn) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return "timeout" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _error_summary(self, exc: Exception) -> str:
        text = str(exc).strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        return f"{type(exc).__name__}:{text}"

    def _fetch_stock_rows_fallback(self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]) -> tuple[object, str]:
        prefixed_symbol = f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"
        try:
            return self._call_akshare(
                "akshare.stock_zh_a_daily",
                "stock_zh_a_daily",
                provider_warnings,
                symbol=prefixed_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ), "akshare_sina"
        except Exception:
            pass

        try:
            return [self._fetch_sina_realtime_row(symbol, provider_warnings)], "sina_quote"
        except Exception:
            pass

        if self.efinance_provider is None:
            raise RuntimeError(f"stock history fetch failed for {symbol}")
        payload = self._call_with_retry(
            "efinance.stock.get_quote_history",
            lambda: self.efinance_provider.fetch_stock_history(f"{symbol}.SH" if symbol.startswith(("5", "6")) else f"{symbol}.SZ"),
            provider_warnings,
        )
        return payload["payload"].get("history", []), payload.get("source_name", "efinance")

    def _fetch_sina_realtime_row(self, symbol: str, provider_warnings: list[str]) -> dict:
        prefixed_symbol = f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"
        response = self._call_with_retry(
            "sina.hq_sinajs_cn",
            lambda: requests.get(
                f"https://hq.sinajs.cn/list={prefixed_symbol}",
                headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            ),
            provider_warnings,
        )
        response.raise_for_status()
        _, payload = response.text.split("=", 1)
        fields = payload.strip().strip('";').split(",")
        if len(fields) < 32:
            raise RuntimeError(f"unexpected sina quote payload for {symbol}")
        previous_close = self._to_float(fields[2])
        close = self._to_float(fields[3])
        pct_change = None
        if previous_close:
            pct_change = round((close - previous_close) / previous_close * 100, 4)
        return {
            "trade_date": self._normalize_date(fields[30]),
            "open": self._to_float(fields[1]),
            "high": self._to_float(fields[4]),
            "low": self._to_float(fields[5]),
            "close": close,
            "pct_change_1d": pct_change,
            "volume": self._to_float(fields[8]),
            "amount": self._to_float(fields[9]),
        }

    def _fetch_etf_rows(self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]):
        try:
            return self._call_akshare(
                "akshare.fund_etf_hist_em",
                "fund_etf_hist_em",
                provider_warnings,
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
        except Exception:
            pass

        try:
            prefixed_symbol = f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"
            rows = self._call_akshare("akshare.fund_etf_hist_sina", "fund_etf_hist_sina", provider_warnings, symbol=prefixed_symbol)
            if rows is not None:
                return rows
        except Exception:
            pass

        if self.efinance_provider is not None:
            payload = self._call_with_retry(
                "efinance.fund.get_quote_history",
                lambda: self.efinance_provider.fetch_history(f"{symbol}.SH" if symbol.startswith(("5", "6")) else f"{symbol}.SZ"),
                provider_warnings,
            )
            return payload["payload"].get("history", [])

        raise RuntimeError(f"ETF history fetch failed for {symbol}")

    def _fetch_fund_rows(self, symbol: str, provider_warnings: list[str]):
        if not hasattr(self.ak_module, "fund_open_fund_info_em"):
            return []
        try:
            return self._call_akshare("akshare.fund_open_fund_info_em", "fund_open_fund_info_em", provider_warnings, symbol=symbol, indicator="单位净值走势")
        except TypeError:
            return self._call_akshare("akshare.fund_open_fund_info_em", "fund_open_fund_info_em", provider_warnings, symbol=symbol)

    def _prefixed_index_symbol(self, symbol: str) -> str:
        """Return Sina-style prefixed symbol for index APIs (sh000001 / sz399001)."""
        plain = symbol.split(".")[0]
        if plain.startswith(("0", "5", "6")):
            return f"sh{plain}"
        return f"sz{plain}"

    def _fetch_index_rows(self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]):
        prefixed = self._prefixed_index_symbol(symbol)
        for method_name, kwargs in (
            ("stock_zh_index_daily", {"symbol": prefixed}),
            ("index_zh_a_hist", {"symbol": symbol, "period": "daily", "start_date": start_date, "end_date": end_date}),
            ("stock_zh_index_spot_em", {}),
        ):
            if not hasattr(self.ak_module, method_name):
                continue
            endpoint = f"akshare.{method_name}"
            try:
                rows = _rows_to_records(self._call_akshare(endpoint, method_name, provider_warnings, **kwargs))
            except TypeError:
                try:
                    rows = _rows_to_records(self._call_akshare(endpoint, method_name, provider_warnings, symbol=symbol))
                except Exception:
                    continue
            except Exception:
                continue
            if rows:
                return self._filter_index_rows(rows, symbol)
        return []

    def _filter_index_rows(self, rows: list[dict], symbol: str) -> list[dict]:
        plain_symbol = symbol.split(".")[0]
        filtered = []
        for row in rows:
            row_code = str(row.get("代码") or row.get("指数代码") or row.get("symbol") or "")
            if row_code and row_code != plain_symbol and row_code.upper() != symbol.upper():
                continue
            filtered.append(row)
        return filtered or rows

    def _fetch_company_info(self, symbol: str) -> dict:
        if not hasattr(self.ak_module, "stock_individual_info_em"):
            return {}
        rows = _rows_to_records(self.ak_module.stock_individual_info_em(symbol=symbol))
        mapping = {}
        for row in rows:
            key = row.get("item") or row.get("项目")
            value = row.get("value") or row.get("值")
            if key:
                mapping[key] = value
        return {
            "canonical_name": mapping.get("股票简称") or mapping.get("证券简称"),
            "industry_name": mapping.get("行业"),
        }

    def _safe_fetch_company_info(self, symbol: str) -> dict:
        try:
            return self._fetch_company_info(symbol)
        except Exception:
            return {}

    def _fetch_industry_snapshot(self, industry_name: str) -> dict | None:
        if not hasattr(self.ak_module, "stock_board_industry_hist_em"):
            return None
        rows = _rows_to_records(
            self.ak_module.stock_board_industry_hist_em(
                symbol=industry_name,
                start_date="20250101",
                end_date="20261231",
                period="日k",
                adjust="",
            )
        )
        if not rows:
            return None
        row = rows[-1]
        trace = self._api_trace(
            "akshare.stock_board_industry_hist_em",
            {
                "symbol": industry_name,
                "start_date": "20250101",
                "end_date": "20261231",
                "period": "日k",
                "adjust": "",
            },
        )
        return {
            "source_name": "akshare",
            **trace,
            "industry_name": industry_name,
            "trade_date": self._normalize_date(row.get("日期")),
            "open": row.get("开盘"),
            "close": row.get("收盘"),
            "pct_change": row.get("涨跌幅"),
            "amount": row.get("成交额"),
        }

    def _safe_fetch_industry_snapshot(self, industry_name: str) -> dict | None:
        try:
            return self._fetch_industry_snapshot(industry_name)
        except Exception:
            return None

    def _identity_industry_snapshot(self, symbol: str, industry_name: str) -> dict:
        trace = self._api_trace("akshare.stock_individual_info_em", {"symbol": symbol})
        return {
            "source_name": "akshare_company_profile",
            "provider": "akshare",
            **trace,
            "industry_name": industry_name,
            "coverage_level": "identity_only",
        }

    def _fetch_fundamental_payload(self, symbol: str) -> dict:
        if not hasattr(self.ak_module, "stock_financial_analysis_indicator"):
            return {}
        rows = _rows_to_records(self.ak_module.stock_financial_analysis_indicator(symbol=symbol, start_year="2020"))
        if not rows:
            return {}
        latest = max(rows, key=lambda row: self._normalize_date(row.get("日期")) or "")
        trace = self._api_trace("akshare.stock_financial_analysis_indicator", {"symbol": symbol, "start_year": "2020"})

        # Try to get pe_ttm/pb from stock_a_indicator_lg
        pe_ttm, pb = None, None
        try:
            val_rows = _rows_to_records(self.ak_module.stock_a_indicator_lg(symbol=symbol))
            if val_rows:
                val_latest = max(val_rows, key=lambda row: self._normalize_date(row.get("trade_date") or row.get("日期")) or "")
                pe_ttm = val_latest.get("pe_ttm")
                pb = val_latest.get("pb")
        except Exception:
            pass

        return {
            "source_name": "akshare",
            **trace,
            "report_date": self._normalize_date(latest.get("日期")),
            "pe_ttm": pe_ttm,
            "pb": pb,
            "roe": latest.get("净资产收益率(%)"),
            "grossprofit_margin": latest.get("主营业务毛利率(%)"),
            "eps": latest.get("每股收益(元)"),
        }

    def _safe_fetch_fundamental_payload(self, symbol: str) -> dict:
        try:
            return self._fetch_fundamental_payload(symbol)
        except Exception:
            return {}

    def _safe_fetch_fund_payloads(self, symbol: str) -> dict:
        try:
            details = self._fetch_fund_detail_mapping(symbol)
        except Exception:
            details = {}
        try:
            nav_payload = self._fetch_fund_nav_payload(symbol)
        except Exception:
            nav_payload = self._empty_fund_nav_payload(symbol)
        return {
            "fund_nav_payload": nav_payload,
            "fund_fee_payload": self._build_fund_fee_payload(symbol, details),
            "fund_redemption_payload": self._build_fund_redemption_payload(symbol, details),
            "fund_profile_payload": self._build_fund_profile_payload(symbol, details),
        }

    def _safe_fetch_index_payloads(self, symbol: str, market_rows: list[dict]) -> dict:
        return {
            "index_daily_payload": self._build_index_daily_payload(symbol, market_rows),
            "index_valuation_payload": self._fetch_index_valuation_payload(symbol),
        }

    def _build_index_daily_payload(self, symbol: str, market_rows: list[dict]) -> dict:
        latest = market_rows[0] if market_rows else {}
        trace = self._api_trace("akshare.stock_zh_index_daily", {"symbol": self._prefixed_index_symbol(symbol)})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "symbol": symbol,
            "trade_date": latest.get("trade_date"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "pct_change_1d": latest.get("pct_change_1d"),
            "volume": latest.get("volume"),
            "amount": latest.get("amount"),
            "history": market_rows[:30],
        }

    def _fetch_index_valuation_payload(self, symbol: str) -> dict:
        rows = []
        for method_name in ("stock_zh_index_value_csindex", "index_value_name_funddb", "index_analysis_daily_sw"):
            if not hasattr(self.ak_module, method_name):
                continue
            try:
                rows = _rows_to_records(getattr(self.ak_module, method_name)(symbol=symbol))
            except TypeError:
                try:
                    rows = _rows_to_records(getattr(self.ak_module, method_name)())
                except Exception:
                    continue
            except Exception:
                continue
            rows = self._filter_index_rows(rows, symbol)
            if rows:
                break
        latest = max(rows, key=lambda row: self._normalize_date(row.get("日期") or row.get("date")) or "") if rows else {}
        trace = self._api_trace("akshare.stock_zh_index_value_csindex", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "symbol": symbol,
            "valuation_date": self._normalize_date(latest.get("日期") or latest.get("date")),
            "pe": self._first_present(latest, "市盈率1", "市盈率", "PE", "pe"),
            "pb": self._first_present(latest, "市净率1", "市净率", "PB", "pb"),
            "dividend_yield": self._first_present(latest, "股息率1", "股息率", "股息率(%)", "dividend_yield"),
            "percentile": self._first_present(latest, "百分位", "估值分位", "pe_percentile", "percentile"),
        }

    def _fetch_fund_nav_payload(self, symbol: str) -> dict:
        rows = []
        if hasattr(self.ak_module, "fund_open_fund_info_em"):
            try:
                rows = _rows_to_records(self.ak_module.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势"))
            except TypeError:
                rows = _rows_to_records(self.ak_module.fund_open_fund_info_em(symbol=symbol))
        normalized = []
        for row in rows:
            nav_date = self._normalize_date(row.get("净值日期") or row.get("日期") or row.get("nav_date"))
            normalized.append(
                {
                    "nav_date": nav_date,
                    "latest_nav": row.get("单位净值") if row.get("单位净值") is not None else row.get("latest_nav"),
                    "accumulated_nav": row.get("累计净值") if row.get("累计净值") is not None else row.get("accumulated_nav"),
                }
            )
        normalized.sort(key=lambda item: item.get("nav_date") or "", reverse=True)
        latest = normalized[0] if normalized else {}
        trace = self._api_trace("akshare.fund_open_fund_info_em", {"symbol": symbol, "indicator": "单位净值走势"})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "latest_nav": latest.get("latest_nav"),
            "accumulated_nav": latest.get("accumulated_nav"),
            "nav_date": latest.get("nav_date"),
            "history": normalized[:30],
        }

    def _empty_fund_nav_payload(self, symbol: str) -> dict:
        trace = self._api_trace("akshare.fund_open_fund_info_em", {"symbol": symbol, "indicator": "单位净值走势"})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "latest_nav": None,
            "accumulated_nav": None,
            "nav_date": None,
            "history": [],
        }

    def _fetch_fund_detail_mapping(self, symbol: str) -> dict:
        mapping = {}
        for method_name in ("fund_individual_detail_info_xq", "fund_etf_fund_info_em", "fund_etf_spot_em"):
            if not hasattr(self.ak_module, method_name):
                continue
            try:
                rows = _rows_to_records(getattr(self.ak_module, method_name)(symbol=symbol))
            except TypeError:
                try:
                    rows = _rows_to_records(getattr(self.ak_module, method_name)())
                except Exception:
                    continue
            except Exception:
                continue
            mapping.update(self._rows_to_mapping(rows, symbol))
        return mapping

    def _rows_to_mapping(self, rows: list[dict], symbol: str) -> dict:
        mapping = {}
        plain_symbol = symbol.split(".")[0]
        for row in rows:
            row_code = str(row.get("基金代码") or row.get("代码") or row.get("symbol") or row.get("基金简称") or "")
            if row_code and row_code.isdigit() and row_code != plain_symbol:
                continue
            key = row.get("item") or row.get("项目") or row.get("key") or row.get("指标")
            value = row.get("value") or row.get("值") or row.get("数值")
            if key:
                mapping[str(key)] = value
                continue
            for item_key, item_value in row.items():
                if item_value is not None and item_value != "":
                    mapping[str(item_key)] = item_value
        return mapping

    def _build_fund_fee_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "management_fee": self._pick_detail(details, "管理费率", "管理费"),
            "custodian_fee": self._pick_detail(details, "托管费率", "托管费"),
            "sales_service_fee": self._pick_detail(details, "销售服务费率", "销售服务费"),
            "purchase_fee": self._pick_detail(details, "申购费率", "买入费率", "认购费率"),
            "redeem_fee": self._pick_detail(details, "赎回费率", "卖出费率"),
        }

    def _build_fund_redemption_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "subscription_status": self._pick_detail(details, "申购状态", "认购状态"),
            "redemption_status": self._pick_detail(details, "赎回状态"),
            "purchase_min": self._pick_detail(details, "最低申购金额", "起购金额", "最小申购单位"),
            "redemption_rule": self._pick_detail(details, "赎回规则", "赎回到账", "赎回确认"),
            "purchase_fee": self._pick_detail(details, "申购费率", "买入费率", "认购费率"),
            "redeem_fee": self._pick_detail(details, "赎回费率", "卖出费率"),
        }

    def _build_fund_profile_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "subscription_status": self._pick_detail(details, "申购状态", "认购状态"),
            "redemption_status": self._pick_detail(details, "赎回状态"),
            "purchase_min": self._pick_detail(details, "最低申购金额", "起购金额", "最小申购单位"),
            "redemption_rule": self._pick_detail(details, "赎回规则", "赎回到账", "赎回确认"),
            "trading_rule": self._pick_detail(details, "交易规则", "交易方式", "运作方式"),
            "tracking_index": self._pick_detail(details, "跟踪标的", "跟踪指数", "标的指数"),
            "fund_manager": self._pick_detail(details, "基金经理", "基金管理人"),
        }

    def _pick_detail(self, details: dict, *keys: str):
        for key in keys:
            value = details.get(key)
            if value is not None and value != "":
                return value
        return None

    def _first_present(self, row: dict, *keys: str):
        for key in keys:
            if key in row and row[key] is not None and row[key] != "":
                return row[key]
        return None

    def _normalize_date(self, value: str | date | datetime | None) -> str | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if "T" in value:
            return value
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except ValueError:
                return value

    def _to_float(self, value: str | int | float | None) -> float | None:
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fill_missing_pct_change(self, rows: list[dict]) -> None:
        for index, row in enumerate(rows[:-1]):
            if row.get("pct_change_1d") is not None:
                continue
            previous_close = row.get("close")
            prior_close = rows[index + 1].get("close")
            if previous_close is None or not prior_close:
                continue
            row["pct_change_1d"] = round((previous_close - prior_close) / prior_close * 100, 4)
