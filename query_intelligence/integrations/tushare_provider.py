from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from urllib.parse import urlencode


def _row_get(row: object, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


@dataclass
class TushareMarketProvider:
    client: object

    @classmethod
    def from_token(cls, token: str) -> "TushareMarketProvider":
        import tushare as ts

        return cls(client=ts.pro_api(token))

    def fetch_bundle(self, symbol: str, start_date: str, end_date: str) -> dict:
        errors: list[str] = []
        daily_fields = "ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
        fina_fields = "ts_code,end_date,roe,grossprofit_margin,netprofit_yoy,profit_dedt"
        basic_fields = "ts_code,trade_date,pe_ttm,pb"
        try:
            daily_rows = self.client.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                fields=daily_fields,
            )
        except Exception as exc:  # noqa: BLE001
            daily_rows = None
            errors.append(f"daily:{exc}")
        try:
            fina_rows = self.client.fina_indicator(
                ts_code=symbol,
                fields=fina_fields,
            )
        except Exception as exc:  # noqa: BLE001
            fina_rows = None
            errors.append(f"fina_indicator:{exc}")

        daily_row = self._first_row(daily_rows)
        fina_row = self._first_row(fina_rows)
        basic_trade_date = ""
        if daily_row:
            trade_date = _row_get(daily_row, "trade_date")
            if trade_date:
                basic_trade_date = str(trade_date).replace("-", "")

        basic_row = None
        if basic_trade_date:
            try:
                basic_rows = self.client.daily_basic(
                    ts_code=symbol,
                    trade_date=basic_trade_date,
                    fields=basic_fields,
                )
            except Exception as exc:  # noqa: BLE001
                basic_rows = None
                errors.append(f"daily_basic:{exc}")

            basic_row = self._first_row(basic_rows)
        if daily_row is None and fina_row is None:
            raise RuntimeError("; ".join(errors) or f"no data for {symbol}")
        daily_trace = self._api_trace(
            "tushare.daily",
            {
                "ts_code": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "fields": daily_fields,
            },
        )
        history = self._build_history(daily_rows)
        payload = {
            "symbol": symbol,
            "source_name": "tushare",
            **daily_trace,
            "trade_date": self._normalize_trade_date(_row_get(daily_row, "trade_date")) if daily_row else None,
            "open": _row_get(daily_row, "open") if daily_row else None,
            "high": _row_get(daily_row, "high") if daily_row else None,
            "low": _row_get(daily_row, "low") if daily_row else None,
            "close": _row_get(daily_row, "close") if daily_row else None,
            "volume": _row_get(daily_row, "vol") if daily_row else None,
            "amount": _row_get(daily_row, "amount") if daily_row else None,
            "pct_change_1d": _row_get(daily_row, "pct_chg") if daily_row else None,
            "history": history,
            "roe": _row_get(fina_row, "roe") if fina_row else None,
            "grossprofit_margin": _row_get(fina_row, "grossprofit_margin") if fina_row else None,
            "netprofit_yoy": _row_get(fina_row, "netprofit_yoy") if fina_row else None,
            "profit_dedt": _row_get(fina_row, "profit_dedt") if fina_row else None,
        }
        fundamental_payload = {}
        if fina_row:
            fina_trace = self._api_trace(
                "tushare.fina_indicator",
                {
                    "ts_code": symbol,
                    "fields": fina_fields,
                },
            )
            fundamental_payload = {
                "source_name": "tushare",
                **fina_trace,
                "report_date": self._normalize_trade_date(_row_get(fina_row, "end_date")),
                "pe_ttm": _row_get(basic_row, "pe_ttm") if basic_row else None,
                "pb": _row_get(basic_row, "pb") if basic_row else None,
                "roe": _row_get(fina_row, "roe"),
                "grossprofit_margin": _row_get(fina_row, "grossprofit_margin"),
                "netprofit_yoy": _row_get(fina_row, "netprofit_yoy"),
                "profit_dedt": _row_get(fina_row, "profit_dedt"),
            }
        return {
            "source_type": "market_api",
            "source_name": "tushare",
            "payload": payload,
            "fundamental_payload": fundamental_payload,
            "request_trace": {"symbol": symbol, "start_date": start_date, "end_date": end_date},
            "status": "ok" if not errors else "partial",
            "warnings": errors,
        }

    def _build_history(self, daily_rows: object) -> list[dict]:
        """Convert Tushare daily rows into a normalized history list (latest first)."""
        if daily_rows is None:
            return []
        if hasattr(daily_rows, "to_dict"):
            items = daily_rows.to_dict("records")
        elif isinstance(daily_rows, list):
            items = daily_rows
        else:
            return []
        history = []
        for row in items[:30]:
            history.append({
                "trade_date": self._normalize_trade_date(_row_get(row, "trade_date")),
                "open": _row_get(row, "open"),
                "high": _row_get(row, "high"),
                "low": _row_get(row, "low"),
                "close": _row_get(row, "close"),
                "pct_change_1d": _row_get(row, "pct_chg"),
                "volume": _row_get(row, "vol"),
                "amount": _row_get(row, "amount"),
            })
        return history

    def _api_trace(self, endpoint: str, query_params: dict[str, str]) -> dict:
        encoded_params = urlencode(query_params, doseq=True)
        return {
            "provider": "tushare",
            "provider_endpoint": endpoint,
            "query_params": query_params,
            "source_reference": f"api://{endpoint}?{encoded_params}",
        }

    def _first_row(self, rows: object) -> dict | object | None:
        if rows is None:
            return None
        if isinstance(rows, list):
            return rows[0] if rows else None
        if hasattr(rows, "to_dict"):
            items = rows.to_dict("records")
            return items[0] if items else None
        return None

    def _normalize_trade_date(self, trade_date: str | None) -> str | None:
        if not trade_date:
            return None
        if "-" in trade_date:
            return trade_date
        return datetime.strptime(trade_date, "%Y%m%d").date().isoformat()


@dataclass
class TushareNewsProvider:
    client: object
    default_src: str = "新浪财经"

    @classmethod
    def from_token(cls, token: str, default_src: str = "新浪财经") -> "TushareNewsProvider":
        import tushare as ts

        return cls(client=ts.pro_api(token), default_src=default_src)

    def fetch_news(self, symbol: str, canonical_name: str, limit: int = 10) -> list[dict]:
        end = date.today()
        start = end - timedelta(days=30)
        rows = self.client.major_news(
            src=self.default_src,
            start_date=f"{start.isoformat()} 00:00:00",
            end_date=f"{end.isoformat()} 23:59:59",
            fields="title,content,pub_time,src",
        )
        items = rows if isinstance(rows, list) else rows.to_dict("records")
        normalized_symbol = symbol if "." in symbol else (f"{symbol}.SH" if symbol.startswith(("6", "5")) else f"{symbol}.SZ")
        normalized_name = canonical_name or normalized_symbol
        results = []
        for row in items:
            title = _row_get(row, "title") or normalized_name
            content = _row_get(row, "content") or ""
            if normalized_name not in f"{title}{content}" and symbol.split(".")[0] not in f"{title}{content}":
                continue
            results.append(
                {
                    "evidence_id": f"tsnews_{normalized_symbol}_{len(results)+1}",
                    "doc_id": None,
                    "source_type": "news",
                    "source_name": _row_get(row, "src") or "tushare",
                    "source_url": None,
                    "title": title,
                    "summary": content[:120],
                    "body": content,
                    "publish_time": self._normalize_time(_row_get(row, "pub_time")),
                    "product_type": "stock",
                    "credibility_score": 0.76,
                    "entity_symbols": [normalized_symbol],
                }
            )
            if len(results) >= limit:
                break
        return results

    def _normalize_time(self, value: str | None) -> str | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").isoformat()
            except ValueError:
                return value
