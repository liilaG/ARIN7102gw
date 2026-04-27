from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode


_PAYLOAD_METADATA_FIELDS = {"source_name", "provider", "source_url", "provider_endpoint", "query_params", "source_reference"}


class RetrievalPackager:
    def build(
        self,
        nlu_result: dict,
        ranked_docs: list[dict],
        structured_items: list[dict],
        groups: list[dict],
        total_candidates: int,
        executed_sources: list[str],
        analysis_summary: dict | None = None,
    ) -> dict:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        document_items = []
        for doc in ranked_docs:
            body = doc.get("body") or ""
            summary = doc.get("summary") or ""
            source_name = doc.get("source_name")
            source_url = doc.get("source_url") or self._source_reference(doc)
            document_items.append(
                {
                    "evidence_id": doc["evidence_id"],
                    "source_type": doc["source_type"],
                    "source_name": source_name,
                    "source_url": source_url,
                    "provider": doc.get("provider") or source_name,
                    "title": doc.get("title"),
                    "summary": summary or None,
                    "text_excerpt": doc.get("text_excerpt") or self._text_excerpt(summary, body),
                    "body": body or None,
                    "body_available": bool(body),
                    "publish_time": doc.get("publish_time"),
                    "retrieved_at": retrieved_at,
                    "entity_hits": doc.get("entity_symbols"),
                    "retrieval_score": doc.get("retrieval_score"),
                    "rank_score": doc.get("rank_score"),
                    "reason": doc.get("reason", []),
                    "payload": doc.get("payload"),
                }
            )
        structured_data = []
        for item in structured_items:
            payload = item["payload"]
            source_name = item.get("source_name") or payload.get("source_name")
            source_url = item.get("source_url") or payload.get("source_url")
            provider_endpoint = item.get("provider_endpoint") or payload.get("provider_endpoint")
            query_params = item.get("query_params") or payload.get("query_params") or {}
            source_reference = item.get("source_reference") or payload.get("source_reference") or self._structured_source_reference(
                source_name=source_name,
                provider_endpoint=provider_endpoint,
                query_params=query_params,
            )
            structured_data.append(
                {
                    "evidence_id": item["evidence_id"],
                    "source_type": item["source_type"],
                    "source_name": source_name,
                    "source_url": source_url,
                    "provider": item.get("provider") or payload.get("provider") or source_name,
                    "provider_endpoint": provider_endpoint,
                    "query_params": query_params,
                    "source_reference": source_reference,
                    "as_of": item.get("as_of") or self._as_of(payload),
                    "period": item.get("period") or self._period(payload),
                    "field_coverage": self._field_coverage(payload),
                    "quality_flags": self._quality_flags(source_name, source_url, payload),
                    "retrieved_at": retrieved_at,
                    "payload": payload,
                }
            )

        structured_source_types = {item["source_type"] for item in structured_data}
        document_source_types = {item["source_type"] for item in document_items}
        coverage_detail = self._coverage_detail(structured_source_types)
        coverage = {
            "price": bool(structured_source_types.intersection({"market_api", "index_daily", "fund_nav"})),
            "news": "news" in document_source_types,
            "industry": "industry_sql" in structured_source_types,
            "fundamentals": "fundamental_sql" in structured_source_types,
            "announcement": "announcement" in document_source_types,
            "product_mechanism": bool(
                document_source_types.intersection({"faq", "product_doc"})
                or structured_source_types.intersection({"fund_fee", "fund_redemption", "fund_profile"})
            ),
            "macro": bool(structured_source_types.intersection({"macro_sql", "macro_indicator", "policy_event"})),
            "risk": bool(
                structured_source_types.intersection({"market_api", "fundamental_sql", "index_valuation", "fund_fee", "fund_redemption", "macro_indicator"})
                or document_source_types.intersection({"research_note", "announcement", "faq", "product_doc"})
            ),
            "comparison": self._has_comparison_coverage(nlu_result, structured_data, document_items),
        }
        warnings = []
        if "out_of_scope_query" in nlu_result.get("risk_flags", []):
            warnings.append("out_of_scope_query")
        if "clarification_required" in nlu_result.get("risk_flags", []):
            warnings.append("clarification_required_missing_entity")
        if "announcement" in nlu_result.get("source_plan", []) and not coverage["announcement"]:
            warnings.append("announcement_not_found_recent_window")
        for item in structured_data:
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            for warning in payload.get("provider_warnings") or []:
                if warning not in warnings:
                    warnings.append(str(warning))

        ranked_ids = [item["evidence_id"] for item in document_items[:5]]
        confidence = round(
            (
                sum(item.get("rank_score", 0.7) for item in ranked_docs[:3]) / max(len(ranked_docs[:3]), 1)
                + sum(1.0 for value in coverage.values() if value) / len(coverage)
            )
            / 2,
            2,
        )

        return {
            "query_id": nlu_result["query_id"],
            "nlu_snapshot": {
                "raw_query": nlu_result.get("raw_query"),
                "normalized_query": nlu_result.get("normalized_query"),
                "product_type": nlu_result["product_type"]["label"],
                "intent_labels": [item["label"] for item in nlu_result["intent_labels"]],
                "entities": [item["symbol"] for item in nlu_result["entities"] if item.get("symbol")],
                "source_plan": nlu_result["source_plan"],
            },
            "executed_sources": executed_sources,
            "documents": document_items,
            "structured_data": structured_data,
            "evidence_groups": groups,
            "coverage": coverage,
            "coverage_detail": coverage_detail,
            "warnings": warnings,
            "retrieval_confidence": confidence,
            "debug_trace": {
                "candidate_count": total_candidates,
                "after_dedup": len(ranked_docs),
                "top_ranked": ranked_ids,
            },
            "analysis_summary": analysis_summary or {},
        }

    def _coverage_detail(self, structured_source_types: set[str]) -> dict[str, bool]:
        return {
            "price_history": "market_api" in structured_source_types,
            "financials": "fundamental_sql" in structured_source_types,
            "valuation": bool(structured_source_types.intersection({"index_valuation", "fundamental_sql"})),
            "industry_snapshot": "industry_sql" in structured_source_types,
            "fund_nav": "fund_nav" in structured_source_types,
            "fund_fee": "fund_fee" in structured_source_types,
            "fund_redemption": "fund_redemption" in structured_source_types,
            "fund_profile": "fund_profile" in structured_source_types,
            "index_daily": "index_daily" in structured_source_types,
            "index_valuation": "index_valuation" in structured_source_types,
            "macro_indicator": bool(structured_source_types.intersection({"macro_indicator", "macro_sql"})),
            "policy_event": "policy_event" in structured_source_types,
        }

    def _text_excerpt(self, summary: str, body: str, max_chars: int = 500) -> str | None:
        text = (summary or body).strip()
        if not text:
            return None
        return text[:max_chars]

    def _source_reference(self, doc: dict) -> str | None:
        source_name = doc.get("source_name") or doc.get("provider")
        if not source_name:
            return None
        reference_id = doc.get("doc_id") or doc.get("source_row_id") or doc.get("evidence_id")
        if not reference_id:
            return None
        safe_source = str(source_name).strip().replace(" ", "_")
        safe_ref = str(reference_id).strip().replace(" ", "_")
        return f"dataset://{safe_source}/{safe_ref}"

    def _structured_source_reference(
        self,
        source_name: str | None,
        provider_endpoint: str | None,
        query_params: dict,
    ) -> str | None:
        if provider_endpoint:
            encoded_params = urlencode(query_params, doseq=True)
            return f"api://{provider_endpoint}?{encoded_params}" if encoded_params else f"api://{provider_endpoint}"
        if source_name:
            return f"provider://{str(source_name).strip().replace(' ', '_')}"
        return None

    def _as_of(self, payload: dict) -> str | None:
        value = payload.get("as_of") or payload.get("trade_date") or payload.get("report_date") or payload.get("metric_date")
        return str(value) if value is not None else None

    def _period(self, payload: dict) -> str | None:
        value = payload.get("period") or payload.get("report_date") or payload.get("trade_date") or payload.get("metric_date")
        return str(value) if value is not None else None

    def _field_coverage(self, payload: dict) -> dict:
        business_payload = self._business_payload(payload)
        total = len(business_payload)
        missing_fields = sorted(key for key, value in business_payload.items() if value is None)
        return {
            "total_fields": total,
            "non_null_fields": total - len(missing_fields),
            "missing_fields": missing_fields,
        }

    def _quality_flags(self, source_name: str | None, source_url: str | None, payload: dict) -> list[str]:
        flags = []
        normalized_source = (source_name or "").lower()
        if "seed" in normalized_source:
            flags.append("seed_source")
        if not source_url:
            flags.append("missing_source_url")
        business_payload = self._business_payload(payload)
        if not business_payload:
            flags.append("empty_payload")
        elif any(value is None for value in business_payload.values()):
            flags.append("missing_values")
        return flags

    def _business_payload(self, payload: dict) -> dict:
        return {key: value for key, value in payload.items() if key not in _PAYLOAD_METADATA_FIELDS}

    def _has_comparison_coverage(self, nlu_result: dict, structured_data: list[dict], document_items: list[dict]) -> bool:
        intent_labels = {item["label"] for item in nlu_result.get("intent_labels", [])}
        topic_labels = {item["label"] for item in nlu_result.get("topic_labels", [])}
        compare_like = "peer_compare" in intent_labels or "comparison" in topic_labels
        if not compare_like:
            return False

        covered_symbols = {
            item["payload"].get("symbol")
            for item in structured_data
            if item.get("payload", {}).get("symbol")
        }
        for doc in document_items:
            covered_symbols.update(doc.get("entity_hits") or [])
        return len(covered_symbols) >= 2
