from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    tushare_token: str | None = None
    postgres_dsn: str | None = None
    cninfo_announcement_url: str = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    cninfo_static_base: str = "https://static.cninfo.com.cn/"
    request_timeout_seconds: int = 15
    use_live_market: bool = True
    use_live_macro: bool = True
    use_live_news: bool = True
    use_live_announcement: bool = True
    use_postgres_retrieval: bool = False
    training_dataset_path: str | None = None
    models_dir: str = "models"
    training_manifest_path: str | None = None
    enable_external_data: bool = False
    dataset_allowlist: tuple[str, ...] = ()
    enable_translation: bool = False
    force_refresh_data: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        live_market_raw = os.getenv("QI_USE_LIVE_MARKET", "true")
        return cls(
            tushare_token=os.getenv("TUSHARE_TOKEN"),
            postgres_dsn=os.getenv("QI_POSTGRES_DSN"),
            cninfo_announcement_url=os.getenv("CNINFO_ANNOUNCEMENT_URL", "https://www.cninfo.com.cn/new/hisAnnouncement/query"),
            cninfo_static_base=os.getenv("CNINFO_STATIC_BASE", "https://static.cninfo.com.cn/"),
            request_timeout_seconds=int(os.getenv("QI_HTTP_TIMEOUT_SECONDS", "15")),
            use_live_market=live_market_raw.lower() in {"1", "true", "yes"},
            use_live_macro=os.getenv("QI_USE_LIVE_MACRO", "true").lower() in {"1", "true", "yes"},
            use_live_news=os.getenv("QI_USE_LIVE_NEWS", live_market_raw).lower() in {"1", "true", "yes"},
            use_live_announcement=os.getenv("QI_USE_LIVE_ANNOUNCEMENT", live_market_raw).lower() in {"1", "true", "yes"},
            use_postgres_retrieval=os.getenv("QI_USE_POSTGRES_RETRIEVAL", "").lower() in {"1", "true", "yes"},
            training_dataset_path=os.getenv("QI_TRAINING_DATASET"),
            models_dir=os.getenv("QI_MODELS_DIR", "models"),
            training_manifest_path=os.getenv("QI_TRAINING_MANIFEST"),
            enable_external_data=os.getenv("QI_ENABLE_EXTERNAL_DATA", "").lower() in {"1", "true", "yes"},
            dataset_allowlist=tuple(
                s.strip() for s in os.getenv("QI_DATASET_ALLOWLIST", "").split(",") if s.strip()
            ),
            enable_translation=os.getenv("QI_ENABLE_TRANSLATION", "").lower() in {"1", "true", "yes"},
            force_refresh_data=os.getenv("QI_FORCE_REFRESH_DATA", "").lower() in {"1", "true", "yes"},
        )
