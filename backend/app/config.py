"""
Central configuration. Replaces the old google.colab.userdata calls
(which only work inside a Colab notebook) with plain environment
variables loaded from a .env file, so this runs as a normal script
or service anywhere.
"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Fetch an env var, but don't crash at import time if it's missing —
    fail when the caller actually tries to use the feature instead."""
    return os.getenv(name, "")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = _require("OPENAI_API_KEY")
    google_cse_api_key: str = _require("GOOGLE_CSE_API_KEY")
    google_cse_cx: str = _require("GOOGLE_CSE_CX")
    user_agent: str = os.getenv("USER_AGENT", "AutoWiki/1.0 (contact: you@example.com)")

    # models
    fast_model: str = os.getenv("AUTOWIKI_FAST_MODEL", "gpt-4o-mini")
    strong_model: str = os.getenv("AUTOWIKI_STRONG_MODEL", "gpt-4.1")

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return self.openai_api_key

    def require_google_cse(self) -> tuple[str, str]:
        if not self.google_cse_api_key or not self.google_cse_cx:
            raise RuntimeError(
                "GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX must both be set in your .env file."
            )
        return self.google_cse_api_key, self.google_cse_cx


settings = Settings()
