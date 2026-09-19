"""Central configuration. Everything tunable lives here; nothing is hardcoded
at the call sites.

Secrets come from the environment (loaded from a .env at the project root).
They are never written to disk by this project and never appear in run logs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Models that do not fit comfortably in 8 GB of VRAM at q4. Not fatal — Ollama
# will offload the overflow to system RAM — but it is the single most likely
# reason a run feels slow, so we say so once at startup.
_LARGE_MODELS = ("30b", "32b", "27b", "24b", "70b")


class ConfigError(RuntimeError):
    pass


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_host: str = field(default_factory=lambda: _str("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _str("OLLAMA_MODEL", "qwen3:30b-a3b"))
    embed_model: str = field(default_factory=lambda: _str("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    num_parallel: int = field(default_factory=lambda: _int("OLLAMA_NUM_PARALLEL", 2))
    num_ctx: int = field(default_factory=lambda: _int("OLLAMA_NUM_CTX", 8192))
    temperature: float = field(default_factory=lambda: _float("OLLAMA_TEMPERATURE", 0.2))
    think: bool = field(default_factory=lambda: _bool("OLLAMA_THINK", False))

    # Secrets
    tavily_api_key: str = field(default_factory=lambda: _str("TAVILY_API_KEY", ""))
    exa_api_key: str = field(default_factory=lambda: _str("EXA_API_KEY", ""))

    # Guardrails
    call_timeout_s: float = field(default_factory=lambda: _float("CALL_TIMEOUT_S", 30.0))
    max_turns: int = field(default_factory=lambda: _int("MAX_TURNS", 5))
    max_subquestions: int = field(default_factory=lambda: _int("MAX_SUBQUESTIONS", 5))
    min_subquestions: int = field(default_factory=lambda: _int("MIN_SUBQUESTIONS", 3))
    dedup_threshold: float = field(default_factory=lambda: _float("DEDUP_THRESHOLD", 0.9))
    contradiction_sim_threshold: float = field(
        default_factory=lambda: _float("CONTRADICTION_SIM_THRESHOLD", 0.72)
    )
    # Ceiling on extra model calls spent comparing claim pairs. This is the
    # easiest knob to turn when a run is too slow on a small GPU.
    max_contradiction_checks: int = field(
        default_factory=lambda: _int("MAX_CONTRADICTION_CHECKS", 12)
    )

    # Search shaping
    tavily_max_results: int = field(default_factory=lambda: _int("TAVILY_MAX_RESULTS", 5))
    tavily_search_depth: str = field(default_factory=lambda: _str("TAVILY_SEARCH_DEPTH", "basic"))
    extract_top_n: int = field(default_factory=lambda: _int("EXTRACT_TOP_N", 4))
    exa_num_results: int = field(default_factory=lambda: _int("EXA_NUM_RESULTS", 4))

    # Paths
    root: Path = ROOT
    run_log_path: Path = field(
        default_factory=lambda: (ROOT / _str("RUN_LOG_PATH", "logs/runs.jsonl"))
    )
    eval_dir: Path = ROOT / "eval"
    results_dir: Path = ROOT / "eval" / "results"

    @property
    def search_workers(self) -> int:
        """The worker pool is deliberately the same size as Ollama's parallel
        slot count. Ollama has no continuous batching: a third concurrent
        request queues behind the first two and adds latency without adding
        throughput."""
        return max(1, self.num_parallel)

    def require_search_keys(self) -> None:
        missing = [
            name
            for name, value in (("TAVILY_API_KEY", self.tavily_api_key), ("EXA_API_KEY", self.exa_api_key))
            if not value
        ]
        if missing:
            raise ConfigError(
                "Missing "
                + ", ".join(missing)
                + ". Copy config/.env.example to .env at the project root and fill them in."
            )

    def warn_if_oversized(self, stream=sys.stderr) -> None:
        lowered = self.ollama_model.lower()
        if any(tag in lowered for tag in _LARGE_MODELS):
            print(
                f"[config] {self.ollama_model} does not fit in 8 GB of VRAM at q4 "
                f"(~18 GB for qwen3:30b-a3b). Ollama will run the overflow on CPU, so "
                f"expect slow but correct runs. Set OLLAMA_MODEL=qwen3:8b in .env for a "
                f"fully resident alternative.",
                file=stream,
            )
        if self.num_ctx > 8192 and any(tag in lowered for tag in _LARGE_MODELS):
            print(
                f"[config] OLLAMA_NUM_CTX={self.num_ctx} on top of a 30B model will push "
                f"more layers to CPU. 8192 is the tested value.",
                file=stream,
            )


def load_settings() -> Settings:
    return Settings()
