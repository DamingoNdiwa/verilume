"""First-run setup checklist.

The step computation is pure (no Streamlit) so it can be unit tested; the
renderer is a thin Streamlit wrapper. The checklist only appears until every
step is satisfied, then stays out of the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from verilume.settings import AppSettings

# Web providers that need no API key are "ready" as soon as they are selected.
_KEYLESS_WEB_PROVIDERS = {"duckduckgo", "none", "disabled", ""}
# Known per-provider key attributes, checked defensively via getattr.
_WEB_KEY_ATTRS = (
    "tavily_api_key",
    "brave_api_key",
    "exa_api_key",
    "serpapi_api_key",
    "bing_api_key",
    "google_cse_api_key",
    "custom_search_api_key",
    "custom_search_endpoint",
)


@dataclass(frozen=True)
class SetupStep:
    key: str
    title: str
    detail: str
    done: bool
    focus_section: str | None = None


def _model_step(settings: AppSettings) -> SetupStep:
    backend = (getattr(settings, "generation_backend", "") or "").lower()
    if backend == "ollama":
        ready = bool((settings.ollama_model or "").strip())
        detail = f"Using {settings.ollama_model}" if ready else "Select an Ollama model"
    else:
        ready = bool((settings.hf_token or "").strip() and (settings.hf_llm_model or "").strip())
        detail = (
            f"Using {settings.hf_llm_model}"
            if ready
            else "Add a Hugging Face token and choose a model"
        )
    return SetupStep("model", "Set up the AI model", detail, ready, "models")


def _web_step(settings: AppSettings) -> SetupStep:
    provider = (getattr(settings, "web_search_provider", "") or "").lower()
    ready = provider in _KEYLESS_WEB_PROVIDERS or any(
        str(getattr(settings, attr, "") or "").strip() for attr in _WEB_KEY_ATTRS
    )
    detail = "Keyless or key configured" if ready else "Add a web provider key, or use DuckDuckGo"
    return SetupStep("web", "Choose a web search option", detail, ready, "search")


def compute_setup_steps(
    settings: AppSettings,
    *,
    uploaded_documents: int,
    chunks_indexed: int,
) -> list[SetupStep]:
    return [
        _model_step(settings),
        _web_step(settings),
        SetupStep(
            "upload",
            "Upload documents",
            f"{uploaded_documents} uploaded" if uploaded_documents else "No documents yet",
            uploaded_documents > 0,
            "documents",
        ),
        SetupStep(
            "build",
            "Build the knowledge base",
            f"{chunks_indexed} chunks indexed" if chunks_indexed else "Not built yet",
            chunks_indexed > 0,
            "documents",
        ),
    ]


def setup_is_complete(steps: list[SetupStep]) -> bool:
    return all(step.done for step in steps)


def render_onboarding(settings: AppSettings, stats: dict[str, int]) -> None:
    """Render the first-run checklist; render nothing once every step is done."""
    steps = compute_setup_steps(
        settings,
        uploaded_documents=int(stats.get("uploaded_documents", 0) or 0),
        chunks_indexed=int(stats.get("chunks_indexed", 0) or 0),
    )
    if setup_is_complete(steps):
        return

    done_count = sum(1 for step in steps if step.done)
    rows = "".join(
        (
            '<div class="veri-onboard-step">'
            f'<span class="veri-onboard-mark {"done" if step.done else "todo"}">'
            f'{"✓" if step.done else "○"}</span>'
            f'<span class="veri-onboard-step-title">{step.title}</span>'
            f'<span class="veri-onboard-step-detail">{step.detail}</span>'
            "</div>"
        )
        for step in steps
    )
    st.markdown(
        f"""
<div class="veri-onboard">
  <div class="veri-onboard-head">
    <span class="veri-onboard-kicker">Getting started</span>
    <span class="veri-onboard-progress">{done_count}/{len(steps)} complete</span>
  </div>
  <div class="veri-onboard-steps">{rows}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
