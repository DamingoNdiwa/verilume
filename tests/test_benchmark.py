from __future__ import annotations

from verilume.core.benchmark import (
    AI_ONLY,
    FULL,
    LOCAL_ONLY,
    WEB_ONLY,
    BenchmarkReport,
    choose_best_mode,
    make_benchmark_result,
)
from verilume.core.schemas import LocalSource, RAGResponse, WebSource
from verilume.rag import _benchmark_mode_settings, _search_mode_allows_local, _search_mode_key
from verilume.settings import AppSettings


def test_benchmark_result_counts_sources_and_latency() -> None:
    response = RAGResponse(
        answer="Local and web answer [S1] [W1]",
        local_sources=[
            LocalSource("S1", "doc.pdf", 1, "chunk-1", "Local text", 0.92),
        ],
        web_sources=[
            WebSource("W1", "Web", "https://example.com", "Web text", 0.88),
        ],
        used_web=True,
        confidence="high",
        diagnostics={"answer_verification_score": 0.84},
    )

    result = make_benchmark_result(FULL, response, 1.25)

    assert result.source_count == 2
    assert result.local_source_count == 1
    assert result.web_source_count == 1
    assert result.latency_seconds == 1.25
    assert result.faithfulness_score == 0.84


def test_benchmark_report_converts_to_rag_response() -> None:
    local = make_benchmark_result(
        LOCAL_ONLY,
        RAGResponse("Local answer [S1]", [LocalSource("S1", "doc.pdf", 1, "c1", "x", 1.0)], [], False, "local-grounded"),
        0.2,
    )
    ai = make_benchmark_result(
        AI_ONLY,
        RAGResponse("AI answer", [], [], False, "model-only"),
        0.1,
    )
    report = BenchmarkReport("Question?", [local, ai], choose_best_mode([local, ai]))

    response = report.to_rag_response()

    assert "Benchmark Results" in response.answer
    assert response.diagnostics["benchmark_mode"] is True
    assert response.diagnostics["benchmark_report"]["best_mode"] == LOCAL_ONLY


def test_benchmark_mode_settings_isolate_strategies() -> None:
    settings = AppSettings(benchmark_mode=True, semantic_cache_enabled=True, enable_web_search=True)
    modes = dict(_benchmark_mode_settings(settings))

    assert set(modes) == {FULL, LOCAL_ONLY, AI_ONLY, WEB_ONLY}
    assert all(not mode_settings.benchmark_mode for mode_settings in modes.values())
    assert all(not mode_settings.semantic_cache_enabled for mode_settings in modes.values())
    assert _search_mode_key(modes[AI_ONLY]) == "ai_only"
    assert _search_mode_allows_local("ai_only") is False
    assert modes[LOCAL_ONLY].enable_web_search is False
