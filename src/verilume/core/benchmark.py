"""Benchmark reports for comparing Verilume evidence strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from verilume.core.schemas import RAGResponse

FULL = "full"
LOCAL_ONLY = "local_only"
AI_ONLY = "ai_only"
WEB_ONLY = "web_only"

BENCHMARK_MODES = (FULL, LOCAL_ONLY, AI_ONLY, WEB_ONLY)

MODE_LABELS = {
    FULL: "Full",
    LOCAL_ONLY: "Local Only",
    AI_ONLY: "AI Only",
    WEB_ONLY: "Web Only",
}


@dataclass(slots=True)
class BenchmarkResult:
    mode: str
    answer: str
    confidence: str
    source_count: int
    local_source_count: int
    web_source_count: int
    latency_seconds: float
    faithfulness_score: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkReport:
    question: str
    results: list[BenchmarkResult]
    best_mode: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "best_mode": self.best_mode,
            "best_mode_label": MODE_LABELS.get(self.best_mode, self.best_mode),
            "notes": list(self.notes),
            "results": [result.to_dict() for result in self.results],
        }

    def to_rag_response(self) -> RAGResponse:
        answer = self._answer_markdown()
        return RAGResponse(
            answer=answer,
            local_sources=[],
            web_sources=[],
            used_web=any(result.web_source_count for result in self.results),
            confidence=_report_confidence(self.results),
            diagnostics={
                "benchmark_mode": True,
                "benchmark_report": self.to_dict(),
                "evidence_winner": self.best_mode,
                "evidence_streams": [result.mode for result in self.results],
            },
        )

    def _answer_markdown(self) -> str:
        if not self.results:
            return "Benchmark mode could not produce any results."

        best = _result_by_mode(self.results, self.best_mode) or best_benchmark_result(self.results)
        rows = [
            "| Mode | Confidence | Sources | Time | Faithfulness |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for result in self.results:
            faithfulness = (
                f"{int(round(result.faithfulness_score * 100))}%"
                if result.faithfulness_score is not None
                else "N/A"
            )
            rows.append(
                "| "
                f"{MODE_LABELS.get(result.mode, result.mode)} | "
                f"{result.confidence} | "
                f"{result.source_count} | "
                f"{result.latency_seconds:.2f}s | "
                f"{faithfulness} |"
            )

        notes = "\n".join(f"- {note}" for note in self.notes)
        notes_block = f"\n\n**Notes**\n{notes}" if notes else ""
        return (
            f"**Best mode:** {MODE_LABELS.get(best.mode, best.mode)}\n\n"
            f"{best.answer.strip()}\n\n"
            "**Benchmark Results**\n"
            + "\n".join(rows)
            + notes_block
        ).strip()


def make_benchmark_result(mode: str, response: RAGResponse, latency_seconds: float) -> BenchmarkResult:
    diagnostics = dict(response.diagnostics or {})
    faithfulness = diagnostics.get("answer_verification_score")
    try:
        faithfulness_score = float(faithfulness) if faithfulness is not None else None
    except (TypeError, ValueError):
        faithfulness_score = None

    local_count = len(response.local_sources or [])
    web_count = len(response.web_sources or [])
    return BenchmarkResult(
        mode=mode,
        answer=response.answer,
        confidence=response.confidence,
        source_count=local_count + web_count,
        local_source_count=local_count,
        web_source_count=web_count,
        latency_seconds=max(0.0, float(latency_seconds)),
        faithfulness_score=faithfulness_score,
        diagnostics=diagnostics,
    )


def best_benchmark_result(results: list[BenchmarkResult]) -> BenchmarkResult:
    if not results:
        raise ValueError("Cannot choose a benchmark winner without results.")
    return max(results, key=_benchmark_score)


def choose_best_mode(results: list[BenchmarkResult]) -> str:
    return best_benchmark_result(results).mode if results else FULL


def benchmark_notes(results: list[BenchmarkResult], best_mode: str) -> list[str]:
    notes: list[str] = []
    best = _result_by_mode(results, best_mode)
    if best is not None:
        notes.append(
            f"{MODE_LABELS.get(best.mode, best.mode)} had the strongest combined confidence, "
            f"source count, and verification score."
        )
    if any(result.mode == AI_ONLY and result.source_count == 0 for result in results):
        notes.append("AI-only answers are useful for comparison but are not independently cited.")
    if any(result.mode == WEB_ONLY and result.web_source_count == 0 for result in results):
        notes.append("Web-only mode found no configured web evidence for this question.")
    if any(result.mode == LOCAL_ONLY and result.local_source_count == 0 for result in results):
        notes.append("Local-only mode found no indexed local evidence for this question.")
    return notes


def _benchmark_score(result: BenchmarkResult) -> tuple[float, int, float, float]:
    confidence_rank = {
        "current-information": 4.0,
        "local-grounded": 4.0,
        "local-web-assisted": 4.0,
        "web-assisted": 3.5,
        "high": 3.0,
        "medium": 2.0,
        "model-only": 1.5,
        "low": 1.0,
    }.get(str(result.confidence).lower(), 1.5)
    faithfulness = result.faithfulness_score if result.faithfulness_score is not None else 0.0
    return (
        confidence_rank,
        result.source_count,
        faithfulness,
        -result.latency_seconds,
    )


def _report_confidence(results: list[BenchmarkResult]) -> str:
    if not results:
        return "low"
    best = best_benchmark_result(results)
    if str(best.confidence).lower() in {"low", "model-only"}:
        return best.confidence
    return "benchmark"


def _result_by_mode(results: list[BenchmarkResult], mode: str) -> BenchmarkResult | None:
    for result in results:
        if result.mode == mode:
            return result
    return None
