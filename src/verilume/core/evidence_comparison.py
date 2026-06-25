"""Compare extracted claims with local, web, and AI evidence streams."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from verilume.core.claim_extraction import Claim, extract_claims
from verilume.core.evidence import EvidencePolicy, FactType
from verilume.core.schemas import LocalSource, WebSource


SUPPORTS = "supports"
CONTRADICTS = "contradicts"
NOT_FOUND = "not_found"
UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    source_type: str
    source_label: str
    support: str
    confidence: float
    snippet: str


@dataclass(frozen=True, slots=True)
class ClaimComparison:
    claim: Claim
    local_support: list[EvidenceSupport]
    web_support: list[EvidenceSupport]
    ai_support: list[EvidenceSupport]
    decision: str
    winning_source_type: str
    conflict_detected: bool


def compare_answer_to_evidence(
    answer: str,
    *,
    local_sources: Sequence[LocalSource],
    web_sources: Sequence[WebSource],
    model_answer: str | None,
    fact_type: str | FactType | None = None,
    policy: str | EvidencePolicy | None = None,
    max_claims: int = 6,
) -> list[ClaimComparison]:
    return [
        compare_claim_to_evidence(
            claim,
            local_sources=list(local_sources),
            web_sources=list(web_sources),
            model_answer=model_answer,
            fact_type=fact_type,
            policy=policy,
        )
        for claim in extract_claims(answer)[:max_claims]
    ]


def compare_claim_to_evidence(
    claim: Claim,
    *,
    local_sources: list[LocalSource],
    web_sources: list[WebSource],
    model_answer: str | None,
    fact_type: str | FactType | None = None,
    policy: str | EvidencePolicy | None = None,
) -> ClaimComparison:
    local_support = [
        _support_from_text("local", source.label, claim, source.text, base_score=source.score)
        for source in local_sources
    ]
    web_support = [
        _support_from_text("web", source.label, claim, source.content, base_score=source.score)
        for source in web_sources
    ]
    ai_support = []
    if model_answer:
        ai_support.append(_support_from_text("ai", "AI", claim, model_answer, base_score=0.6))

    local_support = _best_support(local_support)
    web_support = _best_support(web_support)
    ai_support = _best_support(ai_support)

    decision, winner = _decision(
        local_support=local_support,
        web_support=web_support,
        ai_support=ai_support,
        fact_type=fact_type,
        policy=policy,
    )
    conflict = _has_conflict(local_support, web_support, ai_support)

    return ClaimComparison(
        claim=claim,
        local_support=local_support,
        web_support=web_support,
        ai_support=ai_support,
        decision=decision,
        winning_source_type=winner,
        conflict_detected=conflict,
    )


def claim_comparisons_to_dicts(comparisons: Sequence[ClaimComparison]) -> list[dict[str, Any]]:
    return [asdict(comparison) for comparison in comparisons]


def _support_from_text(
    source_type: str,
    source_label: str,
    claim: Claim,
    text: str,
    *,
    base_score: float | None,
) -> EvidenceSupport:
    snippet, overlap = _best_snippet(claim.text, text)
    entity_score = _entity_score(claim.entities, snippet)
    confidence = max(overlap, entity_score) * 0.82 + min(float(base_score or 0.0), 1.0) * 0.18
    support = _support_label(claim.text, snippet, confidence)
    if support == NOT_FOUND:
        confidence = 0.0
    return EvidenceSupport(
        source_type=source_type,
        source_label=source_label,
        support=support,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        snippet=snippet[:360],
    )


def _best_support(items: Sequence[EvidenceSupport]) -> list[EvidenceSupport]:
    useful = [item for item in items if item.support != NOT_FOUND]
    if not useful:
        return []
    return sorted(useful, key=lambda item: item.confidence, reverse=True)[:3]


def _support_label(claim: str, snippet: str, confidence: float) -> str:
    if not snippet or confidence < 0.28:
        return NOT_FOUND
    if _negation_conflict(claim, snippet) and confidence >= 0.42:
        return CONTRADICTS
    if confidence >= 0.38:
        return SUPPORTS
    return UNCLEAR


def _decision(
    *,
    local_support: Sequence[EvidenceSupport],
    web_support: Sequence[EvidenceSupport],
    ai_support: Sequence[EvidenceSupport],
    fact_type: str | FactType | None,
    policy: str | EvidencePolicy | None,
) -> tuple[str, str]:
    local_ok = _has_support(local_support)
    web_ok = _has_support(web_support)
    ai_ok = _has_support(ai_support)
    fact_value = _enum_value(fact_type)
    policy_value = _enum_value(policy)
    dynamic = fact_value in {FactType.DYNAMIC.value, FactType.NEWS.value}
    local_only = policy_value == EvidencePolicy.LOCAL_ONLY.value

    if dynamic and web_ok:
        return "Web evidence wins for current or changing facts.", "web"
    if local_only and local_ok:
        return "Local evidence wins for this local-document question.", "local"
    if local_ok and web_ok:
        return "Local and web evidence agree.", "hybrid"
    if local_ok:
        return "Local evidence supports the claim.", "local"
    if web_ok:
        return "Web evidence supports the claim.", "web"
    if ai_ok:
        return "Only AI knowledge supports the claim; treat as unverified.", "ai"
    return "Unsupported claim; no evidence stream clearly supports it.", "unsupported"


def _has_conflict(*groups: Sequence[EvidenceSupport]) -> bool:
    supports = any(item.support == SUPPORTS for group in groups for item in group)
    contradicts = any(item.support == CONTRADICTS for group in groups for item in group)
    return supports and contradicts


def _has_support(items: Sequence[EvidenceSupport]) -> bool:
    return any(item.support == SUPPORTS for item in items)


def _best_snippet(claim: str, text: str) -> tuple[str, float]:
    claim_terms = _terms(claim)
    if not claim_terms or not text:
        return "", 0.0
    best_sentence = ""
    best_score = 0.0
    for sentence in _sentences(text):
        sentence_terms = _terms(sentence)
        if not sentence_terms:
            continue
        score = len(claim_terms & sentence_terms) / max(1, len(claim_terms))
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence, best_score


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if sentence.strip()
    ] or [cleaned]


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]*", (text or "").lower())
        if token not in stopwords and len(token) > 1
    }


def _entity_score(entities: Sequence[str], snippet: str) -> float:
    if not entities:
        return 0.0
    normalized = (snippet or "").lower()
    matches = sum(1 for entity in entities if entity.lower() in normalized)
    return matches / max(1, len(entities))


def _negation_conflict(claim: str, snippet: str) -> bool:
    claim_negated = _has_negation(claim)
    snippet_negated = _has_negation(snippet)
    return claim_negated != snippet_negated


def _has_negation(text: str) -> bool:
    return bool(re.search(r"\b(?:not|never|no|without|isn't|aren't|wasn't|weren't|cannot|can't)\b", text.lower()))


def _enum_value(value: str | FactType | EvidencePolicy | None) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))
