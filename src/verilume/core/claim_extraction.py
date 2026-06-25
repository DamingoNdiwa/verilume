"""Rule-based factual claim extraction for evidence comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    text: str
    claim_type: str
    entities: list[str]


_VAGUE_MARKERS = (
    "i hope this helps",
    "let me know",
    "based on the context",
    "based on the provided",
    "the answer is",
    "confidence:",
    "sources:",
)

_FACT_VERBS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " had ",
    " became ",
    " becomes ",
    " includes ",
    " contains ",
    " covers ",
    " describes ",
    " explains ",
    " models ",
    " refers ",
    " means ",
    " uses ",
    " works ",
    " consists ",
    " provides ",
)


def extract_claims(answer: str) -> list[Claim]:
    claims: list[Claim] = []
    for sentence in _split_sentences(answer):
        if not _looks_factual(sentence):
            continue
        claim_id = f"C{len(claims) + 1}"
        claims.append(
            Claim(
                id=claim_id,
                text=sentence,
                claim_type=_claim_type(sentence),
                entities=_entities(sentence),
            )
        )
    return claims


def _split_sentences(answer: str) -> list[str]:
    text = _clean_answer(answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _clean_answer(answer: str) -> str:
    text = re.sub(r"\[[SW]\d+\]", "", answer or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_factual(sentence: str) -> bool:
    normalized = f" {sentence.strip().lower()} "
    if len(sentence.split()) < 4:
        return False
    if any(marker in normalized for marker in _VAGUE_MARKERS):
        return False
    if any(verb in normalized for verb in _FACT_VERBS):
        return True
    if re.search(r"\b\d{3,4}\b", sentence):
        return True
    return bool(_entities(sentence) and len(sentence.split()) >= 6)


def _claim_type(sentence: str) -> str:
    normalized = f" {sentence.lower()} "
    if re.search(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|\b\d{4}\b", sentence):
        return "date"
    if " refers to " in normalized or " means " in normalized:
        return "definition"
    if " is " in normalized or " are " in normalized:
        return "identity"
    if " contains " in normalized or " includes " in normalized or " covers " in normalized:
        return "document_summary"
    return "factual"


def _entities(sentence: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\b",
        sentence,
    ):
        value = match.group(0).strip()
        if value.lower() in {"i", "the", "this", "it"}:
            continue
        if value not in entities:
            entities.append(value)
    return entities
