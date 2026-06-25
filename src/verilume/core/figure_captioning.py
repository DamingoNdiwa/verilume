"""Fallback figure captioning utilities."""

from __future__ import annotations


class FigureCaptioner:
    """Generate conservative captions from available OCR/formula text."""

    def caption(
        self,
        *,
        ocr_text: str = "",
        formula_text: str = "",
        existing_caption: str = "",
    ) -> str:
        if existing_caption.strip():
            return existing_caption.strip()
        if ocr_text.strip():
            return _compact(ocr_text)
        if formula_text.strip():
            return f"Formula: {_compact(formula_text)}"
        return "Visual content stored without readable OCR text."


def _compact(text: str, limit: int = 240) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}..."
