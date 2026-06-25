"""Retrieve visual/OCR evidence from the multimodal store."""

from __future__ import annotations

from dataclasses import dataclass

from verilume.core.multimodal_store import MultimodalStore, VisualItem


@dataclass(frozen=True, slots=True)
class VisualEvidence:
    label: str
    document: str
    page: int | None
    caption: str
    ocr_text: str
    formula_text: str
    image_path: str
    score: float
    item: VisualItem


class MultimodalRetriever:
    def __init__(self, store: MultimodalStore) -> None:
        self.store = store

    def retrieve(self, question: str, *, limit: int = 6) -> list[VisualEvidence]:
        items = self.store.search_visual_items(question, limit=limit)
        evidence: list[VisualEvidence] = []
        for index, item in enumerate(items, start=1):
            evidence.append(
                VisualEvidence(
                    label=f"V{index}",
                    document=item.document,
                    page=item.page,
                    caption=item.caption,
                    ocr_text=item.ocr_text,
                    formula_text=item.formula_text,
                    image_path=item.image_path,
                    score=1.0 / index,
                    item=item,
                )
            )
        return evidence
