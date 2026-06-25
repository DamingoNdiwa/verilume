from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.figure_captioning import FigureCaptioner
from verilume.core.multimodal_retrieval import MultimodalRetriever
from verilume.core.multimodal_store import MultimodalStore


class MultimodalRetrievalTests(unittest.TestCase):
    def test_captioner_uses_ocr_text_as_safe_fallback(self) -> None:
        caption = FigureCaptioner().caption(ocr_text="Luxembourg passport number appears here.")

        self.assertIn("Luxembourg passport number", caption)

    def test_store_retrieves_visual_item_for_page(self) -> None:
        with TemporaryDirectory() as tmp:
            store = MultimodalStore(Path(tmp) / "visual.sqlite")
            store.add_visual_item(
                document="scan.pdf",
                page=4,
                caption="Diagram showing model architecture",
                ocr_text="encoder decoder architecture",
            )

            items = store.get_visual_items_for_page("scan.pdf", 4)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].caption, "Diagram showing model architecture")

    def test_retriever_finds_plot_by_caption_and_ocr(self) -> None:
        with TemporaryDirectory() as tmp:
            store = MultimodalStore(Path(tmp) / "visual.sqlite")
            store.add_visual_item(
                document="figures.pdf",
                page=8,
                caption="Plot about regression analysis",
                ocr_text="residual regression trend",
            )
            retriever = MultimodalRetriever(store)

            evidence = retriever.retrieve("Find the plot about regression analysis")

            self.assertTrue(evidence)
            self.assertEqual(evidence[0].document, "figures.pdf")
            self.assertEqual(evidence[0].page, 8)

    def test_retriever_finds_formula_text(self) -> None:
        with TemporaryDirectory() as tmp:
            store = MultimodalStore(Path(tmp) / "visual.sqlite")
            store.add_visual_item(
                document="math.pdf",
                page=8,
                formula_text="E = mc^2",
                caption="Formula region",
            )

            evidence = MultimodalRetriever(store).retrieve("What formula appears on page 8?")

            self.assertTrue(evidence)
            self.assertEqual(evidence[0].formula_text, "E = mc^2")


if __name__ == "__main__":
    unittest.main()
