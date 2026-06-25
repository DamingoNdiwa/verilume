from __future__ import annotations

import unittest

from verilume.core.claim_extraction import extract_claims
from verilume.core.evidence import EvidencePolicy, FactType
from verilume.core.evidence_comparison import (
    SUPPORTS,
    compare_answer_to_evidence,
    compare_claim_to_evidence,
)
from verilume.core.schemas import LocalSource, WebSource


class EvidenceComparisonTests(unittest.TestCase):
    def test_extract_claims_keeps_atomic_factual_sentences(self) -> None:
        claims = extract_claims(
            "Brussels is the capital of Belgium. "
            "It is also the administrative centre of the European Union. "
            "I hope this helps."
        )

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].text, "Brussels is the capital of Belgium.")
        self.assertIn("Brussels", claims[0].entities)

    def test_compare_claim_prefers_local_for_local_policy(self) -> None:
        claim = extract_claims("Regression analysis models relationships between variables.")[0]
        comparison = compare_claim_to_evidence(
            claim,
            local_sources=[
                LocalSource(
                    label="S1",
                    document="regression.pdf",
                    page=1,
                    chunk_id="c1",
                    text="Regression analysis models relationships between variables in data.",
                    score=0.95,
                )
            ],
            web_sources=[],
            model_answer="Regression analysis is a broad statistical method.",
            policy=EvidencePolicy.LOCAL_ONLY,
        )

        self.assertEqual(comparison.local_support[0].support, SUPPORTS)
        self.assertEqual(comparison.winning_source_type, "local")

    def test_compare_claim_prefers_web_for_dynamic_fact(self) -> None:
        comparison = compare_answer_to_evidence(
            "Jonas Gahr Store is the prime minister of Norway.",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Government page",
                    url="https://example.gov/no",
                    content="Jonas Gahr Store is the prime minister of Norway.",
                    score=0.9,
                )
            ],
            model_answer="Norway has a prime minister.",
            fact_type=FactType.DYNAMIC,
        )[0]

        self.assertEqual(comparison.web_support[0].support, SUPPORTS)
        self.assertEqual(comparison.winning_source_type, "web")

    def test_ai_only_support_is_marked_unverified(self) -> None:
        comparison = compare_answer_to_evidence(
            "Spectral analysis studies signals in the frequency domain.",
            local_sources=[],
            web_sources=[],
            model_answer="Spectral analysis studies signals in the frequency domain.",
        )[0]

        self.assertEqual(comparison.ai_support[0].support, SUPPORTS)
        self.assertEqual(comparison.winning_source_type, "ai")
        self.assertIn("unverified", comparison.decision)


if __name__ == "__main__":
    unittest.main()
