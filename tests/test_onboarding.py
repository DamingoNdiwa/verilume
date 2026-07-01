from __future__ import annotations

import unittest

from verilume.settings import AppSettings
from verilume.ui.onboarding import compute_setup_steps, setup_is_complete


def _steps(settings: AppSettings, *, docs: int = 0, chunks: int = 0):
    return compute_setup_steps(settings, uploaded_documents=docs, chunks_indexed=chunks)


class OnboardingStepTests(unittest.TestCase):
    def test_fresh_install_has_no_steps_done(self) -> None:
        settings = AppSettings(generation_backend="huggingface", hf_token="", hf_llm_model="")
        steps = _steps(settings)
        self.assertFalse(setup_is_complete(steps))
        self.assertFalse(next(s for s in steps if s.key == "model").done)
        self.assertFalse(next(s for s in steps if s.key == "upload").done)
        self.assertFalse(next(s for s in steps if s.key == "build").done)

    def test_huggingface_model_step_done_with_token_and_model(self) -> None:
        settings = AppSettings(
            generation_backend="huggingface",
            hf_token="tok",
            hf_llm_model="Qwen/Qwen2.5-7B-Instruct",
        )
        model_step = next(s for s in _steps(settings) if s.key == "model")
        self.assertTrue(model_step.done)

    def test_ollama_model_step_done_with_model_name(self) -> None:
        settings = AppSettings(generation_backend="ollama", ollama_model="llama3")
        model_step = next(s for s in _steps(settings) if s.key == "model")
        self.assertTrue(model_step.done)

    def test_duckduckgo_web_step_is_ready_without_key(self) -> None:
        settings = AppSettings(web_search_provider="duckduckgo")
        web_step = next(s for s in _steps(settings) if s.key == "web")
        self.assertTrue(web_step.done)

    def test_complete_when_all_conditions_met(self) -> None:
        settings = AppSettings(
            generation_backend="ollama",
            ollama_model="llama3",
            web_search_provider="duckduckgo",
        )
        steps = _steps(settings, docs=3, chunks=42)
        self.assertTrue(setup_is_complete(steps))


if __name__ == "__main__":
    unittest.main()
