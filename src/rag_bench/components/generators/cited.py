"""Answer generation constrained to the retrieved context, with checked citations.

Two properties matter more than fluency here, and both are enforced in code rather than
hoped for in the prompt. An answer must cite context that was actually retrieved, and a
model with insufficient context must decline rather than invent. The benchmark measures
the second directly through its negative questions, so it has to be observable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from rag_bench.core.exceptions import CitationError, ConfigValidationError, LLMProviderError
from rag_bench.core.interfaces import BaseGenerator
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Answer, Citation, ScoredChunk, TokenUsage
from rag_bench.core.registry import register_generator
from rag_bench.core.settings import LLMSettings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

#: The model emits this exact token when the context does not support an answer. A
#: sentinel is used rather than prose matching because "I could not find" and "the
#: context does not say" are the same event and must be scored as one.
ABSTENTION_TOKEN = "INSUFFICIENT_CONTEXT"

#: What the user sees in place of the sentinel.
ABSTENTION_MESSAGE = (
    "The retrieved context does not contain enough information to answer this question."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You answer questions strictly from the numbered context passages you are given.

Rules:
1. Use only the context. Never use prior knowledge, and never guess.
2. Cite every claim with the passage number in square brackets, like [1] or [2][3].
3. Cite only numbers that appear in the context you were given.
4. If the context does not contain enough information to answer, reply with exactly
   {abstention_token} and nothing else. Do not apologise or speculate.
5. Be concise and factual. Quote defined terms exactly as the context words them.\
"""

USER_PROMPT = """\
Context passages:

{context}

Question: {question}

Answer:\
"""


@register_generator("cited")
class CitedGenerator(BaseGenerator):
    """Generates an answer from context and resolves its citations back to chunks."""

    def __init__(
        self,
        max_context_chars: int = 8000,
        temperature: float | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            max_context_chars: Budget for the assembled context. Passages are added best
                first and the rest are dropped, so a small budget costs recall rather
                than overflowing the model's window mid-benchmark.
            temperature: Overrides the configured sampling temperature. Benchmarks want
                zero, so that a rerun measures the pipeline and not the sampler.
            chat_model: A prebuilt chat model, used by tests to avoid a provider call.

        Raises:
            ConfigValidationError: If the context budget is not positive.
        """
        if max_context_chars <= 0:
            raise ConfigValidationError(
                f"generator.params.max_context_chars must be positive, got {max_context_chars}"
            )
        self._max_context_chars = max_context_chars
        self._temperature = temperature
        self._model = chat_model

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """Answer a question from the supplied context only.

        Args:
            question: The user's question.
            contexts: Retrieved chunks, best first.

        Returns:
            The answer, with citations resolved to the chunks they refer to.

        Raises:
            CitationError: If the model cited a passage number it was not given.
            LLMProviderError: If the provider call failed.
        """
        if not contexts:
            return self._abstention(question, contexts)

        used = self._fit_to_budget(contexts)
        messages = [
            ("system", SYSTEM_PROMPT.format(abstention_token=ABSTENTION_TOKEN)),
            ("user", USER_PROMPT.format(context=_render(used), question=question)),
        ]

        try:
            response = self._chat_model().invoke(messages)
        except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
            raise LLMProviderError(
                f"The chat model call failed: {exc}",
                details={"provider": get_settings().llm.provider},
            ) from exc

        text = _as_text(response.content).strip()
        usage = _usage(response)

        if ABSTENTION_TOKEN in text:
            return self._abstention(question, contexts, usage)

        citations = _resolve_citations(text, used)
        return Answer(
            question=question,
            text=text,
            abstained=False,
            citations=citations,
            contexts=tuple(contexts),
            usage=usage,
        )

    def _abstention(
        self,
        question: str,
        contexts: Sequence[ScoredChunk],
        usage: TokenUsage | None = None,
    ) -> Answer:
        """Build the answer for a question the context cannot support."""
        return Answer(
            question=question,
            text=ABSTENTION_MESSAGE,
            abstained=True,
            citations=(),
            contexts=tuple(contexts),
            usage=usage or TokenUsage(),
        )

    def _fit_to_budget(self, contexts: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Take passages best first until the character budget is spent."""
        kept: list[ScoredChunk] = []
        spent = 0
        for context in contexts:
            length = len(context.chunk.text)
            if kept and spent + length > self._max_context_chars:
                break
            kept.append(context)
            spent += length
        if len(kept) < len(contexts):
            logger.debug("generator.context_truncated", kept=len(kept), offered=len(contexts))
        return kept

    def _chat_model(self) -> BaseChatModel:
        """Build the chat model on first use, honouring any temperature override."""
        if self._model is None:
            from rag_bench.core.llm import build_chat_model

            settings = get_settings().llm
            if self._temperature is not None:
                settings = LLMSettings.model_validate(
                    settings.model_dump() | {"temperature": self._temperature}
                )
            self._model = build_chat_model(settings)
        return self._model


def _render(contexts: Sequence[ScoredChunk]) -> str:
    """Number the passages so the model has something short and unambiguous to cite."""
    blocks = []
    for position, context in enumerate(contexts, start=1):
        source = ", ".join(context.chunk.section_refs) or context.chunk.doc_id
        blocks.append(f"[{position}] ({source})\n{context.chunk.text}")
    return "\n\n".join(blocks)


def _resolve_citations(text: str, contexts: Sequence[ScoredChunk]) -> tuple[Citation, ...]:
    """Map the markers in an answer back to the chunks they name.

    Args:
        text: The generated answer.
        contexts: The passages the model was given, in the order it saw them.

    Returns:
        One citation per distinct marker, in the order first cited.

    Raises:
        CitationError: If a marker falls outside the passages supplied. An answer citing
            something that was never retrieved is a defect, not a formatting quirk, and
            silently dropping it would let a hallucinated source look like a real one.
    """
    seen: dict[int, None] = {}
    for match in _CITATION_RE.finditer(text):
        seen.setdefault(int(match.group(1)), None)

    citations = []
    for marker in seen:
        if not 1 <= marker <= len(contexts):
            raise CitationError(
                f"The answer cites passage [{marker}], but only {len(contexts)} passages "
                "were supplied",
                details={"marker": marker, "supplied": len(contexts)},
            )
        chunk = contexts[marker - 1].chunk
        citations.append(
            Citation(marker=str(marker), chunk_id=chunk.id, section_refs=chunk.section_refs)
        )
    return tuple(citations)


def _as_text(content: object) -> str:
    """Flatten a chat response body, which may arrive as a string or as content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _usage(response: object) -> TokenUsage:
    """Read token counts off a response, tolerating providers that report none."""
    metadata = getattr(response, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(metadata.get("input_tokens", 0) or 0),
        completion_tokens=int(metadata.get("output_tokens", 0) or 0),
    )
