"""Builders shared by the metric and scorer tests."""

from rag_bench.benchmark.evalset import Difficulty, EvalQuestion
from rag_bench.benchmark.metrics import QuestionOutcome
from rag_bench.core.models import Answer, Chunk, ScoredChunk, TokenUsage


def question(qid: str, refs: tuple[str, ...], difficulty: str = "single_hop") -> EvalQuestion:
    """An evaluation question with the given references."""
    return EvalQuestion(
        id=qid,
        question="Q?",
        ground_truth="A",
        source_refs=refs,
        difficulty=Difficulty(difficulty),
    )


def answer(
    ref_ranks: list[tuple[str, ...]],
    *,
    abstained: bool = False,
    retrieval_ms: float = 10.0,
    generation_ms: float = 100.0,
    usage: TokenUsage | None = None,
) -> Answer:
    """An answer whose contexts carry the given section refs, in rank order."""
    contexts = tuple(
        ScoredChunk(
            chunk=Chunk.create(
                doc_id="d", ordinal=rank, text=f"chunk {rank}", char_start=0, char_end=5
            ).with_sections(refs),
            score=1.0 - rank / 10,
            rank=rank,
        )
        for rank, refs in enumerate(ref_ranks)
    )
    return Answer(
        question="Q?",
        text="A",
        abstained=abstained,
        contexts=contexts,
        usage=usage or TokenUsage(),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


def outcome(
    refs: tuple[str, ...],
    ranks: list[tuple[str, ...]],
    difficulty: str = "single_hop",
    **kwargs: object,
) -> QuestionOutcome:
    """A question paired with an answer over the given retrieved refs."""
    return QuestionOutcome(
        question=question("q", refs, difficulty),
        answer=answer(ranks, **kwargs),  # type: ignore[arg-type]
    )
