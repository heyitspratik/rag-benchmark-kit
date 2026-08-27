"""Expanding a sweep into configurations, and grouping them by what they cost to build.

The grouping is the point of this module. Chunker and embedder determine what gets
indexed; the retriever does not. Running the default grid naively would ingest the corpus
24 times when 8 distinct indexes exist, so every retriever variant would pay the full
embedding cost again for an index identical to one already built. Grouping on
:meth:`PipelineConfig.index_fingerprint` turns that back into 8 ingestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rag_bench.core.config import SWEEPABLE_STAGES, PipelineConfig, SweepConfig
from rag_bench.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexGroup:
    """Configurations that share one built index, and can therefore share one ingestion."""

    fingerprint: str
    configurations: tuple[PipelineConfig, ...]

    @property
    def representative(self) -> PipelineConfig:
        """Any member, since they agree on everything the index depends on."""
        return self.configurations[0]

    def __len__(self) -> int:
        """How many configurations share this index."""
        return len(self.configurations)


def expand(sweep: SweepConfig, base: PipelineConfig) -> list[PipelineConfig]:
    """Expand a sweep into the Cartesian product of its axes.

    Axes are applied in pipeline order so the resulting list is stable between runs,
    which keeps a resumed run's progress comparable to the original.

    Args:
        sweep: The sweep declaration.
        base: The pipeline every configuration starts from.

    Returns:
        One config per point in the grid. A sweep with no axes yields the base alone.
    """
    axes = [(stage, sweep.sweep[stage]) for stage in SWEEPABLE_STAGES if stage in sweep.sweep]
    if not axes:
        return [base]

    configurations = []
    for combination in product(*(names for _, names in axes)):
        config = base
        for (stage, _), name in zip(axes, combination, strict=True):
            config = config.with_component(stage, name, sweep.params_for(stage, name))
        configurations.append(config)

    logger.info(
        "grid.expanded",
        configurations=len(configurations),
        axes={stage: len(names) for stage, names in axes},
    )
    return configurations


def group_by_index(configurations: list[PipelineConfig]) -> list[IndexGroup]:
    """Group configurations by the index they need, preserving first-seen order.

    Args:
        configurations: The expanded grid.

    Returns:
        One group per distinct index, each holding every configuration that can reuse it.
    """
    grouped: dict[str, list[PipelineConfig]] = {}
    for config in configurations:
        grouped.setdefault(config.index_fingerprint(), []).append(config)

    groups = [
        IndexGroup(fingerprint=fingerprint, configurations=tuple(members))
        for fingerprint, members in grouped.items()
    ]
    logger.info(
        "grid.grouped",
        configurations=len(configurations),
        indexes=len(groups),
        ingestions_saved=len(configurations) - len(groups),
    )
    return groups


def plan(sweep: SweepConfig, base: PipelineConfig) -> list[IndexGroup]:
    """Expand a sweep and group it in one step.

    Args:
        sweep: The sweep declaration.
        base: The pipeline every configuration starts from.

    Returns:
        Index groups covering the whole grid.
    """
    return group_by_index(expand(sweep, base))
