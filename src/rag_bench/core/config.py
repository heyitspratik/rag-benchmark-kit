"""Validated models for the committed YAML experiment configuration.

This is the second configuration layer. Settings describe *the machine*: hosts,
credentials, directories. They live in the environment. These files describe *an
experiment*: which chunker, which embedder, which retriever. They are committed, because
they are the reproducible description of a benchmark run.

Everything is validated on load so that a typo fails immediately, naming the offending
field, rather than raising a ``KeyError`` at configuration 19 of a two-hour sweep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from rag_bench.core.exceptions import ConfigValidationError, UnknownComponentError
from rag_bench.core.registry import CHUNKERS, EMBEDDERS, GENERATORS, LOADERS, RETRIEVERS, STORES


class _NameLookup(Protocol):
    """The read-only slice of a registry that config validation needs.

    A protocol rather than ``Registry[object]`` because ``Registry`` is invariant in its
    element type, so the six concrete registries have no common supertype.
    """

    def names(self) -> list[str]:
        """Every registered name, sorted."""
        ...

    def __contains__(self, name: object) -> bool:
        """Whether a name is registered."""
        ...


#: Stages a sweep is allowed to vary, in the order they appear in the pipeline.
SWEEPABLE_STAGES = ("chunker", "embedder", "store", "retriever", "generator")


class ComponentConfig(BaseModel):
    """A registry name plus the keyword arguments its constructor receives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)

    def with_name(self, name: str) -> ComponentConfig:
        """Return a copy of this component pinned to a different implementation."""
        return self.model_copy(update={"name": name})


class CorpusConfig(BaseModel):
    """Which corpus to ingest, and where its cached files live."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Registered loader name, e.g. 'eu_regulations'.")
    path: Path


class PipelineConfig(BaseModel):
    """A fully resolved description of one end-to-end pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: CorpusConfig
    chunker: ComponentConfig
    embedder: ComponentConfig
    store: ComponentConfig
    retriever: ComponentConfig
    generator: ComponentConfig

    def component(self, stage: str) -> ComponentConfig:
        """Return the component config for a stage name.

        Args:
            stage: One of :data:`SWEEPABLE_STAGES`.

        Returns:
            The component configuration.

        Raises:
            ConfigValidationError: If the stage is not sweepable.
        """
        if stage not in SWEEPABLE_STAGES:
            raise ConfigValidationError(
                f"Unknown pipeline stage {stage!r}. Expected one of: {', '.join(SWEEPABLE_STAGES)}"
            )
        value: ComponentConfig = getattr(self, stage)
        return value

    def with_component(self, stage: str, name: str) -> PipelineConfig:
        """Return a copy with one stage swapped to a different implementation.

        Args:
            stage: One of :data:`SWEEPABLE_STAGES`.
            name: The replacement implementation's registry name.

        Returns:
            A new config; the original is untouched.
        """
        return self.model_copy(update={stage: self.component(stage).with_name(name)})

    def fingerprint(self) -> str:
        """A stable short hash of the whole configuration.

        Used to key persisted results, so a resumed run recognises configurations it has
        already finished even across processes.
        """
        return _digest(self.model_dump(mode="json"))

    def index_fingerprint(self) -> str:
        """A stable short hash of only the stages that determine the built index.

        Retrieval strategy does not change what is indexed, so every retriever variant of
        the same corpus, chunker, embedder and store shares one index. The benchmark
        harness groups on this to avoid re-ingesting the corpus for each variant.
        """
        return _digest(
            {
                "corpus": self.corpus.model_dump(mode="json"),
                "chunker": self.chunker.model_dump(mode="json"),
                "embedder": self.embedder.model_dump(mode="json"),
                "store": self.store.model_dump(mode="json"),
            }
        )


class SweepConfig(BaseModel):
    """A benchmark experiment: a base pipeline plus the axes to vary over it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    eval_set: Path
    base_config: Path
    sweep: dict[str, list[str]]

    @model_validator(mode="after")
    def _check_sweep_axes(self) -> Self:
        """Reject unknown stages and empty axes before any expensive work starts."""
        unknown = sorted(set(self.sweep) - set(SWEEPABLE_STAGES))
        if unknown:
            raise ValueError(
                f"sweep: unknown stage(s) {', '.join(unknown)}. "
                f"Sweepable stages are: {', '.join(SWEEPABLE_STAGES)}"
            )
        for stage, names in self.sweep.items():
            if not names:
                raise ValueError(f"sweep.{stage}: needs at least one implementation name")
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(f"sweep.{stage}: duplicate entries {', '.join(duplicates)}")
        return self

    @property
    def run_count(self) -> int:
        """How many configurations this sweep expands to."""
        total = 1
        for names in self.sweep.values():
            total *= len(names)
        return total


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and validate a pipeline config file.

    Args:
        path: Path to a YAML file shaped like ``configs/default.yaml``.

    Returns:
        The validated config.

    Raises:
        ConfigValidationError: If the file is missing, unparseable, or invalid.
    """
    return _load_model(path, PipelineConfig)


def load_sweep_config(path: Path) -> SweepConfig:
    """Load and validate a benchmark sweep file.

    Args:
        path: Path to a YAML file shaped like ``configs/experiments/full_grid.yaml``.

    Returns:
        The validated config.

    Raises:
        ConfigValidationError: If the file is missing, unparseable, or invalid.
    """
    return _load_model(path, SweepConfig)


def validate_against_registries(config: PipelineConfig) -> None:
    """Check every component name in a config is actually registered.

    Separate from schema validation because it requires the implementation modules to have
    been imported; the caller decides when that happens.

    Args:
        config: A schema-valid pipeline config.

    Raises:
        UnknownComponentError: If any stage names an unregistered implementation.
    """
    registries: dict[str, _NameLookup] = {
        "corpus": LOADERS,
        "chunker": CHUNKERS,
        "embedder": EMBEDDERS,
        "store": STORES,
        "retriever": RETRIEVERS,
        "generator": GENERATORS,
    }
    missing: list[str] = []
    for stage, registry in registries.items():
        name = config.corpus.name if stage == "corpus" else config.component(stage).name
        if name not in registry:
            missing.append(f"{stage}={name!r} (available: {', '.join(registry.names()) or 'none'})")
    if missing:
        raise UnknownComponentError(
            "Config names components that are not registered: " + "; ".join(missing),
            details={"missing": missing},
        )


def _load_model[T: BaseModel](path: Path, model: type[T]) -> T:
    """Read YAML from ``path`` and validate it into ``model``."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError(
            f"Config file not found: {path}", details={"path": str(path)}
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"{path} is not valid YAML: {exc}", details={"path": str(path)}
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}",
            details={"path": str(path)},
        )

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"{path} is invalid:\n{_format_errors(exc)}",
            details={"path": str(path), "errors": exc.error_count()},
        ) from exc


def _format_errors(exc: ValidationError) -> str:
    """Render pydantic errors as one readable ``field: message`` line each."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def _digest(payload: object) -> str:
    """Return a short, stable hash of a JSON-serialisable payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
