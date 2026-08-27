from pathlib import Path

import pytest

from rag_bench.core.config import (
    PipelineConfig,
    load_pipeline_config,
    load_sweep_config,
    validate_against_registries,
)
from rag_bench.core.exceptions import ConfigValidationError, UnknownComponentError

DEFAULT_CONFIG = Path("configs/default.yaml")
FULL_GRID = Path("configs/experiments/full_grid.yaml")

_SWEEP_HEADER = "name: x\neval_set: e.jsonl\nbase_config: configs/default.yaml\n"


def test_the_committed_default_config_is_valid() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)

    assert config.corpus.name == "eu_regulations"
    assert config.chunker.name == "structural"
    assert config.retriever.params["top_k"] == 5


def test_the_committed_sweep_expands_to_twenty_four_runs() -> None:
    sweep = load_sweep_config(FULL_GRID)

    assert sweep.run_count == 24
    assert sweep.base_config == DEFAULT_CONFIG


def test_missing_file_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="not found"):
        load_pipeline_config(tmp_path / "nope.yaml")


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("corpus: [unclosed\n")

    with pytest.raises(ConfigValidationError, match="not valid YAML"):
        load_pipeline_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n")

    with pytest.raises(ConfigValidationError, match="mapping at the top level"):
        load_pipeline_config(path)


def test_invalid_config_names_the_offending_field(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("corpus:\n  name: eu_regulations\n  path: data/corpus\n")

    with pytest.raises(ConfigValidationError) as excinfo:
        load_pipeline_config(path)

    message = str(excinfo.value)
    assert "chunker: Field required" in message
    assert "generator: Field required" in message


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    path = tmp_path / "extra.yaml"
    path.write_text(DEFAULT_CONFIG.read_text() + "\nrerank: {}\n")

    with pytest.raises(ConfigValidationError, match="rerank: Extra inputs are not permitted"):
        load_pipeline_config(path)


def test_sweep_rejects_an_unsweepable_stage(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(_SWEEP_HEADER + "sweep:\n  reranker: [a, b]\n")

    with pytest.raises(ConfigValidationError, match=r"unknown stage\(s\) reranker"):
        load_sweep_config(path)


def test_sweep_rejects_an_empty_axis(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(_SWEEP_HEADER + "sweep:\n  chunker: []\n")

    with pytest.raises(ConfigValidationError, match="needs at least one implementation"):
        load_sweep_config(path)


def test_sweep_rejects_duplicate_entries(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(_SWEEP_HEADER + "sweep:\n  chunker: [fixed, fixed]\n")

    with pytest.raises(ConfigValidationError, match="duplicate entries fixed"):
        load_sweep_config(path)


def test_with_component_swaps_one_stage_only() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)

    swapped = config.with_component("retriever", "dense")

    assert swapped.retriever.name == "dense"
    assert swapped.retriever.params == config.retriever.params
    assert swapped.chunker == config.chunker
    assert config.retriever.name == "hybrid_rerank"


def test_component_rejects_a_stage_that_is_not_part_of_the_sweep() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)

    with pytest.raises(ConfigValidationError, match="Unknown pipeline stage"):
        config.component("corpus")


def test_fingerprint_is_stable_and_sensitive_to_any_change() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)

    assert config.fingerprint() == load_pipeline_config(DEFAULT_CONFIG).fingerprint()
    assert config.fingerprint() != config.with_component("retriever", "dense").fingerprint()


def test_index_fingerprint_ignores_the_retriever_but_not_the_chunker() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)
    other_retriever = config.with_component("retriever", "dense")
    other_chunker = config.with_component("chunker", "fixed")

    # Retrieval strategy does not change what was indexed; chunking does. This is what
    # lets the harness build 8 indexes for 24 runs.
    assert config.index_fingerprint() == other_retriever.index_fingerprint()
    assert config.index_fingerprint() != other_chunker.index_fingerprint()


def test_registry_validation_reports_the_unregistered_component() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG).with_component("chunker", "does_not_exist")

    with pytest.raises(UnknownComponentError) as excinfo:
        validate_against_registries(config)

    assert "chunker='does_not_exist'" in str(excinfo.value)


def test_pipeline_config_is_immutable() -> None:
    config: PipelineConfig = load_pipeline_config(DEFAULT_CONFIG)

    with pytest.raises(ValueError, match="frozen"):
        config.chunker = config.chunker  # type: ignore[misc]


def test_the_committed_full_grid_declares_per_implementation_params() -> None:
    # Without these, sweeping the retriever hands dense the base config's bm25_weight
    # and the configuration fails at construction.
    sweep = load_sweep_config(FULL_GRID)

    assert sweep.params_for("retriever", "dense") == {"top_k": 5}
    assert "bm25_weight" in (sweep.params_for("retriever", "hybrid") or {})
    # The semantic chunker cuts on meaning, so it has no overlap setting to receive.
    assert "overlap" not in (sweep.params_for("chunker", "semantic") or {})


def test_params_for_an_undeclared_implementation_is_none() -> None:
    sweep = load_sweep_config(FULL_GRID)

    assert sweep.params_for("embedder", "bge_small") is None
    assert sweep.params_for("retriever", "not_swept") is None


def test_params_for_a_stage_that_is_not_swept_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        _SWEEP_HEADER + "sweep:\n  chunker: [fixed]\nparams:\n  reranker:\n    x: {a: 1}\n"
    )

    with pytest.raises(ConfigValidationError, match=r"params: unknown stage 'reranker'"):
        load_sweep_config(path)


def test_params_for_an_implementation_outside_the_sweep_is_rejected(tmp_path: Path) -> None:
    # Almost always a typo, and silently ignoring it leaves the run using parameters
    # nobody intended.
    path = tmp_path / "sweep.yaml"
    path.write_text(
        _SWEEP_HEADER + "sweep:\n  chunker: [fixed]\nparams:\n  chunker:\n    structural: {a: 1}\n"
    )

    with pytest.raises(ConfigValidationError, match=r"params\.chunker: structural is not in"):
        load_sweep_config(path)


def test_with_component_can_replace_params_as_well_as_the_name() -> None:
    config = load_pipeline_config(DEFAULT_CONFIG)

    swapped = config.with_component("retriever", "dense", {"top_k": 3})

    assert swapped.retriever.name == "dense"
    assert swapped.retriever.params == {"top_k": 3}
    assert "bm25_weight" not in swapped.retriever.params
