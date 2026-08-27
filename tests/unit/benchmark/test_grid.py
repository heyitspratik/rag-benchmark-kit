from pathlib import Path

from rag_bench.benchmark.grid import expand, group_by_index, plan
from rag_bench.core.config import SweepConfig, load_pipeline_config, load_sweep_config

BASE = load_pipeline_config(Path("configs/default.yaml"))
FULL_GRID = Path("configs/experiments/full_grid.yaml")


def _sweep(**overrides: object) -> SweepConfig:
    fields: dict[str, object] = {
        "name": "test",
        "eval_set": Path("data/eval/smoke.jsonl"),
        "base_config": Path("configs/default.yaml"),
        "sweep": {"chunker": ["fixed", "structural"], "retriever": ["dense", "hybrid"]},
    }
    return SweepConfig.model_validate(fields | overrides)


def test_expansion_is_the_cartesian_product() -> None:
    configs = expand(_sweep(), BASE)

    assert len(configs) == 4
    assert {(c.chunker.name, c.retriever.name) for c in configs} == {
        ("fixed", "dense"),
        ("fixed", "hybrid"),
        ("structural", "dense"),
        ("structural", "hybrid"),
    }


def test_stages_not_swept_keep_the_base_value() -> None:
    for config in expand(_sweep(), BASE):
        assert config.embedder.name == BASE.embedder.name
        assert config.corpus.name == BASE.corpus.name


def test_expansion_order_is_stable() -> None:
    # A resumed run must see the same grid in the same order as the original.
    first = [c.fingerprint() for c in expand(_sweep(), BASE)]
    second = [c.fingerprint() for c in expand(_sweep(), BASE)]

    assert first == second


def test_a_sweep_with_no_axes_yields_the_base_alone() -> None:
    assert expand(_sweep(sweep={}), BASE) == [BASE]


def test_per_implementation_params_are_applied() -> None:
    # Without this, sweeping the retriever hands every implementation the base config's
    # params, and a dense retriever rejects bm25_weight.
    sweep = _sweep(
        sweep={"retriever": ["dense", "hybrid"]},
        params={"retriever": {"dense": {"top_k": 3}, "hybrid": {"top_k": 7}}},
    )

    by_name = {c.retriever.name: c.retriever.params for c in expand(sweep, BASE)}

    assert by_name["dense"] == {"top_k": 3}
    assert by_name["hybrid"] == {"top_k": 7}


def test_an_implementation_without_declared_params_keeps_the_base_params() -> None:
    sweep = _sweep(
        sweep={"retriever": ["dense", "hybrid"]},
        params={"retriever": {"dense": {"top_k": 3}}},
    )

    by_name = {c.retriever.name: c.retriever.params for c in expand(sweep, BASE)}

    assert by_name["hybrid"] == BASE.retriever.params


def test_grouping_collapses_configurations_that_share_an_index() -> None:
    # The retriever does not change what is indexed, so both retriever variants of one
    # chunker share a single ingestion.
    groups = group_by_index(expand(_sweep(), BASE))

    assert len(groups) == 2
    assert all(len(group) == 2 for group in groups)


def test_a_group_member_can_stand_in_for_the_others() -> None:
    group = group_by_index(expand(_sweep(), BASE))[0]

    assert all(
        c.index_fingerprint() == group.representative.index_fingerprint()
        for c in group.configurations
    )


def test_changing_the_chunker_forces_a_separate_index() -> None:
    groups = group_by_index(expand(_sweep(sweep={"chunker": ["fixed", "structural"]}), BASE))

    assert len(groups) == 2


def test_the_committed_full_grid_needs_eight_indexes_for_twenty_four_runs() -> None:
    # This is the saving the harness exists to make: 24 configurations, 8 ingestions.
    sweep = load_sweep_config(FULL_GRID)

    groups = plan(sweep, BASE)

    assert sum(len(g) for g in groups) == 24
    assert len(groups) == 8
    assert all(len(g) == 3 for g in groups)
