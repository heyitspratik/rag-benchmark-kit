"""Static checks on the container build.

Docker is not available in the test environment, so these cannot prove the image runs.
They do prove the things that silently rot: a service losing its healthcheck, a
dependency ordered by nothing, a COPY of a path the build context excludes.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

COMPOSE = Path("docker/docker-compose.yml")
DOCKERFILE = Path("docker/Dockerfile")
DOCKERIGNORE = Path(".dockerignore")

#: Services that stay up and therefore need a healthcheck others can wait on.
LONG_RUNNING = {"postgres", "qdrant", "ollama", "api"}

#: Services that run once and exit.
ONE_SHOT = {"migrate", "bootstrap", "ollama-pull"}

VALID_CONDITIONS = {"service_healthy", "service_completed_successfully", "service_started"}


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text())


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text()


def test_the_stack_declares_every_service(compose: dict[str, Any]) -> None:
    assert set(compose["services"]) == LONG_RUNNING | ONE_SHOT


def test_every_long_running_service_has_a_healthcheck(compose: dict[str, Any]) -> None:
    # Without one, `depends_on: service_healthy` has nothing to wait on and the stack
    # races itself on startup.
    missing = [
        name
        for name in LONG_RUNNING
        if name != "api" and "healthcheck" not in compose["services"][name]
    ]

    assert missing == []


def test_the_api_healthcheck_lives_in_the_image(dockerfile: str) -> None:
    # The API's probe is a HEALTHCHECK instruction rather than a compose entry, so it
    # applies wherever the image runs and not only under compose.
    assert "HEALTHCHECK" in dockerfile
    assert "/health/live" in dockerfile


def test_no_dependency_is_ordered_by_nothing(compose: dict[str, Any]) -> None:
    # A bare `depends_on: [postgres]` waits for the container to exist, not for the
    # database to accept connections.
    for name, service in compose["services"].items():
        depends = service.get("depends_on", {})
        assert isinstance(depends, dict), f"{name} uses list-form depends_on"
        for upstream, spec in depends.items():
            assert spec.get("condition") in VALID_CONDITIONS, f"{name} -> {upstream}"


def test_the_api_waits_for_everything_it_needs(compose: dict[str, Any]) -> None:
    depends = compose["services"]["api"]["depends_on"]

    assert depends["postgres"]["condition"] == "service_healthy"
    assert depends["qdrant"]["condition"] == "service_healthy"
    assert depends["migrate"]["condition"] == "service_completed_successfully"
    assert depends["ollama-pull"]["condition"] == "service_completed_successfully"
    # Waits for the corpus to be indexed, so one curl against a fresh stack returns
    # an answer rather than INDEX_NOT_READY.
    assert depends["bootstrap"]["condition"] == "service_completed_successfully"


def test_migrations_run_before_anything_touches_the_database(
    compose: dict[str, Any],
) -> None:
    migrate = compose["services"]["migrate"]

    assert migrate["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "alembic" in " ".join(migrate["command"])


def test_one_shot_services_do_not_restart(compose: dict[str, Any]) -> None:
    # Without this a completed job is restarted forever, because compose cannot tell a
    # finished task from a crashed server.
    for name in ONE_SHOT:
        assert compose["services"][name].get("restart") == "no", name


def test_services_reach_each_other_by_container_name(compose: dict[str, Any]) -> None:
    # localhost inside a container is that container. This is exactly the mistake the
    # configurable base URL exists to prevent.
    environment = compose["services"]["api"]["environment"]

    assert "@postgres:" in environment["POSTGRES_DSN"]
    assert environment["QDRANT_URL"] == "http://qdrant:6333"
    assert environment["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert not any("localhost" in str(v) for v in environment.values())


def test_state_survives_a_restart(compose: dict[str, Any]) -> None:
    declared = set(compose["volumes"])
    mounts = {
        mount.split(":")[0]
        for service in compose["services"].values()
        for mount in service.get("volumes", [])
    }

    assert {"postgres_data", "qdrant_data", "ollama_models"} <= declared
    assert mounts <= declared, "a service mounts a volume that is not declared"


def test_model_weights_are_mounted_rather_than_baked_in(
    compose: dict[str, Any], dockerfile: str
) -> None:
    # Baking them in would re-download gigabytes on every rebuild and inflate the image.
    assert "hf_cache" in compose["volumes"]
    assert "hf_cache:/app/.cache/huggingface" in compose["services"]["api"]["volumes"]
    assert "HF_HOME=/app/.cache/huggingface" in dockerfile


def test_postgres_provides_the_pgvector_extension(compose: dict[str, Any]) -> None:
    assert "pgvector" in compose["services"]["postgres"]["image"]


def test_the_build_is_multi_stage(dockerfile: str) -> None:
    stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", dockerfile, re.MULTILINE)

    assert stages == ["builder", "runtime"]


def test_the_runtime_image_does_not_carry_the_build_tooling(dockerfile: str) -> None:
    runtime = dockerfile.split("AS runtime", 1)[1]

    assert "uv sync" not in runtime
    assert "--no-dev" in dockerfile.split("AS runtime", 1)[0]


def test_the_container_does_not_run_as_root(dockerfile: str) -> None:
    assert re.search(r"^USER appuser", dockerfile, re.MULTILINE)
    # After the last COPY, or the process could still write to its own code.
    assert dockerfile.index("USER appuser") > dockerfile.rindex("COPY --chown")


def test_writable_directories_are_owned_before_volumes_mount_over_them(
    dockerfile: str,
) -> None:
    # A named volume inherits the ownership of the image path it covers. Without this
    # the volume is created root-owned and the non-root process cannot write to it.
    assert "chown -R appuser:appuser /app/data" in dockerfile


def test_every_copied_path_exists_and_is_in_the_build_context(dockerfile: str) -> None:
    # A COPY of a path the .dockerignore excludes fails the build, and only at build
    # time, which is the slowest possible moment to find out.
    allowed = {
        line[1:].rstrip("/")
        for line in DOCKERIGNORE.read_text().splitlines()
        if line.startswith("!")
    }
    sources = [
        source
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "--from=" not in line
        for source in line.split()[1:-1]
        if not source.startswith("--")
    ]

    assert sources, "no COPY instructions found"
    for source in sources:
        assert Path(source).exists(), f"{source} does not exist in the repository"
        top = source.rstrip("/").split("/")[0]
        assert top in allowed or source.rstrip("/") in allowed, f"{source} is excluded"


def test_the_corpus_is_not_shipped_in_the_image() -> None:
    # It is large, gitignored, and baking it in would tie the image to one snapshot.
    assert "data/corpus/" in DOCKERIGNORE.read_text()
