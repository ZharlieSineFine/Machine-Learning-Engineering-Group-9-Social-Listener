#Validate the GitHub Actions CI workflow.
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.exists(), f"missing workflow at {WORKFLOW}"
    with WORKFLOW.open() as fh:
        return yaml.safe_load(fh)


def test_workflow_has_expected_jobs(workflow):
    expected = {"lint", "unit-tests", "smoke-docker", "compose-integration"}
    assert expected.issubset(set(workflow["jobs"].keys()))


def test_workflow_triggers_on_pr_and_main(workflow):
    # PyYAML parses the bare `on` key as the boolean True. Handle both.
    triggers = workflow.get("on") or workflow.get(True)
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]


def test_workflow_references_existing_requirements(workflow):
    """The unit-tests job installs from three requirements files. They must exist."""
    steps_text = yaml.dump(workflow["jobs"]["unit-tests"])
    for req in ["api/requirements.txt", "dashboard/requirements.txt", "tests/requirements.txt"]:
        assert req in steps_text, f"CI references {req} but it must exist"
        assert (ROOT / req).exists(), f"CI references {req} but it does not exist in repo"


def test_workflow_runs_pytest_excluding_integration(workflow):
    """The unit-tests job must NOT run tests/integration (those need Docker)."""
    steps_text = yaml.dump(workflow["jobs"]["unit-tests"])
    assert "--ignore=tests/integration" in steps_text


def test_workflow_smoke_docker_references_real_dockerfile(workflow):
    steps_text = yaml.dump(workflow["jobs"]["smoke-docker"])
    assert "infra/docker/smoke/Dockerfile" in steps_text
    assert (ROOT / "infra" / "docker" / "smoke" / "Dockerfile").exists()


def test_ruff_version_is_pinned(workflow):
    """Reproducibility: lint job should pin ruff so a new ruff release can't break a green PR."""
    steps_text = yaml.dump(workflow["jobs"]["lint"])
    assert "ruff==" in steps_text, "pin ruff with == in CI"


def test_pyproject_ruff_config_present():
    """Local `ruff check .` and CI must use the same ruleset."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "[tool.ruff]" in pyproject


def test_compose_integration_job_depends_on_cheap_jobs(workflow):
    """Don't waste compose-minutes on a branch whose lint/unit-tests are red."""
    needs = workflow["jobs"]["compose-integration"].get("needs", [])
    assert "lint" in needs
    assert "unit-tests" in needs


def test_compose_integration_runs_pytest_integration_dir(workflow):
    steps_text = yaml.dump(workflow["jobs"]["compose-integration"])
    assert "tests/integration" in steps_text


def test_compose_integration_brings_up_minimum_services(workflow):
    """The compose-integration job must boot at least postgres, minio, mlflow."""
    steps_text = yaml.dump(workflow["jobs"]["compose-integration"])
    for svc in ("postgres", "minio", "mlflow"):
        assert svc in steps_text, f"CI integration job should start `{svc}`"


def test_compose_integration_collects_logs_on_failure(workflow):
    """When CI integration fails, surface docker compose logs — saves hours."""
    steps_text = yaml.dump(workflow["jobs"]["compose-integration"])
    assert "if: failure()" in steps_text
    assert "compose logs" in steps_text
