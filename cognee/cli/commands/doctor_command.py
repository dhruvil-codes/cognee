import argparse
import asyncio
import importlib.metadata
import importlib.util
import logging
import os
import shutil
import sys
from contextlib import contextmanager

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException
from cognee.cli.reference import SupportsCliCommand
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@contextmanager
def _suppressed_error_logs():
    """The checks below log failures at ERROR with full tracebacks (meant for
    server logs); doctor reports the same failures as ✗ lines, so silence the
    duplicate traceback dumps for the duration of the check."""
    logging.disable(logging.ERROR)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


# Environment checks (zero network, hermetic). Each returns (status, message)
# with status one of "pass" / "warn" / "fail"; only failures fail the command.
# Messages never embed secret values.
_SUPPORTED_PYTHON_RANGE = ((3, 10), (3, 14))


def _check_python(version: tuple[int, int, int] | None = None) -> tuple[str, str]:
    """Pass while the running Python is within the supported 3.10–3.14 range."""
    version = version or tuple(sys.version_info[:3])
    low, high = _SUPPORTED_PYTHON_RANGE
    detail = f"Python {version[0]}.{version[1]}.{version[2]} (supported: 3.10–3.14)"
    return ("pass" if low <= version[:2] <= high else "fail"), detail


def _check_cognee_version() -> tuple[str, str]:
    try:
        version = importlib.metadata.version("cognee")
    except importlib.metadata.PackageNotFoundError:
        return "fail", "cognee package metadata not found (broken or non-uv install?)"
    return "pass", f"cognee {version} installed"


def _check_llm_api_key() -> tuple[str, str]:
    if (os.environ.get("LLM_API_KEY") or "").strip():
        # Never echo the key itself.
        return "pass", "LLM_API_KEY is set (value hidden)"
    return (
        "warn",
        "LLM_API_KEY not set — running in keyless mode (LLM-dependent stages will be skipped)",
    )


def _check_gliner() -> tuple[str, str]:
    # find_spec only — importing gliner2 pulls torch, which doctor must not do.
    if importlib.util.find_spec("gliner2") is not None:
        return "pass", "gliner extra installed"
    return (
        "warn",
        'gliner extra not installed — keyless graph extraction unavailable (pip install "cognee[gliner]")',
    )


def _check_ui_and_mcp_tools() -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []
    if shutil.which("node") and shutil.which("npm"):
        checks.append(("pass", "node and npm found"))
    else:
        checks.append(("warn", "Node.js/npm not found — required for `cognee-cli -ui`"))
    if shutil.which("docker"):
        checks.append(("pass", "docker found"))
    else:
        checks.append(("warn", "Docker not found — required for the MCP service"))
    return checks


class DoctorCommand(SupportsCliCommand):
    """Diagnose cognee configuration and local services before first use."""

    command_string = "doctor"
    help_string = "Diagnose configuration and local services (config traps, databases, providers)"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `doctor` command runs preflight diagnostics and prints a pass/fail report:

0. Environment — Python version range, installed cognee version, whether
   LLM_API_KEY is set (never printed), gliner extra importability, and
   Node.js/npm + Docker availability for the UI and MCP service. Warnings
   here (keyless mode, missing Node/Docker) do not fail the command.
1. Provider configuration consistency — catches the only-LLM-or-only-embeddings
   trap where the unconfigured side silently defaults to OpenAI and breaks
   minutes into the first ingestion.
2. Local services — relational, vector and graph databases plus file storage
   (same checks as the /health endpoint).
3. With --probe: live LLM and embedding round-trips (makes network calls and
   may incur token costs).

Exits non-zero when any check fails, so it can gate CI and setup scripts.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--probe",
            action="store_true",
            help="Also run live LLM and embedding connectivity probes (network calls; may cost tokens)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        failures = asyncio.run(self._run(probe=args.probe))
        if failures:
            raise CliCommandException(
                f"doctor found {failures} problem{'s' if failures != 1 else ''} (see above)",
                error_code=1,
            )
        fmt.success("\nAll checks passed. cognee is ready to use.")

    async def _run(self, probe: bool) -> int:
        failures = 0

        # --- 0. Environment (zero network; warnings never fail the command) ---
        fmt.bold("Environment")
        env_checks: list[tuple[str, str]] = [
            _check_python(),
            _check_cognee_version(),
            _check_llm_api_key(),
            _check_gliner(),
            *_check_ui_and_mcp_tools(),
        ]
        _glyphs = {"pass": "✓", "warn": "⚠️", "fail": "✗"}
        for status, message in env_checks:
            line = f"  {_glyphs[status]} {message}"
            if status == "fail":
                fmt.error(line)
                failures += 1
            elif status == "warn":
                fmt.echo(line, color="yellow")
            else:
                fmt.echo(line)

        # --- 1. Provider configuration consistency (zero network) ---
        fmt.bold("Configuration")
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_context_config,
        )
        from cognee.infrastructure.llm.config import get_llm_context_config
        from cognee.modules.preflight import check_provider_config

        llm_config = get_llm_context_config()
        embedding_config = get_embedding_context_config()
        fmt.echo(
            f"  LLM:        provider={llm_config.llm_provider} model={llm_config.llm_model} "
            f"api_key={'set' if (llm_config.llm_api_key or '').strip() else 'NOT SET'}"
        )
        fmt.echo(
            f"  Embeddings: provider={embedding_config.embedding_provider} "
            f"model={embedding_config.embedding_model} "
            f"api_key={'set' if (embedding_config.embedding_api_key or '').strip() else 'NOT SET'} "
            f"dimensions={embedding_config.embedding_dimensions}"
        )

        problems = check_provider_config(llm_config, embedding_config)
        for problem in problems:
            fmt.error(f"  ✗ {problem}")
            failures += 1
        if not problems:
            fmt.echo("  ✓ LLM/embedding provider configuration is consistent")

        # --- 2. Local services (same checks as the /health endpoint) ---
        fmt.bold("\nServices")
        from cognee.api.v1.health.health import HealthStatus, health_checker

        with _suppressed_error_logs():
            health = await health_checker.get_health_status(detailed=False)
        for name, component in health.components.items():
            if component.status == HealthStatus.HEALTHY:
                fmt.echo(f"  ✓ {name} ({component.provider}, {component.response_time_ms}ms)")
            else:
                fmt.error(f"  ✗ {name} ({component.provider}): {component.details}")
                failures += 1
                if name == "relational_db":
                    fmt.note(
                        "A missing local database is created on first use — run any "
                        "`cognee-cli remember` command, or check DB_* settings if you "
                        "configured Postgres."
                    )

        # --- 3. Optional live provider probes ---
        if probe:
            fmt.bold("\nProviders (live probes)")
            from cognee.infrastructure.llm.utils import (
                test_embedding_connection,
                test_llm_connection,
            )

            try:
                with _suppressed_error_logs():
                    await test_llm_connection()
                fmt.echo("  ✓ LLM round-trip succeeded")
            except Exception as error:
                logger.debug("Ignoring exception in DoctorCommand._run", exc_info=True)
                fmt.error(f"  ✗ LLM round-trip failed: {error}")
                failures += 1

            try:
                with _suppressed_error_logs():
                    dimensions = await test_embedding_connection()
                fmt.echo(f"  ✓ Embedding round-trip succeeded (dimensions={dimensions})")
            except Exception as error:
                logger.debug("Ignoring exception in DoctorCommand._run", exc_info=True)
                fmt.error(f"  ✗ Embedding round-trip failed: {error}")
                failures += 1
        else:
            fmt.echo("")
            fmt.note("Run `cognee-cli doctor --probe` to also test live LLM/embedding calls.")

        return failures
