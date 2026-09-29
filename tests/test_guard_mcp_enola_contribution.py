"""Launcher and tool-state validation for the enola MCP contribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_plugin_scanner.guard.local_cli_trust import apply_local_mcp_extension_decision
from codex_plugin_scanner.guard.mcp_tool_calls import build_tool_call_artifact
from codex_plugin_scanner.guard.runtime.command_extensions import BUILT_IN_COMMAND_EXTENSION_REGISTRY
from codex_plugin_scanner.guard.runtime.extension_control_authority import (
    AuthorityHealth,
    ExtensionControlAuthorityView,
)
from codex_plugin_scanner.guard.runtime.extension_control_contract import (
    CONTROL_SCHEMA_VERSION,
    ControlLayerKind,
    ControlState,
    ControlTarget,
    ControlTargetKind,
    ExtensionControl,
    ExtensionControlLayer,
)
from codex_plugin_scanner.guard.runtime.extension_trust import trust_class_for
from codex_plugin_scanner.guard.runtime.mcp_protection import build_mcp_server_identity
from codex_plugin_scanner.guard.runtime.mcp_server_contribution import mcp_tool_state, validate_mcp_contribution
from tests.support.extension_freshness import requires_fresh_projections

_ENOLA = Path(__file__).resolve().parents[1] / "contributions/mcp-servers/mcp.enola.json"
_CATALOG_ID = "command.mcp-enola"
_WRITING_TOOLS = ("generate_snapshot", "set_baseline", "diff_snapshot")
_READ_ONLY_TOOLS = (
    "query_facts",
    "show_symbol",
    "explore",
    "traverse",
    "find_path",
    "impact_analysis",
    "endpoint_impact",
    "query_insights",
    "governing_intent",
    "constraints_for",
    "plan_check",
    "coverage_report",
    "snapshot_receipt",
    "compare_receipts",
    "architecture_history",
    "architecture_blame",
    "find_orphans",
    "package_metrics",
    "analyze_performance",
)
_UVX_LAUNCH = ("uvx", ("--from", "enola-cli", "enola"))


def _payload() -> dict[str, object]:
    payload = json.loads(_ENOLA.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _artifact(tool_name: str, launch: tuple[str, tuple[str, ...]] = _UVX_LAUNCH):
    command, args = launch
    identity = build_mcp_server_identity(
        config_path="",
        command=command,
        args=args,
        transport="stdio",
    )
    return build_tool_call_artifact(
        harness="claude-code",
        server_name="enola",
        tool_name=tool_name,
        source_scope="user",
        config_path=".claude.json",
        transport="stdio",
        server_identity=identity,
    )


class _AuthorityStore:
    def __init__(self, layers: tuple[ExtensionControlLayer, ...] = ()) -> None:
        self.layers = layers

    def read_local_mcp_grant(self, *_args: object, **_kwargs: object) -> None:
        return None

    def read_extension_control_authority_for_registry(self, registry: object) -> ExtensionControlAuthorityView:
        digest = getattr(registry, "catalog_digest", "0" * 64)
        assert isinstance(digest, str)
        return ExtensionControlAuthorityView(
            health=AuthorityHealth.PROTECTED,
            revision=1,
            catalog_digest=digest,
            layers=self.layers,
        )


def _enabled(kind: ControlLayerKind, *, global_lockdown: bool = False) -> _AuthorityStore:
    return _AuthorityStore(
        (
            ExtensionControlLayer(
                schema_version=CONTROL_SCHEMA_VERSION,
                kind=kind,
                catalog_digest=BUILT_IN_COMMAND_EXTENSION_REGISTRY.catalog_digest,
                global_lockdown=global_lockdown,
                controls=(
                    ExtensionControl(
                        target=ControlTarget(ControlTargetKind.EXTENSION, _CATALOG_ID),
                        state=ControlState.ENABLED,
                    ),
                ),
            ),
        )
    )


def test_contribution_validates() -> None:
    validate_mcp_contribution(_payload(), filename="mcp.enola.json")


@requires_fresh_projections
def test_trust_map_marks_enola_external() -> None:
    assert trust_class_for(_CATALOG_ID) == "external"


@requires_fresh_projections
def test_catalog_item_is_external_opt_in_uvx_launch() -> None:
    extension = BUILT_IN_COMMAND_EXTENSION_REGISTRY.get(_CATALOG_ID)
    assert extension is not None
    payload = extension.to_dict()
    assert payload["enabled"] is False
    assert payload["trust_class"] == "external"
    assert payload["activation"] == "opt-in"
    assert payload["surface"] == "mcp"
    launch = payload["mcp_launch"]
    assert isinstance(launch, dict)
    assert launch["command"] == "uvx"
    assert launch["package"] == "enola-cli"


def test_every_listed_tool_is_classified() -> None:
    tools = _payload()["tools"]
    assert isinstance(tools, list)
    names = [tool["name"] for tool in tools if isinstance(tool, dict)]
    assert sorted(names) == sorted((*_READ_ONLY_TOOLS, *_WRITING_TOOLS, "other"))


@pytest.mark.parametrize("tool_name", _READ_ONLY_TOOLS)
def test_read_only_tools_are_allowed(tool_name: str) -> None:
    assert mcp_tool_state(_payload(), tool_name) == "allow"


@pytest.mark.parametrize("tool_name", _WRITING_TOOLS)
def test_state_writing_tools_inherit(tool_name: str) -> None:
    assert mcp_tool_state(_payload(), tool_name) == "inherit"


def test_unknown_tools_inherit() -> None:
    assert mcp_tool_state(_payload(), "unknown_tool") == "inherit"


@requires_fresh_projections
@pytest.mark.parametrize("tool_name", _READ_ONLY_TOOLS)
def test_allow_applies_only_after_local_admin_enable(tool_name: str) -> None:
    artifact = _artifact(tool_name)
    assert apply_local_mcp_extension_decision(_AuthorityStore(), artifact, "review") is None
    assert apply_local_mcp_extension_decision(_enabled(ControlLayerKind.SIGNED_CLOUD), artifact, "review") is None
    allowed = apply_local_mcp_extension_decision(_enabled(ControlLayerKind.LOCAL_ADMIN), artifact, "review")
    assert allowed is not None
    assert allowed[0] == "allow"
    assert allowed[1] == "catalog-mcp-extension"


def test_allow_never_lowers_a_block() -> None:
    artifact = _artifact("query_facts")
    assert apply_local_mcp_extension_decision(_enabled(ControlLayerKind.LOCAL_ADMIN), artifact, "block") is None


def test_allow_stands_down_under_global_lockdown() -> None:
    artifact = _artifact("query_facts")
    store = _enabled(ControlLayerKind.LOCAL_ADMIN, global_lockdown=True)
    assert apply_local_mcp_extension_decision(store, artifact, "review") is None


@pytest.mark.parametrize("tool_name", _WRITING_TOOLS)
def test_state_writing_tools_keep_guard_handling_when_enabled(tool_name: str) -> None:
    artifact = _artifact(tool_name)
    assert apply_local_mcp_extension_decision(_enabled(ControlLayerKind.LOCAL_ADMIN), artifact, "review") is None


@pytest.mark.parametrize(
    "launch",
    [
        ("uvx", ("--from", "enola-cli@0.4.26", "enola")),
        ("uvx", ("--from", "enola-cli==0.4.26", "enola")),
        ("pipx", ("run", "--spec", "enola-cli", "enola")),
    ],
)
def test_package_launch_variants_match(launch: tuple[str, tuple[str, ...]]) -> None:
    artifact = _artifact("query_facts", launch)
    allowed = apply_local_mcp_extension_decision(_enabled(ControlLayerKind.LOCAL_ADMIN), artifact, "review")
    assert allowed is not None
    assert allowed[0] == "allow"


def test_bare_binary_launch_is_not_matched() -> None:
    artifact = _artifact("query_facts", ("enola", ()))
    assert apply_local_mcp_extension_decision(_enabled(ControlLayerKind.LOCAL_ADMIN), artifact, "review") is None
