"""The Agent templates tab's backend: roster, create, delete, and the widened PATCH.

A template is a shared definition; the tab manages the shared file itself, as
distinct from the crew pane's private copies. The rules under test are the ones a
management page must not get wrong: a package or runtime spec is read-only
(its installer rewrites it), a template still referenced by a crew, the default
agent, a schedule, a chat folder, a webhook or a private copy cannot be deleted,
and a create never lands on a name a crew binding or an installed spec already
resolves.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from kiro_crew import agent_state
from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig
from kiro_crew.config.paths import config_dir
from kiro_crew.dashboard.handlers.agent_templates import (
    api_agent_template_create,
    api_agent_template_delete,
    api_agent_templates,
)
from kiro_crew.dashboard.handlers.agents import api_agent_detail
from kiro_crew.webhooks import token_store


@pytest.fixture(autouse=True)
def _owner_caller(monkeypatch):
    """Past the owner boundary; owner-auth has its own enumerated coverage."""
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
        lambda request: True,
    )


@pytest.fixture
def agents_dir(tmp_path):
    d = tmp_path / "agents"
    d.mkdir()
    with patch("kiro_crew.agent.KIRO_AGENTS_DIR", d):
        yield d


def _request(
    method: str,
    name: str | None = None,
    body=None,
    *,
    bad_json: bool = False,
    folders: list[dict] | None = None,
):
    request = MagicMock(spec=web.Request)
    request.method = method
    request.match_info = {"name": name} if name else {}
    state = MagicMock()
    # The folder store is read on the loop through ``read_folders(reader)``;
    # the handlers see whatever *folders* this request's dashboard holds.
    state.read_folders = AsyncMock(side_effect=lambda read: read(folders or []))

    # ...and held across a section through ``hold_folders(section)``.
    async def _hold(section):
        return await section(folders or [])

    state.hold_folders = AsyncMock(side_effect=_hold)
    request.app = {"state": state}

    async def _json():
        if bad_json:
            raise ValueError("not json")
        return body

    request.json = _json
    return request


def _write(agents_dir, filename: str, **spec) -> None:
    data = {"name": filename.rsplit(".", 1)[0], "tools": ["fs_read"], **spec}
    (agents_dir / filename).write_text(json.dumps(data), encoding="utf-8")


def _seed_config(agents: dict[str, str] | None = None, default: str = "") -> None:
    cfg = KiroCrewConfig()
    cfg.agents = {crew: KiroCrewAgentConfig(kiro_agent=t) for crew, t in (agents or {}).items()}
    if default:
        cfg.default_agent = default
    cfg.save()


def _seed_cron(agent_id: str, name: str = "nightly triage") -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    (config_dir() / "crons.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job-1",
                        "name": name,
                        "message": "go",
                        "schedule": {"kind": "every", "every_secs": 3600},
                        "agent_id": agent_id,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


async def _body(resp):
    return json.loads(resp.text)


# ── roster ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_roster_marks_editability_and_references(agents_dir):
    _write(agents_dir, "reviewer.json", description="mine")
    _write(agents_dir, "SomePkg-atlas.json", name="atlas")
    _write(agents_dir, "kirocrew-worker.json", name="kirocrew-worker")
    _seed_config({"pr-bot": "reviewer"}, default="pr-bot")
    _seed_cron("reviewer")
    agent_state.set_fork_info("pr-bot-copy", forked_from="reviewer", private_to="pr-bot")

    resp = await api_agent_templates(_request("GET"))
    assert resp.status == 200
    rows = {r["name"]: r for r in (await _body(resp))["templates"]}

    assert rows["reviewer"]["read_only"] is None
    kinds = [(u["kind"], u["id"]) for u in rows["reviewer"]["used_by"]]
    assert ("crew", "pr-bot") in kinds
    assert ("schedule", "job-1") in kinds
    assert ("private_copy", "pr-bot-copy") in kinds
    # The package file is read-only for the package's reason, the runtime file
    # for the runtime's -- two different remedies, so two different labels.
    assert rows["atlas"]["read_only"] == "package"
    assert rows["kirocrew-worker"]["read_only"] == "runtime"
    assert rows["atlas"]["used_by"] == []


@pytest.mark.asyncio
async def test_roster_counts_a_chat_folder_pin_as_a_reference(agents_dir):
    """A folder's ``default_agent`` is what every session filed there starts on."""
    _write(agents_dir, "reviewer.json")
    _write(agents_dir, "scratch.json")
    _seed_config()
    folders = [
        {"id": "f-1", "name": "Reviews", "default_agent": "reviewer"},
        {"id": "f-2", "name": "Inherits", "default_agent": ""},
    ]
    resp = await api_agent_templates(_request("GET", folders=folders))
    rows = {r["name"]: r for r in (await _body(resp))["templates"]}
    assert rows["reviewer"]["used_by"] == [{"kind": "folder", "id": "f-1", "label": "Reviews"}]
    assert rows["scratch"]["used_by"] == []


# ── create ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_roster_counts_a_webhook_agent_pin_as_a_reference(agents_dir):
    """A webhook token's ``agent`` is what its calls run; gone, they are refused."""
    _write(agents_dir, "reviewer.json")
    _seed_config()
    _raw, _secret, entry = token_store().create(
        "ci-hook", require_signature=False, agent="reviewer"
    )
    resp = await api_agent_templates(_request("GET"))
    rows = {r["name"]: r for r in (await _body(resp))["templates"]}
    assert rows["reviewer"]["used_by"] == [
        {"kind": "webhook", "id": entry["id"], "label": "ci-hook"}
    ]


# ── create ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_blank_writes_a_minimal_runnable_spec(agents_dir):
    _seed_config()
    resp = await api_agent_template_create(
        _request("POST", body={"name": "pr-summarizer", "description": "Sums up PRs"})
    )
    assert resp.status == 201, resp.text
    assert (await _body(resp)) == {
        "ok": True,
        "name": "pr-summarizer",
        "filename": "pr-summarizer.json",
    }
    spec = json.loads((agents_dir / "pr-summarizer.json").read_text(encoding="utf-8"))
    assert spec["name"] == "pr-summarizer"
    assert spec["description"] == "Sums up PRs"
    assert spec["tools"] and spec["prompt"] == ""
    # A created template is shared, not anyone's private copy.
    assert agent_state.get_fork_info("pr-summarizer") is None


@pytest.mark.asyncio
async def test_create_from_copies_a_package_template_without_lineage(agents_dir):
    _write(agents_dir, "SomePkg-atlas.json", name="atlas", prompt="Be grounded.", tools=["x"])
    _seed_config()
    resp = await api_agent_template_create(
        _request("POST", body={"name": "my-atlas", "from": "atlas"})
    )
    assert resp.status == 201, resp.text
    spec = json.loads((agents_dir / "my-atlas.json").read_text(encoding="utf-8"))
    assert spec["name"] == "my-atlas"
    assert spec["prompt"] == "Be grounded."
    assert spec["tools"] == ["x"]
    assert agent_state.get_fork_info("my-atlas") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, status, code",
    [
        ({"name": "has space"}, 400, "invalid_template_name"),
        ({"name": "kirocrew"}, 400, "template_name_reserved"),
        ({"name": "x", "from": "nope"}, 404, "template_not_found"),
        ({"name": "reviewer"}, 409, "name_taken"),
        ({"name": "pr-bot"}, 409, "name_bound"),
    ],
)
async def test_create_refusals(agents_dir, body, status, code):
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "reviewer"})
    resp = await api_agent_template_create(_request("POST", body=body))
    assert resp.status == status, resp.text
    assert (await _body(resp))["code"] == code
    assert not (agents_dir / f"{body['name']}.json").exists() or body["name"] == "reviewer"


@pytest.mark.asyncio
async def test_create_copies_the_source_as_it_is_inside_the_lock(agents_dir, monkeypatch):
    """The pre-lock read only resolves the source's path; its body is read again
    under the spec lock, so an edit that lands in between is what gets copied."""
    from kiro_crew.dashboard.handlers import agents as agents_handlers

    _write(agents_dir, "SomePkg-atlas.json", name="atlas", prompt="v1")
    _seed_config()
    real = agents_handlers._spec_stem_on_disk

    def _edit_source_then_check(agents_dir_, name):
        # Runs inside ``agents_spec_lock`` right before the source is re-read:
        # the latest point a concurrent save could have committed.
        (agents_dir / "SomePkg-atlas.json").write_text(
            json.dumps({"name": "atlas", "tools": ["fs_read"], "prompt": "v2"}), encoding="utf-8"
        )
        return real(agents_dir_, name)

    monkeypatch.setattr(agents_handlers, "_spec_stem_on_disk", _edit_source_then_check)
    resp = await api_agent_template_create(_request("POST", body={"name": "mine", "from": "atlas"}))
    assert resp.status == 201, resp.text
    assert json.loads((agents_dir / "mine.json").read_text(encoding="utf-8"))["prompt"] == "v2"


@pytest.mark.asyncio
async def test_successful_create_and_delete_emit_operation_labelled_sel_events(
    agents_dir, monkeypatch
):
    """The owner gate logs only denials; a successful write of a machine-global
    spec gets its own labelled line beside the middleware's request-level one."""
    from kiro_crew.dashboard.handlers import agent_templates as module

    events: list[dict] = []
    monkeypatch.setattr(
        module,
        "sel",
        lambda: types.SimpleNamespace(log_api_access=lambda **kw: events.append(kw)),
    )
    _seed_config()
    assert (
        await api_agent_template_create(_request("POST", body={"name": "scratch"}))
    ).status == 201
    assert (await api_agent_template_delete(_request("DELETE", "scratch"))).status == 200
    assert [(e["operation"], e["outcome"], e["resources"]) for e in events] == [
        ("agent_templates.create", "ok", "template:scratch"),
        ("agent_templates.delete", "ok", "template:scratch file:scratch.json"),
    ]
    # A refused write logs nothing here: the gate and the middleware own that trail.
    assert (
        await api_agent_template_create(_request("POST", body={"name": "has space"}))
    ).status == 400
    assert len(events) == 2


@pytest.mark.asyncio
async def test_create_refuses_a_name_bound_only_in_the_overlay(agents_dir):
    """A binding present only in ``config.local.json`` still makes the new file
    resolve for a dangling reference the moment it lands."""
    from kiro_crew.config.loader import config_local_path

    _seed_config()
    config_local_path().write_text(
        json.dumps({"agents": {"pr-bot": {"kiro_agent": "ghost-writer"}}}), encoding="utf-8"
    )
    resp = await api_agent_template_create(_request("POST", body={"name": "ghost-writer"}))
    assert resp.status == 409, resp.text
    assert (await _body(resp))["code"] == "name_bound"
    assert not (agents_dir / "ghost-writer.json").exists()


@pytest.mark.asyncio
async def test_create_from_an_ambiguous_name_is_a_409_not_a_500(agents_dir):
    """Two files declaring one name is the user's to untangle, and is said so."""
    _write(agents_dir, "atlas.json", name="atlas")
    _write(agents_dir, "SomePkg-atlas.json", name="atlas")
    _seed_config()
    resp = await api_agent_template_create(_request("POST", body={"name": "mine", "from": "atlas"}))
    assert resp.status == 409, resp.text
    assert (await _body(resp))["code"] == "ambiguous_template_name"
    assert not (agents_dir / "mine.json").exists()


@pytest.mark.asyncio
async def test_create_rejects_malformed_bodies(agents_dir):
    _seed_config()
    assert (await api_agent_template_create(_request("POST", bad_json=True))).status == 400
    assert (await api_agent_template_create(_request("POST", body=["x"]))).status == 400


# ── delete ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_refuses_while_referenced_and_lists_the_references(agents_dir):
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "reviewer"}, default="pr-bot")
    _seed_cron("reviewer")
    resp = await api_agent_template_delete(_request("DELETE", "reviewer"))
    assert resp.status == 409
    body = await _body(resp)
    assert body["code"] == "template_referenced"
    assert {(r["kind"], r["id"]) for r in body["references"]} == {
        ("crew", "pr-bot"),
        ("schedule", "job-1"),
    }
    assert (agents_dir / "reviewer.json").exists()


@pytest.mark.asyncio
async def test_delete_refuses_while_a_chat_folder_pins_the_template(agents_dir):
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "kirocrew"}, default="pr-bot")
    folders = [{"id": "f-1", "name": "Reviews", "default_agent": "reviewer"}]
    resp = await api_agent_template_delete(_request("DELETE", "reviewer", folders=folders))
    assert resp.status == 409
    body = await _body(resp)
    assert body["code"] == "template_referenced"
    assert body["references"] == [{"kind": "folder", "id": "f-1", "label": "Reviews"}]
    assert (agents_dir / "reviewer.json").exists()


@pytest.mark.asyncio
async def test_delete_checks_and_unlinks_inside_the_folder_store_hold(agents_dir):
    """A pin that lands while the guard runs is seen: the check and the unlink
    are one section run while the folder store lock is held, on the committed
    list, so there is no snapshot for a concurrent folder update to go stale
    against -- and the section's file work happens off the loop."""
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "kirocrew"}, default="pr-bot")
    request = _request("DELETE", "reviewer")
    live: list[dict] = []

    async def _hold(section):
        # The store as the guard sees it under the lock -- a folder pinned after
        # the request started but before the lock was taken.
        live.append({"id": "f-late", "name": "Late", "default_agent": "reviewer"})
        return await section(live)

    request.app["state"].hold_folders = AsyncMock(side_effect=_hold)
    resp = await api_agent_template_delete(request)
    assert resp.status == 409
    assert (await _body(resp))["references"] == [
        {"kind": "folder", "id": "f-late", "label": "Late"}
    ]
    assert (agents_dir / "reviewer.json").exists()
    # One hold for the whole section; no separate snapshot read.
    request.app["state"].hold_folders.assert_awaited_once()
    request.app["state"].read_folders.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_counts_a_binding_that_lives_only_in_the_overlay(agents_dir):
    """``config.local.json`` deep-merges over the base and can carry a crew's
    effective ``kiro_agent`` on its own; the guard reads the merged config
    under both layers' locks."""
    from kiro_crew.config.loader import config_local_path

    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "kirocrew"}, default="pr-bot")
    config_local_path().write_text(
        json.dumps({"agents": {"pr-bot": {"kiro_agent": "reviewer"}}}), encoding="utf-8"
    )
    resp = await api_agent_template_delete(_request("DELETE", "reviewer"))
    assert resp.status == 409
    assert (await _body(resp))["references"] == [
        {"kind": "crew", "id": "pr-bot", "label": "pr-bot"}
    ]
    assert (agents_dir / "reviewer.json").exists()


@pytest.mark.asyncio
async def test_delete_refuses_while_a_webhook_names_the_template(agents_dir):
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "kirocrew"}, default="pr-bot")
    _raw, _secret, entry = token_store().create(
        "ci-hook", require_signature=False, agent="reviewer"
    )
    resp = await api_agent_template_delete(_request("DELETE", "reviewer"))
    assert resp.status == 409
    assert (await _body(resp))["references"] == [
        {"kind": "webhook", "id": entry["id"], "label": "ci-hook"}
    ]
    assert (agents_dir / "reviewer.json").exists()


@pytest.mark.asyncio
async def test_delete_refuses_the_default_agent_by_stem_too(agents_dir):
    _write(agents_dir, "scratch.json", name="Scratch Pad")
    cfg = KiroCrewConfig()
    cfg.agent.default_agent = "scratch"
    cfg.save()
    resp = await api_agent_template_delete(_request("DELETE", "scratch"))
    assert resp.status == 409
    # The fresh config's own `default` crew resolves the same template, so the
    # crew row rides along; the fallback row is what this test pins.
    assert "default" in {r["kind"] for r in (await _body(resp))["references"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename, reason",
    [("SomePkg-atlas.json", "package"), ("kirocrew-worker.json", "runtime")],
)
async def test_delete_refuses_read_only_specs(agents_dir, filename, reason):
    _write(
        agents_dir,
        filename,
        name=(
            filename.split("-", 1)[1].rsplit(".", 1)[0]
            if reason == "package"
            else "kirocrew-worker"
        ),
    )
    _seed_config()
    name = "atlas" if reason == "package" else "kirocrew-worker"
    resp = await api_agent_template_delete(_request("DELETE", name))
    assert resp.status == 409
    body = await _body(resp)
    assert body["code"] == "template_read_only" and body["reason"] == reason
    assert (agents_dir / filename).exists()


@pytest.mark.asyncio
async def test_delete_removes_an_unreferenced_template(agents_dir):
    _write(agents_dir, "reviewer.json")
    _seed_config({"pr-bot": "kirocrew"}, default="pr-bot")
    agent_state.set_model_managed("reviewer", True)
    resp = await api_agent_template_delete(_request("DELETE", "reviewer"))
    assert resp.status == 200, resp.text
    assert not (agents_dir / "reviewer.json").exists()
    assert agent_state.get_model_managed("reviewer") is None


@pytest.mark.asyncio
async def test_delete_unknown_is_404(agents_dir):
    _seed_config()
    assert (await api_agent_template_delete(_request("DELETE", "ghost"))).status == 404


# ── PATCH: the definition keys ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_writes_the_definition_keys_on_an_owned_template(agents_dir):
    _write(agents_dir, "reviewer.json", prompt="old", description="old")
    _seed_config()
    resp = await api_agent_detail(
        _request(
            "PATCH",
            "reviewer",
            {
                "prompt": "Review carefully.",
                "description": "Careful reviewer",
                "tools": ["fs_read", "grep", "@docs/search"],
                # An MCP ref survives; a builtin (`fs_read`) would be withheld by
                # the governance sanitizer, whose floor the ceiling may speak to.
                "allowedTools": ["@docs/search", "fs_read"],
            },
        )
    )
    assert resp.status == 200, resp.text
    spec = json.loads((agents_dir / "reviewer.json").read_text(encoding="utf-8"))
    assert spec["prompt"] == "Review carefully."
    assert spec["description"] == "Careful reviewer"
    assert spec["tools"] == ["fs_read", "grep", "@docs/search"]
    assert spec["allowedTools"] == ["@docs/search"]


@pytest.mark.asyncio
async def test_patch_definition_is_refused_on_a_package_template(agents_dir):
    _write(agents_dir, "SomePkg-atlas.json", name="atlas", prompt="shipped")
    _seed_config()
    resp = await api_agent_detail(_request("PATCH", "atlas", {"prompt": "mine now"}))
    assert resp.status == 409
    assert (await _body(resp))["code"] == "template_read_only"
    spec = json.loads((agents_dir / "SomePkg-atlas.json").read_text(encoding="utf-8"))
    assert spec["prompt"] == "shipped"


@pytest.mark.asyncio
async def test_patch_definition_classifies_the_targeted_file_not_its_name(agents_dir):
    """``atlas.json`` beside ``SomePkg-atlas.json``: the roster keeps only the
    package twin, so a lookup by declared name would answer for the wrong file.
    The plain file is the user's and stays editable; the package file stays
    refused, whichever the name-first lookup happens to land on."""
    _write(agents_dir, "atlas.json", name="atlas", prompt="mine")
    _write(agents_dir, "SomePkg-atlas.json", name="atlas", prompt="shipped")
    _seed_config()
    from kiro_crew.dashboard.handlers.agent_templates import read_only_reason_for_path

    assert read_only_reason_for_path(agents_dir / "atlas.json") is None
    assert read_only_reason_for_path(agents_dir / "SomePkg-atlas.json") == "package"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"prompt": 12},
        {"tools": "fs_read"},
        {"tools": ["", "x"]},
        {"tools": ["x" * 257]},
        {"description": "x" * 2001},
    ],
)
async def test_patch_definition_shape_is_validated(agents_dir, body):
    _write(agents_dir, "reviewer.json", prompt="old")
    _seed_config()
    resp = await api_agent_detail(_request("PATCH", "reviewer", body))
    assert resp.status == 400
    assert (await _body(resp))["code"] == "invalid_definition"
