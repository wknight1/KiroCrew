"""Does `memory.recall` actually fire on a real dashboard turn?

Every other suite for this point drives it directly or through the store. None of them
answers the question that decides whether the feature exists: a turn goes through
`ContextBuilder.build_message` -> `build_session_context` -> `MemoryStore.get_context`
-> `VectorMemoryStore.get_episodic_context`, and a gate anywhere on that chain makes the
point unreachable no matter how correct it is in isolation.

So this drives the WHOLE chain, with the real gate (a real keystone, a real config
snapshot, the real consent and scope reads) and only the HTTP transport faked. The
assertion is the one an operator would make: did a `memory.recall` row land in the
decision log.

It is deliberately the least mocked suite in this feature. A unit test that passes while
this fails describes a function nobody calls.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiro_crew import session_surface
from kiro_crew.context import ContextBuilder
from kiro_crew.decisions import log as _log
from kiro_crew.learn import LessonStore
from kiro_crew.memory import MemoryStore
from kiro_crew.skills import SkillsLoader
from kiro_crew.vector_memory import VectorMemoryStore

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
SESSION = "chat-reachability"
QUERY = "where do we deploy the signer"

#: Stored vectors are unit vectors, so their dot product with this is their cosine.
_Q = [1.0, 0.0, 0.0, 0.0]


def _unit(cos: float) -> list[float]:
    return [cos, math.sqrt(max(0.0, 1.0 - cos * cos)), 0.0, 0.0]


def _seed(store: VectorMemoryStore) -> None:
    """Three episodes close enough in wording to clear the relevance gate."""
    ts = datetime.now(timezone.utc).isoformat()
    for mem_id, text in (
        ("mem-one", "the deploy target for the signer is us-west-2"),
        ("mem-two", "an unrelated aside about lunch"),
        ("mem-three", "the signer rollback runbook lives in the ops repo"),
    ):
        vec = _unit(0.95)
        store.db.execute(
            "INSERT INTO episodic_memories (id, conversation_id, text, tags, embedding,"
            " importance, created_at, last_accessed_at, is_deleted)"
            " VALUES (?, '', ?, '[]', ?, 0.5, ?, ?, 0)",
            (mem_id, text, struct.pack(f"{len(vec)}f", *vec), ts, ts),
        )
    store.db.commit()


class _Oracle:
    """Answers every question `keep`, so a row lands whenever the point is reached.

    Only the TRANSPORT is faked. Consent, the scope, the bucket and the scrub are the
    real gate, because "is this reachable" is a question about those too.
    """

    asked: list = []

    def __init__(self, _provider):
        pass

    async def ask(self, state, questions):
        from kiro_crew.decisions.types import Answer

        _Oracle.asked.append(state)
        return {q.id: Answer(id=q.id, value="keep", p=0.9) for q in questions}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A consented keystone, a live config, a seeded store, and a surfaced session."""
    _Oracle.asked = []

    keystone = tmp_path / "decisions_consent.json"
    keystone.write_text(
        json.dumps(
            {
                "enabled": True,
                "endpoint": ENDPOINT,
                "history_budget_chars": 0,
                "tool_args": False,
                "memory_text": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("kiro_crew.config.loader.decisions_consent_path", lambda: keystone)
    monkeypatch.setattr(_log, "log_dir", lambda: tmp_path / "decisions")

    provider = SimpleNamespace(endpoint=ENDPOINT, model="jev-1", timeout_ms=5000, api_key="")
    snapshot = SimpleNamespace(
        decisions=SimpleNamespace(bucket=100, provider=provider, history_budget_chars=0)
    )
    monkeypatch.setattr("kiro_crew.decisions.gate._snapshot", lambda: snapshot)
    monkeypatch.setattr("kiro_crew.decisions.impl_jev.JevOracle", _Oracle)
    monkeypatch.setattr(
        "kiro_crew.decisions.capability.is_decisions_denied", lambda *_a, **_kw: False
    )

    store = VectorMemoryStore(db_path=tmp_path / "mem.db")
    store.init()
    _seed(store)
    # The embedder the real path would use; pinned so the query vector is the one the
    # seeded rows were built against rather than a model download.
    store.embed_fn = lambda _text: list(_Q)

    memory = MemoryStore(workspace=tmp_path / "ws")
    memory._vector_store = store

    session_surface.set_dashboard_surfaced({SESSION})
    yield SimpleNamespace(memory=memory, store=store, log_dir=tmp_path / "decisions")
    session_surface.set_dashboard_surfaced(set())


def _rows(log_dir: Path) -> list[dict]:
    if not log_dir.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(log_dir.glob("decisions-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


async def _drive(wired, tmp_path, monkeypatch):
    """One fresh dashboard turn, assembled the way production assembles one.

    `ContextBuilder` is constructed INSIDE the loop, because that is where it captures
    the loop the point submits its `decide` to; and `build_message` is called from a
    worker THREAD, because production reaches it only through `run_in_embed_pool` and
    the point refuses to block a thread that is running a loop.
    """
    builder = ContextBuilder(
        memory=wired.memory,
        skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        lessons=LessonStore(base_dir=tmp_path),
    )
    monkeypatch.setattr(
        ContextBuilder, "get_memory_for", lambda *_a, **_kw: wired.memory, raising=True
    )
    return await asyncio.to_thread(builder.build_message, QUERY, True, SESSION)


#: Why the four assertions below are expected failures rather than deletions or a fix.
#:
#: They are the evidence, in executable form: the claim "this feature works" is false on
#: the main path today, and a suite that quietly omitted the proof would leave the next
#: reader to rediscover it.
UNREACHABLE = (
    "memory.recall is UNREACHABLE on the main path: prompt assembly calls "
    "MemoryStore.get_context with include_activity=False and that flag also gates "
    "episodic retrieval, so get_episodic_context is never called and the keep hook "
    "never runs. The flag is deliberate (#11996 stopped fresh sessions "
    "auto-injecting recalled material), so making retrieval unconditional is a "
    "PRODUCT decision about whether recalled memory belongs in every fresh "
    "session's prompt, not a wiring fix -- escalated rather than taken here. "
    "strict=True so this flips RED the moment reachability is fixed and the "
    "xfail has to come off."
)


class TestTheHookIsReachedOnARealTurn:
    @pytest.mark.xfail(strict=True, reason=UNREACHABLE)
    @pytest.mark.asyncio
    async def test_a_fresh_dashboard_turn_reaches_the_point(self, wired, tmp_path, monkeypatch):
        """The claim this whole suite exists for.

        If the episodic block is not built on this path, the hook is never called, no
        question is asked, and no row is written -- and every other test in this feature
        is describing a function production never runs.
        """
        await _drive(wired, tmp_path, monkeypatch)
        rows = _rows(wired.log_dir)
        points = [row.get("point") for row in rows]
        assert "memory.recall" in points, (
            "no memory.recall row: the point was never reached on a real dashboard turn. "
            f"rows={rows}"
        )

    @pytest.mark.xfail(strict=True, reason=UNREACHABLE)
    @pytest.mark.asyncio
    async def test_the_question_carried_the_recalled_candidates(self, wired, tmp_path, monkeypatch):
        """Reached AND asked about the right thing, not merely reached."""
        await _drive(wired, tmp_path, monkeypatch)
        assert _Oracle.asked, "the oracle was never asked"
        state = _Oracle.asked[0]
        keys = {row["key"] for row in state["candidates"]}
        assert keys, "the question carried no candidates"
        assert keys <= {"mem-one", "mem-two", "mem-three"}

    @pytest.mark.xfail(strict=True, reason=UNREACHABLE)
    @pytest.mark.asyncio
    async def test_the_kept_memories_are_in_the_assembled_prompt(
        self, wired, tmp_path, monkeypatch
    ):
        """And the answer reaches the PROMPT, which is the point of the feature."""
        message, _meta = await _drive(wired, tmp_path, monkeypatch)
        assert "us-west-2" in message, "the kept memory is not in the assembled prompt"

    @pytest.mark.asyncio
    async def test_an_unsurfaced_session_is_not_decided_for(self, wired, tmp_path, monkeypatch):
        """The same turn on a session nobody has open asks nothing.

        Here so a future change that makes the block unconditional cannot also make the
        egress unconditional: the two are separate gates and this pins the second.
        """
        session_surface.set_dashboard_surfaced(set())
        await _drive(wired, tmp_path, monkeypatch)
        assert _Oracle.asked == []
        assert [r for r in _rows(wired.log_dir) if r.get("point") == "memory.recall"] == []


class TestTheBlockIsBuiltForAnExplicitReader:
    """The retrieval itself, held apart from the decision.

    `get_context` served the episodic block only to a caller that asked for ACTIVITY,
    and prompt assembly does not: it passes ``include_activity=False`` to read
    preferences complete without a history or search pass. So the block -- and the hook
    on it -- was unreachable from the one path that matters, and this is the pair of
    assertions that says which flag decides it.
    """

    def test_an_explicit_reader_gets_the_block(self, wired):
        assert "us-west-2" in wired.memory.get_context(query=QUERY, include_activity=True)

    @pytest.mark.xfail(strict=True, reason=UNREACHABLE)
    def test_prompt_assembly_gets_it_too(self, wired):
        """The flag that decides it, isolated from the decision seam entirely.

        `include_activity` reads as "include the activity SECTIONS" -- history, projects,
        the activity index -- and episodic recall is not one of them, so gating recall on
        it is at best a surprise. But the gate is deliberate rather than accidental, which
        is why this is escalated and not patched: see :data:`UNREACHABLE`.
        """
        assert "us-west-2" in wired.memory.get_context(query=QUERY, include_activity=False)

    def test_no_query_still_means_no_recall(self, wired):
        """With no message there is nothing to judge relevance against."""
        assert "us-west-2" not in wired.memory.get_context(query="", include_activity=False)
