"""``memory.recall`` applied live: the kept set is what reaches the prompt.

The point's own suite drives the decision. This one drives the WIRING -- a real
``VectorMemoryStore`` with real rows, a real ``MemoryStore.get_context`` and a real
``ContextBuilder``, so the claim is about the block that actually gets injected
rather than about a list a helper returned.

Two directions, and both matter. With the switch on, the memories Jev kept are the
ones in the block and the dropped one is gone. With the switch off -- the shipped
default -- the block is byte-identical to the one the same tree produced before
this point existed, which is the property that makes the seam safe to place on the
prompt-assembly path.
"""

from __future__ import annotations

import asyncio
import math
import struct
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kiro_crew import context as ctx_mod
from kiro_crew.decisions.points import memory_recall as mr
from kiro_crew.decisions.types import Answer
from kiro_crew.vector_memory import VectorMemoryStore

# Stored vectors are unit vectors, so their dot product with this is exactly their
# cosine similarity (the search path normalises the query).
_Q = [1.0, 0.0, 0.0, 0.0]


def _unit_at_cosine(cos: float) -> list[float]:
    return [cos, math.sqrt(max(0.0, 1.0 - cos * cos)), 0.0, 0.0]


def _store(tmp_path: Path) -> VectorMemoryStore:
    store = VectorMemoryStore(db_path=tmp_path / "mem.db")
    store.init()
    return store


def _insert(store: VectorMemoryStore, mem_id: str, text: str, cosine: float = 0.95) -> None:
    vec = _unit_at_cosine(cosine)
    ts = (datetime.now(timezone.utc) - timedelta(days=0)).isoformat()
    store.db.execute(
        "INSERT INTO episodic_memories "
        "(id, conversation_id, text, tags, embedding, importance, "
        "created_at, last_accessed_at, is_deleted) "
        "VALUES (?, '', ?, '[]', ?, 0.5, ?, ?, 0)",
        (mem_id, text, struct.pack(f"{len(vec)}f", *vec), ts, ts),
    )
    store.db.commit()


@pytest.fixture
def seeded(tmp_path):
    """Three relevant episodes, each recognisable in the injected block."""
    store = _store(tmp_path)
    _insert(store, "keep-one", "KEEPONE the deploy target is us-west-2")
    _insert(store, "drop-two", "DROPTWO an unrelated aside")
    _insert(store, "keep-three", "KEEPTHREE the rollback runbook lives in ops")
    return store


@pytest.fixture
def bg_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    loop.call_soon(ready.set)
    thread = threading.Thread(target=loop.run_forever, name="memory-apply-loop", daemon=True)
    thread.start()
    try:
        assert ready.wait(10), "background loop did not start"
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive(), "background loop did not stop"
        loop.close()


def _answering(monkeypatch, verdict_of):
    """Patch ``core.decide`` so each candidate is answered by *verdict_of(snippet)*."""

    async def _decide(_point, state, questions, **_kw):
        answers = {}
        for index, question in enumerate(questions):
            snippet = state["candidates"][index]["snippet"]
            answers[question.id] = Answer(id=question.id, value=verdict_of(snippet), p=0.9)
        return answers

    monkeypatch.setattr(mr.core, "decide", _decide)


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    from kiro_crew.decisions import log as _log

    monkeypatch.setattr(_log, "log_dir", lambda: tmp_path / "decisions")
    monkeypatch.setattr(mr.core, "is_enabled", lambda *a, **k: True)
    monkeypatch.setattr(mr.core, "timeout_secs", lambda *a, **k: 1.0)


class TestTheKeptSetIsWhatIsInjected:
    def test_a_dropped_memory_is_not_in_the_block(self, seeded, bg_loop, enabled, monkeypatch):
        _answering(monkeypatch, lambda snippet: "drop" if "DROPTWO" in snippet else "keep")
        hook = mr.keep_hook("where do we deploy", session_key="s", loop=bg_loop, owner_turn=True)
        block = seeded.get_episodic_context(query_embedding=_Q, keep=hook)
        assert "KEEPONE" in block
        assert "KEEPTHREE" in block
        assert "DROPTWO" not in block

    def test_keeping_nothing_injects_no_block_at_all(self, seeded, bg_loop, enabled, monkeypatch):
        """An empty kept set is an empty block, not a header with no rows under it."""
        _answering(monkeypatch, lambda _snippet: "drop")
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        assert seeded.get_episodic_context(query_embedding=_Q, keep=hook) == ""

    def test_the_block_keeps_the_rankers_order(self, seeded, bg_loop, enabled, monkeypatch):
        """Jev answers keep/drop, so the surviving rows stay in similarity order."""
        _answering(monkeypatch, lambda _snippet: "keep")
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        decided = seeded.get_episodic_context(query_embedding=_Q, keep=hook)
        baseline = seeded.get_episodic_context(query_embedding=_Q)
        assert decided == baseline


class TestTheCapIsAppliedBeforeTheHook:
    """A hook may only narrow the block, so it must not see rows the cap excluded.

    The ordering IS the property. With the cap applied after the hook, dropping a
    high-ranked memory frees budget a lower-ranked one then fits into -- so a memory
    the block would never have carried enters the prompt because Jev removed a
    different one. That is the hook ADDING a memory, which is the one thing it is
    documented not to be able to do.
    """

    def _cap_for_two(self, store):
        """A cap that admits exactly the first two of three seeded episodes."""
        rows = store.search_episodic(query_embedding=_Q, relevance_filter=True)
        assert len(rows) == 3, [r["id"] for r in rows]
        lines = [store._episode_line(i, r) for i, r in enumerate(rows, 1)]
        return len(lines[0]) + 1 + len(lines[1])

    def test_a_cap_excluded_memory_cannot_enter_when_jev_drops_another(
        self, seeded, bg_loop, enabled, monkeypatch
    ):
        cap = self._cap_for_two(seeded)
        # Baseline at this cap: the first two rows, and the third is out.
        baseline = seeded.get_episodic_context(query_embedding=_Q, cap=cap)
        admitted = [name for name in ("KEEPONE", "DROPTWO", "KEEPTHREE") if name in baseline]
        assert len(admitted) == 2, baseline

        # Jev drops the FIRST admitted row. The freed budget must not let the
        # cap-excluded third row in.
        _answering(monkeypatch, lambda snippet: "drop" if admitted[0] in snippet else "keep")
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        decided = seeded.get_episodic_context(query_embedding=_Q, cap=cap, keep=hook)
        assert admitted[0] not in decided
        assert admitted[1] in decided
        excluded = next(
            name for name in ("KEEPONE", "DROPTWO", "KEEPTHREE") if name not in admitted
        )
        assert excluded not in decided, "the cap excluded it, so no answer may admit it"

    def test_the_hook_is_only_asked_about_cap_admitted_rows(
        self, seeded, bg_loop, enabled, monkeypatch
    ):
        """The menu it is offered is the block, not the search."""
        cap = self._cap_for_two(seeded)
        seen: list[int] = []

        def _keep(candidates):
            seen.append(len(candidates))
            return None

        seeded.get_episodic_context(query_embedding=_Q, cap=cap, keep=_keep)
        assert seen == [2], "three rows were found; two fit the block"

    def test_a_generous_cap_offers_every_ranked_row(self, seeded):
        seen: list[int] = []

        def _keep(candidates):
            seen.append(len(candidates))
            return None

        seeded.get_episodic_context(query_embedding=_Q, cap=100_000, keep=_keep)
        assert seen == [3]


class TestTheBlockKeepsTheRanks:
    """A kept row is rendered under the rank it was MEASURED under.

    Two things break with a fresh 1..n counter. The cap walk measures
    `_episode_line(rank, row)`, so emitting a different number makes the cap a bound on a
    string nobody rendered. And the admitted set can have GAPS on the v2 lineage, where
    an over-budget row is skipped and the walk continues -- so renumbering tells the
    model the third line is the third-best memory when it is the fourth-ranked one.
    """

    def test_dropping_a_row_does_not_renumber_the_survivors(
        self, seeded, bg_loop, enabled, monkeypatch
    ):
        _answering(monkeypatch, lambda snippet: "drop" if "KEEPONE" in snippet else "keep")
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        block = seeded.get_episodic_context(query_embedding=_Q, keep=hook)
        # The first-ranked row is gone, so the block opens at rank 2 rather than
        # relabelling the second-ranked row as 1.
        assert "KEEPONE" not in block
        assert "1. " not in block, block
        assert "2. " in block

    def test_keeping_everything_numbers_from_one(self, seeded, bg_loop, enabled, monkeypatch):
        """The ranks ARE 1..n when nothing was dropped, so this is not an off-by-one."""
        _answering(monkeypatch, lambda _snippet: "keep")
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        assert seeded.get_episodic_context(query_embedding=_Q, keep=hook).count("1. ") == 1

    def test_the_rank_travels_with_the_row_out_of_the_cap_walk(self, seeded):
        """The walk's own contract, held directly: (rank, row) and not position."""
        rows = seeded.search_episodic(query_embedding=_Q, relevance_filter=True)
        admitted = seeded._cap_admitted_episodes(rows, 100_000)
        assert [index for index, _row in admitted] == [1, 2, 3]
        assert [id(row) for _index, row in admitted] == [id(r) for r in rows]


class TestTheSwitchOffChangesNothing:
    def test_a_disabled_point_injects_the_similarity_block_unchanged(
        self, seeded, bg_loop, monkeypatch
    ):
        monkeypatch.setattr(mr.core, "is_enabled", lambda *a, **k: False)
        hook = mr.keep_hook("anything", session_key="s", loop=bg_loop, owner_turn=True)
        assert seeded.get_episodic_context(query_embedding=_Q, keep=hook) == (
            seeded.get_episodic_context(query_embedding=_Q)
        )

    def test_no_hook_at_all_injects_the_similarity_block_unchanged(self, seeded):
        """The shipped default reaches this method with ``keep=None``."""
        assert seeded.get_episodic_context(query_embedding=_Q, keep=None) == (
            seeded.get_episodic_context(query_embedding=_Q)
        )


class TestTheHookNeedsALiveDashboardSurface:
    """The restriction is the CALL SITE's, so it is asserted at the call site.

    A cron turn, a sub-agent, an integration and any session whose tab is closed
    have no reader for the receipt to reach, and the request carries snippets of the
    member's own recalled memories -- so the hook is not built for them at all and
    nothing is sent.

    `has_dashboard_surface` is OBSERVED state, so the parametrised keys below are
    refused because nothing published them, NOT because of how they are spelled --
    which is why one of them is a channel key that IS accepted once published. That
    is the intended reading of the gate: the owner is reading that conversation in
    the dashboard.
    """

    def _builder(self, monkeypatch, surfaced: set[str]):
        monkeypatch.setattr(ctx_mod, "has_dashboard_surface", lambda key: key in surfaced)
        return ctx_mod.ContextBuilder()

    def test_a_dashboard_session_gets_a_hook(self, monkeypatch):
        builder = self._builder(monkeypatch, {"chat-1"})
        assert builder._episodic_keep_hook("chat-1", "a message") is not None

    @pytest.mark.parametrize("key", ["cron:nightly", "subagent:abc", "slack:C1", "_bg"])
    def test_an_unsurfaced_session_gets_no_hook(self, monkeypatch, key):
        builder = self._builder(monkeypatch, {"chat-1"})
        assert builder._episodic_keep_hook(key, "a message") is None

    def test_a_channel_session_with_an_open_tab_gets_one(self, monkeypatch):
        """The gate reads the PUBLISHED set, not the key's prefix.

        `chat_utils._sync_dashboard_slots` publishes each open slot's own key and a
        channel-born slot contributes its channel key, so this is the state a Slack
        conversation with its dashboard tab open is really in. Asserted so the
        documented behaviour and the code cannot drift into two claims.
        """
        builder = self._builder(monkeypatch, {"slack:C1"})
        assert builder._episodic_keep_hook("slack:C1", "a message") is not None

    def test_no_session_key_gets_no_hook(self, monkeypatch):
        builder = self._builder(monkeypatch, {"chat-1"})
        assert builder._episodic_keep_hook(None, "a message") is None

    def test_no_query_gets_no_hook(self, monkeypatch):
        """With no message there is nothing to judge relevance against."""
        builder = self._builder(monkeypatch, {"chat-1"})
        assert builder._episodic_keep_hook("chat-1", "") is None


class TestMemoryStoreCarriesTheHook:
    def test_get_context_hands_the_hook_to_the_store(self, tmp_path, monkeypatch):
        """``MemoryStore`` only CARRIES it; every fallback is the store's."""
        from kiro_crew.memory import MemoryStore

        seen: list[object] = []

        class _Vectors:
            def get_semantic_context(self, **_kw):
                return ""

            def get_preferences_context(self):
                return ""

            def get_episodic_context(self, *, query_text="", cap=0, keep=None):
                seen.append(keep)
                return ""

        store = MemoryStore(workspace=tmp_path)
        store._vector_store = _Vectors()
        sentinel = object()
        store.get_context(query="hi", include_activity=True, episodic_keep=sentinel)
        assert seen == [sentinel]

    def test_a_caller_without_a_hook_passes_none(self, tmp_path):
        from kiro_crew.memory import MemoryStore

        seen: list[object] = []

        class _Vectors:
            def get_semantic_context(self, **_kw):
                return ""

            def get_preferences_context(self):
                return ""

            def get_episodic_context(self, *, query_text="", cap=0, keep=None):
                seen.append(keep)
                return ""

        store = MemoryStore(workspace=tmp_path)
        store._vector_store = _Vectors()
        store.get_context(query="hi", include_activity=True)
        assert seen == [None]
