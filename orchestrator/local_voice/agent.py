"""The agent behind the voice: Pi over RPC, one child per space, fronted by a Pipecat "LLM" service.

AgentHub is process-wide: the space pool (05c's SpacePool), the active space and mode, one turn at a time per Pi
child, and what /v1/status shows. PiAgentService is one per pipeline (per connection).

How a turn maps onto Pipecat 1.12.0 and Pi 0.99.1 (notes 04c §1, 05c, 05d):
- The user aggregator pushes an LLMContextFrame holding the whole Pipecat context; only user messages not sent
  before go to Pi (tracked by object identity), so a re-pushed context never re-prompts it.
- The Pi turn runs in its own task, not the frame-processing task. Pipecat cancels the processing task on every
  InterruptionFrame; a separate task lets the service decide: barge-in cancels the turn (the bridge then aborts Pi in
  the background, never inline, because Pipecat waits at most 1 s), while the person answering a spoken permission
  question must not abort the run that is waiting for that answer.
- Text deltas become LLMTextFrames; thinking deltas are never spoken (05d), but llama-swap's loading banner, which
  arrives as thinking, is a cue to say the model is waking up.
- A tool call is acknowledged at toolcall_start with a canned phrase spoken by the orchestrator (append_to_context
  False, so it stays out of the record) and shown to clients as a `tool` message.
- What the person heard of an interrupted reply (the assistant aggregator's spoken sentences, refined by the
  client's played_ms) prefixes the next prompt; Pi itself drops the aborted partial reply (05c E3a).
- A barge-in while the model is still writing its reply stops the speech but not the run (when no tool call is
  pending and the reply is short): the run finishes silently and stays in Pi's context, and the next prompt waits for
  it. An aborted run leaves ds4 a request that no longer extends its recurrent state, and ds4 then replays the
  conversation from its nearest checkpoint: 9.0 s at 14k tokens and 27.6 s at 33k on 2026-10-05.
- The hold gate is asked right before every prompt; while it is draining or held the text goes to the hold
  coordinator, which answers with the pre-rendered notice and runs the turn when the gate opens.
- Every user turn goes to the brain's turn log when it is over (brain/TURN_LOG.md §1), and after the first completed
  turn of a new session the brain's pending spoken digests are said at the first idle moment (§2).
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from . import approvals
from .approvals import KB_READ_VERBS, Approval, Grants
from .config import Config
from .digests import DigestOutbox
from .hold import HoldMonitor
from .pi_rpc import (ConfirmRequest, Exited, InputRequest, MessageEnd, Notify, PiBusyError, PiChild, PiError,
                     SelectRequest, Settled, SpaceConfig, SpacePool, Status, TextDelta, ThinkingDelta, ToolCallStarted,
                     ToolEnd, ToolStart)
from .pi_rpc import atlas_capture_command, heredoc_body
from .protocol_v1 import ClientMessageFrame, ReplyTimeline, in_order, urgent
from .router import answer_to_ask, musing, route
from .services import store_settings
from .spaces import MODES, READ_TOOLS, SPACE_SWITCH, Spaces, pi_space_config
from .speech_text import LOADING_BANNER, yes_no
from .spoken_cap import SpokenCap, wants_more, words
from .tone_hook import analyze as tone_analyze
from .turnlog import TurnEntry, TurnLog, now_local

UIEvent = Callable[[dict], Frame]


@dataclass
class TurnRecord:
    """What /v1/status shows about the last turn (PROTOCOL.md "Status endpoint")."""
    eos_to_first_audio_ms: float = 0.0
    stt_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    tools: list[str] = field(default_factory=list)
    space: str = ""
    at: float = 0.0


class AgentHub:
    """Process-wide agent state shared by every connection."""

    def __init__(self, cfg: Config, spaces: Spaces, *, state_dir: Path, agent_dir: Path | None, hold: HoldMonitor,
                 extra_env: dict[str, str] | None = None):
        self.cfg = cfg
        self.spaces = spaces
        self.hold = hold
        self.state_dir = state_dir
        self.configs: dict[str, SpaceConfig] = {
            name: pi_space_config(cfg, sp, agent_dir=agent_dir, state_dir=state_dir, extra_env=extra_env,
                                  others=list(spaces.spaces.values()))
            for name, sp in spaces.spaces.items()}
        (state_dir / "pi-logs").mkdir(parents=True, exist_ok=True)
        self.pool = SpacePool(self.configs, transcript_dir=state_dir / "pi-logs")
        for name in self.configs:
            self.configs[name].session_dir.mkdir(parents=True, exist_ok=True)
        self.active = cfg.default_space
        self.mode = spaces.default_mode
        self.locks: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in self.configs}
        self.state = "idle"
        self.tool: dict[str, Any] | None = None
        self.last_turn = TurnRecord()
        self.turns = 0
        self.journal = False           # "just take notes": every utterance in a journal space is saved verbatim
        self._child_mode: dict[int, str] = {}   # id(child) -> the voice mode it was last switched to
        self.approvals = approvals.settings(cfg)   # agent.approvals: waits, spoken lines, card clients
        self.grants = Grants()                     # "allow for this session" answers, per voice session and space
        self.approval: dict[str, Any] | None = None   # the question waiting for an answer, for /v1/status
        # When this process's last Pi run ended (monotonic). ds4-server answers a short continuation in about 0.2 s
        # right after a request but 1-2.5 s after a few seconds idle (its GPU mappings are dropped;
        # measured 2026-10-05), so every latency row says how long the LLM had been idle. Other clients of the rig may have
        # used it in between: this is an upper bound on its idle time.
        self.llm_last_active: float | None = None

    async def child(self, space: str | None = None) -> PiChild:
        child = await self.pool.get(space or self.active)
        if self._child_mode.get(id(child), "conversation") != self.mode:
            # a child started (or restarted) since the last mode switch begins in conversation mode (voice_mode.ts)
            await child.extension_command(f"/voice-mode {self.mode}")
            self._child_mode[id(child)] = self.mode
        return child

    async def enter(self, name: str) -> PiChild:
        """Make `name` the active space, starting its Pi child if it is not running (SPACES.md: lazily, on first
        entry). Raises KeyError for an unknown space, PiError/OSError when it cannot start."""
        sp = self.spaces[name]
        if not sp.root.is_dir():
            raise PiError(f"{name}'s root {sp.root} is missing")
        child = await self.child(name)
        self.active = name
        if not self.space_is_journal(name):
            self.journal = False
        return child

    async def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        await self.child(self.active)

    def space_is_journal(self, name: str | None = None) -> bool:
        """A space whose commands include Atlas's verbatim capture: the orchestrator saves musings there itself."""
        sp = self.spaces[name or self.active]
        return any(p[:3] == ["python3", "atlas.py", "capture"] for p in sp.bash_allow)

    @property
    def space(self):
        return self.spaces[self.active]

    async def close(self) -> None:
        await self.pool.close()

    def status(self) -> dict[str, Any]:
        sp = self.space
        return {"space": self.active, "mode": self.mode, "tier": sp.tier, "model": sp.model, "state": self.state,
                "tool": self.tool, "turns": self.turns,
                "children": {n: {"alive": c.alive, "busy": c.busy} for n, c in self.pool.children.items()},
                "last_turn": {k: round(v, 1) if isinstance(v, float) else v
                              for k, v in self.last_turn.__dict__.items() if k != "at"},
                "approval": self.approval, "grants": self.grants.as_json()}


class FlushSpeechFrame(DataFrame):
    """Pushed at a tool call: the code-fence splitter (pipeline.InOrderTextProcessor) passes on the sentence it holds
    for lookahead, so the model's words before the call are said while the tool runs, not after its answer (the kb
    answer's "Let me search the knowledge base." came 4.0 s after the question in the e2e of 16:57, its first token
    0.4 s, once no canned acknowledgement flushed it)."""


@dataclass
class _Confirm:
    """The question waiting for an answer (PROTOCOL.md "Approvals")."""
    id: str
    child: PiChild
    approval: Approval
    space: str
    question: str                    # as said aloud
    card: bool                       # this connection's client shows it as a card
    asked_at: float                  # monotonic
    clock: asyncio.Task | None = None


@dataclass
class _TurnState:
    """What a barge-in needs to know about the run in progress."""
    prompted: bool = False       # the prompt went to Pi (before that, cancelling costs nothing)
    tool_pending: bool = False   # between toolcall_start and tool_execution_end
    chars: int = 0               # reply text so far
    muted: bool = False          # barged in: finishing silently
    declined: bool = False       # a question was answered no, by silence or with other words: that line was said


class PiAgentService(LLMService):
    """Streams the active space's Pi agent into the pipeline as if it were an LLM."""

    def __init__(self, *, hub: AgentHub, ui_event: UIEvent | None = urgent, ui_in_order: UIEvent | None = in_order,
                 timeline: ReplyTimeline | None = None,
                 on_held_text: Callable[[str, str, dict], Awaitable[None]] | None = None,
                 on_pi_waiting: Callable[[str], Awaitable[None]] | None = None,
                 session_id: str = "", client: str = "", turn_log: TurnLog | None = None,
                 digests: DigestOutbox | None = None, new_session: bool = True, digest_idle_s: float = 1.0,
                 **kwargs):
        """ui_event wraps a protocol v1 message for clients (urgent frames for native clients, RTVI server messages
        for the browser, None for none); ui_in_order sends one behind the audio already queued (protocol v1 only).
        session_id and client go into the brain's turn log; digests is the brain's spoken-digest outbox."""
        kwargs.setdefault("settings", store_settings(LLMSettings(model=None)))
        super().__init__(**kwargs)
        self._hub = hub
        self._cfg = hub.cfg
        self._ui_event = ui_event
        self._ui_in_order = ui_in_order
        self._timeline = timeline
        self.on_held_text = on_held_text
        self.on_pi_waiting = on_pi_waiting
        self._session_id = session_id
        self._client = client
        self._turn_log = turn_log
        self._digests = digests
        self._new_session = new_session
        self._digest_idle_s = digest_idle_s
        self._seen: dict[int, Any] = {}        # context messages already handled, by id(); values keep them alive
        self._turn_task: asyncio.Task | None = None
        self._turn_state: _TurnState | None = None
        self._muted_task: asyncio.Task | None = None   # a barged-in run finishing silently
        self._muted_state: _TurnState | None = None
        self._child: PiChild | None = None      # the child of the turn now running (or last run)
        self._response_open = False
        self._confirm: _Confirm | None = None
        self._confirm_queue: list[tuple[Approval, PiChild, str]] = []   # parallel tool calls: one question at a time
        self._skip_heard_once = False
        self._heard_from_client: str | None = None   # reply id whose heard text came from played_ms
        self._speech_started_at: datetime | None = None
        self._entry: TurnEntry | None = None         # the turn being logged (written once its reply is over)
        self._bot_speaking = False
        self._user_speaking = False
        self._busy_since = time.monotonic()          # the last time anything happened (for the digest's idle wait)
        self._spoke = asyncio.Event()
        self._played = asyncio.Event()
        self._digest_task: asyncio.Task | None = None
        self.completed_turns = 0
        self.last_user_text = ""
        self.last_ttft_ms: float | None = None
        self.last_llm_idle_s: float | None = None
        self.last_settled_at: float | None = None   # monotonic time of the last run's agent_settled
        self.silent_runs = {"settled": 0, "aborted": 0}   # barged-in runs that finished silently, or were aborted
        self.turns_run = 0
        self._ask_spaces: list[str] | None = None    # "Do you mean A or B?" was asked: the candidate spaces
        self._note_next = False                       # "note this" came alone: the next utterance is the musing
        self.tone = None                              # tone_hook.make_tone_hook(cfg): None while tone is off
        self.utterance_audio: Callable[[], bytes] | None = None   # the turn's speech, from the STT (tone only)
        self._last_cut = False                        # the last reply was cut by the spoken cap and offered more
        self._user_stopped_mono: float | None = None  # when the person's last turn ended (monotonic)
        self._gap_before_speech: float | None = None  # from that end to the start of the turn now coming in
        self._captured_last = False                   # the last turn's words were saved into the journal by us
        # the last run: words written and said, the cap, whether it cut, and the acknowledgement (canned, the model's)
        self.last_reply: dict[str, Any] | None = None

    def can_generate_metrics(self) -> bool:
        return True

    # ------------------------------------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._on_context(frame)
            return
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._on_interruption(barge_in=True)
            self._played.set()
            return
        if isinstance(frame, ClientMessageFrame):
            await self._on_client_message(frame.message)
        elif isinstance(frame, RTVIClientMessageFrame):
            # the browser page's own messages (an approval card's button) come as RTVI client messages: {t, d}
            await self._on_client_message({**(frame.data if isinstance(frame.data, dict) else {}), "t": frame.type})
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            self._busy_since = time.monotonic()
            if direction == FrameDirection.DOWNSTREAM and self._speech_started_at is None:
                self._speech_started_at = now_local()
                stop = self._user_stopped_mono
                self._gap_before_speech = None if stop is None else time.monotonic() - stop
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._busy_since = time.monotonic()
            if direction == FrameDirection.DOWNSTREAM:
                self._user_stopped_mono = time.monotonic()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._busy_since = time.monotonic()
            self._spoke.set()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._busy_since = time.monotonic()
            self._played.set()
        await self.push_frame(frame, direction)

    def _new_user_text(self, context: LLMContext) -> tuple[str, dict]:
        """The user messages not seen before, and the first one's metadata (`lv`: how the words came in)."""
        parts, meta = [], None
        for msg in context.get_messages():
            if id(msg) in self._seen:
                continue
            self._seen[id(msg)] = msg
            if isinstance(msg, dict) and msg.get("role") == "user":
                if meta is None and isinstance(msg.get("lv"), dict):
                    meta = msg["lv"]
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content.strip())
                elif isinstance(content, list):
                    parts += [p.get("text", "").strip() for p in content if isinstance(p, dict)]
        return " ".join(p for p in parts if p), dict(meta or {})

    async def _on_context(self, frame: LLMContextFrame):
        if frame.speculation:
            return   # eager end-of-turn speculation is not configured; never prompt the agent speculatively
        text, meta = self._new_user_text(frame.context)
        started, self._speech_started_at = self._speech_started_at, None
        if not text:
            return
        if self._confirm is not None:
            await self._answer_by_voice(text)
            return
        if "input" not in meta:
            meta["input"] = "voice" if started is not None else "text"
        if "t_start" not in meta:
            meta["t_start"] = (started or now_local()).isoformat()
        self.last_user_text = text
        self.last_reply = self.last_ttft_ms = None   # a turn the router answers has no run
        gap, captured, self._captured_last = self._gap_before_speech, self._captured_last, False
        if captured and gap is not None and gap <= self._cfg.journal_join_s and self._hub.space_is_journal():
            # The turn end took a pause in the musing for its end: the person went on within moments of the words
            # just saved, so these words are the same musing ("...this autumn." / "Maybe a walk before breakfast.",
            # e2e 2026-10-05 16:57: the second part went to the model, which asked to run its own capture).
            logger.info(f"{self}: {gap:.2f} s after a saved musing: saving this utterance with it")
            self._note_next = True
        if await self._route(text, meta):
            return
        await self.run_text(text, meta)

    # ------------------------------------------------------------------------------------ the router (M3)

    async def _route(self, text: str, meta: dict) -> bool:
        """Space and mode switches and journal capture asked for in words (router.py), handled before any model
        call. True when the words were only that: nothing goes to the agent."""
        hub = self._hub
        if self._note_next:
            self._note_next = False
            meta["note_next"] = True     # this utterance is the musing the cue announced (_capture)
        if hub.space_is_journal() and musing(text) == "":
            # "Note this." on its own: Smart Turn ends the turn at the pause after the cue ("Note this, the garden
            # was quiet..." split in two in the e2e of 2026-10-05 16:18), and the model, prompted with the cue alone,
            # was then barged in on by the musing and captured it itself. The next utterance is the musing.
            self._note_next = True
            await self._reply_alone(text, meta, "Go ahead.")
            return True
        if self._ask_spaces is not None:
            candidates, self._ask_spaces = self._ask_spaces, None
            pick = answer_to_ask(text, candidates, hub.spaces)
            if pick is not None:
                await self._switch(pick, "", meta, text)
                return True
        r = route(text, hub.spaces)
        if r.kind == "switch":
            await self._switch(r.space, r.rest, meta, text)
            return True
        if r.kind == "ask":
            self._ask_spaces = r.candidates
            descs = [hub.spaces[c].description for c in r.candidates]
            await self._reply_alone(text, meta, f"Do you mean {' or '.join(descs)}?")
            return True
        if r.kind == "mode":
            await self._switch_mode(r.mode, text, meta)
            return True
        if r.kind in ("notes_on", "notes_off"):
            if not hub.space_is_journal():
                await self._reply_alone(text, meta, "I can only take notes in your journal. Say go to my journal first.")
                return True
            hub.journal = r.kind == "notes_on"
            await self._reply_alone(text, meta, "Taking notes. I'll save everything you say until you say stop taking notes."
                                    if hub.journal else "Stopped taking notes.")
            return True
        return False

    def is_command(self, text: str) -> bool:
        """Words the orchestrator answers itself and that are complete as said, so the turn need not wait for Smart
        Turn's silence fallback (turn_end.py): a space or mode switch with nothing after it, an answer to "Do you
        mean A or B?", "note this" alone in a journal space, and a bare yes or no to a permission question (only a
        bare one: "yes, but..." is left to Smart Turn, a permission must never be granted on half a sentence)."""
        hub = self._hub
        if self._confirm is not None:
            return approvals.is_clean_answer(text, self._confirm.approval.options)
        if self._ask_spaces is not None and answer_to_ask(text, self._ask_spaces, hub.spaces) is not None:
            return True
        if hub.space_is_journal() and musing(text) == "":
            return True
        r = route(text, hub.spaces)
        return r.kind in ("mode", "notes_on", "notes_off", "ask") or (r.kind == "switch" and not r.rest)

    async def _reply_alone(self, text: str | None, meta: dict, line: str) -> None:
        """A turn the orchestrator answers itself: the line spoken as its own reply, and logged as the turn."""
        await self._say(line)
        await self._close_response()
        if self._ui_in_order is not None:
            await self.push_frame(self._ui_in_order({"t": "state", "v": "listening"}))
        if text is not None and self._turn_log is not None:
            entry = self._begin_entry(text, meta, self._hub.active)
            if entry is not None:
                entry.reply_text, entry.t_end = line, now_local()
                self._turn_log.write(entry)

    async def _space_message(self) -> None:
        hub = self._hub
        sp = hub.space
        await self._ui({"t": "space", "name": hub.active, "mode": hub.mode, "tier": sp.tier,
                        "description": sp.description})

    async def _switch(self, name: str, rest: str, meta: dict, text: str | None) -> bool:
        """Enter a space (SPACES.md "Switching"): say so, tell the clients, then run the rest of the sentence there."""
        hub = self._hub
        if name not in hub.spaces:
            await self._ui({"t": "error", "code": "space_unknown", "message": f"no space {name!r}"})
            return False
        desc = hub.spaces[name].description
        try:
            await hub.enter(name)
        except (PiError, OSError) as e:
            logger.error(f"{self}: cannot enter {name}: {e}")
            await self._ui({"t": "error", "code": "space_unavailable", "message": str(e)})
            await self._reply_alone(text, meta, f"I can't open {desc} right now.")
            return False
        await self._space_message()
        line = f"Back on {desc}." if name == self._cfg.default_space else f"Switching to {desc}."
        if rest:
            await self._say(line)
            await self.run_text(rest, meta)
        else:
            await self._reply_alone(text, meta, line)
        return True

    async def _switch_mode(self, mode: str | None, text: str | None, meta: dict) -> bool:
        hub = self._hub
        if mode not in MODES:
            await self._ui({"t": "error", "code": "mode_unknown", "message": f"no mode {mode!r}"})
            return False
        try:
            await hub.set_mode(mode)
        except (PiError, OSError) as e:
            logger.error(f"{self}: mode switch failed: {e}")
            await self._ui({"t": "error", "code": "space_unavailable", "message": str(e)})
            return False
        await self._space_message()
        await self._reply_alone(text, meta, "Act mode. I can change things now, and I'll ask before anything risky."
                                if mode == "act" else "Back to just talking.")
        return True

    async def run_text(self, text: str, meta: dict | None = None) -> None:
        """Start a turn with this text (a transcript, typed text, or a held turn resuming).

        meta: `input` (voice or text), `t_start` (ISO 8601 with offset), `system` (True when the words are the
        orchestrator's, as in a resume after a hold cut a reply off: such a turn is not logged as the person's)."""
        prev = self._turn_task
        if prev is not None and not prev.done():
            # A new message while a turn still runs (typed text, a resumed turn): it replaces the running turn, as
            # speech would. The interruption stops any audio still playing.
            await self.broadcast_interruption()
            await self._on_interruption(barge_in=not (meta or {}).get("system"))
        self._busy_since = time.monotonic()
        # the state belongs to the new task from this moment: a barge-in while it waits for a silent run cancels it
        st = self._turn_state = _TurnState()
        self._turn_task = self.create_task(self._run_turn(text, dict(meta or {}), st), name="agent_turn")

    # ---------------------------------------------------------------------------------------- interruption

    async def _on_interruption(self, barge_in: bool = False):
        """barge_in: the person spoke or typed over the turn (not a hold cutting it off), so a short run with no tool
        call pending may finish silently instead of being aborted (the module docstring)."""
        self._response_open = False   # downstream ended the response
        self._busy_since = time.monotonic()
        if self._confirm is not None:
            # The person is answering the spoken question; their words arrive as the next transcript. Pipecat
            # broadcasts an interruption at every user turn start (note 04c §2), and this one must not abort the run.
            self._skip_heard_once = True
            return
        task, st = self._turn_task, self._turn_state
        if task is None or task.done():
            return
        if st is not None and st.muted:
            if not barge_in:
                task.cancel()     # a hold starting (or anything but the person) ends a silent run too
            return
        if (barge_in and self._cfg.finish_after_barge_in and st is not None and st.prompted and not st.tool_pending
                and st.chars <= self._cfg.finish_max_chars):
            st.muted = True
            self._muted_task, self._muted_state = task, st
            logger.info(f"{self}: barge-in mid-generation ({st.chars} chars so far): finishing the run silently")
            return
        if st is not None and st.prompted:
            # Said at INFO because an aborted run makes ds4 replay the conversation on the next request (2-3 s at 3k
            # tokens): a push-to-talk barge-in in testing looked like a missed silent finish, but the story had
            # 171 tokens written, over finish_max_chars (ds4's log: live=3023, then a 2,879-token prefill from zero).
            why = ("not a barge-in" if not barge_in else "a tool call is pending" if st.tool_pending
                   else f"{st.chars} chars written, over {self._cfg.finish_max_chars}" if self._cfg.finish_after_barge_in
                   else "silent finish off")
            logger.info(f"{self}: interrupted mid-run ({why}): aborting the run")
        task.cancel()     # PiChild.turn's cleanup schedules the abort; never awaited here (Pipecat waits <= 1 s)

    def on_assistant_turn_stopped(self, content: str, interrupted: bool) -> None:
        """Pipecat's assistant aggregator closed a reply: played to the end (it gets LLMFullResponseEndFrame only after
        the reply's audio was written) or interrupted, with the sentences actually spoken. This is when a turn is over
        for the turn log, and, when interrupted, what the person heard (refined by played_ms if a client sends it)."""
        if interrupted:
            if self._skip_heard_once:   # the person answering a permission question, not a barge-in
                self._skip_heard_once = False
                return
            if self._child is not None and self._heard_from_client is None:
                self._child.set_heard(content or "")
            entry = self._entry
            if entry is not None and not entry.written:
                if self._heard_from_client is None:
                    entry.heard_text = content or ""
                self._entry_interrupted(entry)
            return
        entry = self._entry
        if entry is not None and not entry.written and entry.settled and self._turn_log is not None:
            self._turn_log.write(entry)

    def note_heard(self, text: str) -> None:
        """Kept for callers of the skeleton's interface: an interrupted reply's spoken sentences."""
        self.on_assistant_turn_stopped(text, True)

    async def _on_client_message(self, msg: dict):
        t = msg.get("t")
        if t == "played_ms" and self._timeline is not None and self._child is not None:
            rid = str(msg.get("reply_id") or "")
            if rid in self._timeline.interrupted:
                heard = self._timeline.heard(rid, float(msg.get("ms") or 0))
                if heard is not None:
                    self._heard_from_client = rid
                    self._child.set_heard(heard)
                    logger.debug(f"{self}: played_ms {msg.get('ms')} of {rid}: heard {heard!r}")
                    entry = self._entry
                    if entry is not None and not entry.written and self._turn_log is not None:
                        entry.heard_text, entry.reply_id, entry.interrupted = heard, rid, True
                        self._turn_log.write(entry)
        elif t == "confirm_response" and self._confirm is not None and msg.get("id") == self._confirm.id:
            # PROTOCOL.md: `choice` is one of the offered ids; a client that sends only `confirmed` means once or no
            choice = msg.get("choice")
            if choice not in self._confirm.approval.options:
                choice = "allow_once" if msg.get("confirmed") is True else "deny"
            await self._answer(str(choice), by="client")
        elif t == "space":
            # PROTOCOL.md: the same as saying "go to ...", answered with a `space` message or an `error`
            await self._switch(str(msg.get("name") or ""), "", {"input": "text"}, None)
        elif t == "mode":
            await self._switch_mode(str(msg.get("name") or ""), None, {"input": "text"})

    # ------------------------------------------------------------------------------------------------ turn

    async def _ui(self, message: dict):
        if self._ui_event is not None:
            await self.push_frame(self._ui_event(message))

    async def _open_response(self):
        if not self._response_open:
            self._response_open = True
            await self.push_frame(LLMFullResponseStartFrame())

    async def _close_response(self):
        if self._response_open:
            self._response_open = False
            await self.push_frame(LLMFullResponseEndFrame())

    async def _say(self, text: str):
        await self._open_response()
        await self.push_frame(TTSSpeakFrame(text, append_to_context=False))

    async def _release(self, said: str, cap: SpokenCap, entry: TurnEntry | None, child: PiChild) -> None:
        """The model's text the spoken cap lets through goes to the TTS; once the cap ends the reply, the offer is
        said and the next prompt will say where the speech stopped. The turn log's reply_text is then what was said
        (TURN_LOG.md: what went to TTS and the captions)."""
        if said:
            await self._open_response()
            await self._push_llm_text(said)
            if cap.limit:
                # whole sentences only: the splitter's lookahead would hold each until the cap lets the next one go
                # (the first sentence waited for the second to be written: 3 barge-in tests, 2026-10-05 17:13)
                await self.push_frame(FlushSpeechFrame())
        if cap.cut and not self._last_cut:
            self._last_cut = True
            offer = self._cfg.spoken_cap_offer
            if entry is not None:
                entry.reply_text = cap.spoken
            child.set_cut(cap.spoken, offer)
            if offer:
                await self._say(offer)

    def _begin_entry(self, text: str, meta: dict, space: str) -> TurnEntry | None:
        if self._turn_log is None or meta.get("system"):
            return None
        try:
            t_start = datetime.fromisoformat(str(meta.get("t_start")))
        except ValueError:
            t_start = now_local()
        if t_start.tzinfo is None:
            t_start = t_start.astimezone()
        return self._turn_log.begin(session=self._session_id, t_start=t_start, client=self._client, space=space,
                                    mode=self._hub.mode, input=str(meta.get("input") or "voice"), user_text=text)

    async def _run_turn(self, text: str, meta: dict, st: _TurnState):
        hub, cfg = self._hub, self._cfg
        space = hub.active
        hs = await hub.hold.check()
        if not hs.allows_gpu:
            # Nothing may call the LLM while the gate is draining or held: keep the words for later.
            if self.on_held_text:
                await self.on_held_text(text, hs.holder or hs.why or hs.phase, meta)
            return
        try:
            child = await hub.child(space)
        except (PiError, OSError) as e:
            logger.error(f"{self}: the {space} agent did not start: {e}")
            await self._say("Sorry, I couldn't start the agent.")
            await self._close_response()
            return
        lock = hub.locks[space]
        cancelled = False
        settled = False
        first_text = True
        record = TurnRecord(space=space, at=time.time())
        prev = self._entry
        if prev is not None and not prev.written and self._turn_log is not None:
            self._turn_log.write(prev)       # a reply still playing when the next turn starts is over now
        entry = self._begin_entry(text, meta, space)
        self._entry = entry
        # the spoken cap (spoken_cap.py): longer when asked for something long, or for more after a cut reply
        limit = cfg.spoken_cap_words
        if limit and (wants_more(text) or (self._last_cut and yes_no(text) is True)):
            limit = cfg.spoken_cap_long_words
        self._last_cut = False
        cap = SpokenCap(limit)
        written: list[str] = []
        ack = None
        switch_to: tuple[str, str] | None = None   # the model's space_switch: (space, what they want done there)
        prompt = text
        if hub.space_is_journal(space) and not meta.get("system"):
            # before the reply, so the words are safe first
            prompt, saved = await self._capture(child, space, text, note_next=bool(meta.get("note_next")))
            if saved and cfg.saved_text:
                # said at once: the reply waits on the journal child's model, whose first request in the M3 e2e
                # (2026-10-05 16:53) was a cold prefill of 7.25 s after the 2.2 s turn end, 9.9 s of silence in all
                await self._say(cfg.saved_text)
        if self.tone is not None and meta.get("input") == "voice" and not meta.get("system"):
            # the delivery hint goes to Pi only: never the words, the turn log's user_text, Atlas or the captions
            res = await tone_analyze(self.tone, self.utterance_audio() if self.utterance_audio else b"", text,
                                     session=self._session_id, turn=self.turns_run + 1, channel=self._client)
            if res is not None and getattr(res, "hint", None):
                prompt = f"{res.hint}\n{prompt}"
            if res is not None and getattr(res, "due", None) and entry is not None:
                entry.tone = {"hint": res.due, "shown": bool(getattr(res, "hint", None))}
        # after the capture and its "Saved.": Pi's RPC bash runs at once, while a run the person barged in on still
        # finishes silently, and that run can be a cold prefill (the second half of a split musing waited 6.8 s for
        # its "Saved.", e2e 2026-10-05 17:17)
        await self._finish_muted()
        async with lock:
            self._child = child
            self._heard_from_client = None
            self.turns_run += 1
            hub.turns += 1
            hub.state = "thinking"
            hub.last_turn = record     # the latency observer fills in the rest while this turn runs
            await self._ui({"t": "state", "v": "thinking"})
            await self._open_response()
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            t0 = time.monotonic()
            self.last_llm_idle_s = None if hub.llm_last_active is None else t0 - hub.llm_last_active
            acked = loading_said = False
            progress: asyncio.Task | None = None
            model_captures: dict[str, str] = {}    # call id -> the bash command of a capture the model runs

            async def still_working():
                # a tool chain running on with nothing said since the acknowledgement: say once that it is alive
                await asyncio.sleep(cfg.progress_after_s)
                if first_text and not st.muted and self._confirm is None:
                    # never while a permission question waits for its answer: there the agent is not looking, and
                    # "Still looking." 8 s after the ack talked over the person's yes (M3 e2e, 2026-10-05 16:17)
                    await self._say(cfg.progress_text)

            async def say_ack(tool: str):
                nonlocal acked, ack, progress
                if acked or cap.cut or st.muted:
                    return
                acked = True
                # the model's own words just before the call already said what is happening: no second
                # "Let me check." after its "Let me check current prices." (2026-10-05)
                ack = "model" if cap.spoken.strip() else "canned"
                if ack == "canned":
                    await self._say(cfg.acks.get(tool, cfg.acks.get("default", "One moment.")))
                if cfg.progress_after_s > 0 and cfg.progress_text:
                    progress = self.create_task(still_working(), name="agent_progress")
            try:
                st.prompted = True
                async for ev in child.turn(prompt, settle_timeout=cfg.settle_timeout_s):
                    self._busy_since = time.monotonic()
                    if st.muted and isinstance(ev, ToolCallStarted):
                        # a tool the person never heard about must not run: abort after all (breaking out of
                        # child.turn() makes its cleanup abort the run)
                        logger.info(f"{self}: the silent run started a tool call ({ev.name}): aborting it")
                        break
                    if isinstance(ev, TextDelta):
                        if not ev.text:
                            continue
                        st.chars += len(ev.text)
                        written.append(ev.text)
                        if first_text:
                            first_text = False
                            await self.stop_ttfb_metrics()
                            self.last_ttft_ms = record.llm_ttft_ms = (time.monotonic() - t0) * 1000
                        if cap.cut:
                            continue     # said up to the cap; the rest stays in Pi's context only
                        if entry is not None:
                            entry.reply_text += ev.text
                        if st.muted:
                            continue
                        await self._release(cap.feed(ev.text), cap, entry, child)
                    elif isinstance(ev, ThinkingDelta):
                        if LOADING_BANNER in ev.text and not loading_said and not st.muted:
                            loading_said = True
                            await self._say(cfg.loading_notice)
                    elif isinstance(ev, ToolCallStarted):
                        if not st.muted:
                            await self._release(cap.boundary(), cap, entry, child)
                            await self.push_frame(FlushSpeechFrame())
                        st.tool_pending = True
                        record.tools.append(ev.name)
                        if entry is not None:
                            entry.tool_started(ev.name)
                        hub.tool = {"name": ev.name, "label": cfg.labels.get(ev.name, ev.name), "since": time.time()}
                        await self._ui({"t": "tool", "phase": "start", "name": ev.name,
                                        "label": cfg.labels.get(ev.name, ev.name)})
                        if self._ack_when(ev.name, space) == "now":
                            await say_ack(ev.name)
                    elif isinstance(ev, ToolStart):
                        # comes after toolcall_start, with the arguments, and before any permission question (05c E4)
                        if ev.name == "kb" and str(((ev.args or {}).get("args") or [""])[0]) in KB_READ_VERBS:
                            await say_ack(ev.name)     # a lookup; a kb verb that writes is asked about instead
                        # a capture the model runs itself is noted for the turn log (TURN_LOG.md: by "model")
                        if ev.name == "bash" and "atlas.py capture" in str((ev.args or {}).get("command") or ""):
                            model_captures[ev.call_id] = str(ev.args["command"])
                        if ev.name == SPACE_SWITCH and not meta.get("switched"):
                            args = ev.args or {}
                            if str(args.get("space") or "").strip().lower() in hub.spaces:
                                switch_to = (str(args["space"]).strip().lower(), str(args.get("request") or ""))
                    elif isinstance(ev, ToolEnd):
                        if ev.call_id in model_captures and ev.ok and entry is not None:
                            self._model_captured(entry, space, text, model_captures.pop(ev.call_id), ev.text)
                        st.tool_pending = False
                        hub.tool = None
                        if entry is not None:
                            entry.tool_ended(ev.name, ev.ok)
                        await self._ui({"t": "tool", "phase": "end", "name": ev.name, "ok": bool(ev.ok)})
                    elif isinstance(ev, (ConfirmRequest, SelectRequest)) and (asked := approvals.from_request(ev)):
                        if await self._ask(asked, child, space):
                            await say_ack(asked.tool or "default")   # a session grant answered it: nothing was asked
                    elif isinstance(ev, (SelectRequest, InputRequest)):
                        await child.cancel_ui(ev.id)   # no spoken form for these; cancelling answers at once
                    elif isinstance(ev, Notify):
                        if "held" in ev.message.lower() and self.on_pi_waiting:
                            await self.on_pi_waiting(ev.message)   # hold.ts waits inside the run (05c E5c)
                    elif isinstance(ev, Status):
                        pass
                    elif isinstance(ev, MessageEnd) and ev.role == "assistant":
                        if not st.muted:
                            await self._release(cap.boundary(), cap, entry, child)
                        if ev.stop_reason == "error":
                            logger.warning(f"{self}: model error: {ev.error}")
                            if first_text:
                                await self._say("Sorry, the model didn't answer.")
                    elif isinstance(ev, (Settled, Exited)):
                        if not st.muted:
                            await self._release(cap.boundary(), cap, entry, child)
                        settled = isinstance(ev, Settled)
                        if settled:
                            self.last_settled_at = time.monotonic()
                        break
            except asyncio.CancelledError:
                cancelled = True
                raise
            except PiBusyError as e:
                logger.warning(f"{self}: {e}")
                await self._say("I'm still finishing the last thing. Give me a moment.")
            except Exception as e:  # noqa: BLE001 - the session must survive a failed turn
                logger.exception(f"{self}: agent turn failed")
                await self.push_error(error_msg=f"{self}: agent turn failed: {e}", exception=e)
            finally:
                if progress is not None and not progress.done():
                    await self.cancel_task(progress, timeout=1.0)
                self.last_reply = {"written": words("".join(written)), "said": words(cap.spoken), "cap": limit,
                                   "cut": cap.cut, "ack": ack}
                if cap.cut:
                    logger.info(f"{self}: the spoken cap ended the reply at {cap.said} words "
                                f"({self.last_reply['written']} written, cap {limit})")
                hub.llm_last_active = time.monotonic()
                if self._confirm is not None or self._confirm_queue:
                    await self._withdraw_questions("overtaken")   # the run ended (a barge-in, an abort) with one open
                hub.tool = None
                hub.state = "idle"
                self._busy_since = time.monotonic()
                if first_text:
                    await self.cancel_ttfb_metrics()
                await self.stop_processing_metrics()
                self._finish_entry(entry, cancelled or st.muted)
                if st.muted:
                    self.silent_runs["settled" if settled else "aborted"] += 1
                    logger.info(f"{self}: the silent run {'settled' if settled else 'was aborted'} "
                                f"({st.chars} chars in all)")
                elif not cancelled:
                    if settled and first_text and record.tools and cfg.no_answer_text and not st.declined:
                        await self._say(cfg.no_answer_text)   # tools ran (or were refused) and nothing was said
                    await self._close_response()
                    if self._ui_in_order is not None:   # after the reply's audio: the agent is listening again
                        await self.push_frame(self._ui_in_order({"t": "state", "v": "listening"}))
                    if settled:
                        self.completed_turns += 1
                        if self.completed_turns == 1:
                            self._maybe_speak_digests()
        if switch_to is not None and settled and not st.muted:
            await self._model_switch(switch_to, text, meta)

    async def _model_switch(self, target: tuple[str, str], text: str, meta: dict) -> None:
        """The model called space_switch (voice_mode.ts) and its run is over: switch, and when the person asked for
        something there, send their own words as that space's first prompt (never the model's paraphrase of them: in a
        journal space they could be saved as the person's). One model switch per utterance, so two spaces can never pass
        a request back and forth."""
        name, request = target
        logger.info(f"{self}: the model switched to {name} ({'with' if request.strip() else 'without'} a request there)")
        if name == self._hub.active or not await self._switch(name, "", meta, None):
            return
        if request.strip():
            st = self._turn_state = _TurnState()
            await self._run_turn(text, {**meta, "switched": True}, st)

    async def _capture(self, child: PiChild, space: str, text: str, note_next: bool = False) -> tuple[str, bool]:
        """Atlas (SPACES.md): a musing is saved word for word with `atlas.py capture --via voice` through Pi's RPC bash,
        kept out of the model's context, before the reply; the model is told it is saved. Deterministic and
        byte-exact (05c, 05d), where the model alone did not reliably capture first. Returns the prompt, and whether
        the words were saved. note_next: the person said "note this" on its own just before, so all of this
        utterance is the musing."""
        hub = self._hub
        words = musing(text)
        if words is None and (hub.journal or note_next):
            words = text
        if not words:
            return text, False
        try:
            res = await child.bash(atlas_capture_command(words), exclude_from_context=True, timeout=30.0)
        except PiError as e:
            logger.error(f"{self}: atlas capture failed: {e}")
            return text, False   # the space's rules tell the model to capture first itself
        out = str((res or {}).get("output") or "")
        if (res or {}).get("exitCode") != 0:
            logger.error(f"{self}: atlas capture exited {(res or {}).get('exitCode')}: {out[-300:]}")
            return text, False
        m = re.search(r"\bto (\S+\.md)\b", out)
        path = m.group(1) if m else None
        if self._entry is not None:
            self._entry.atlas = {"root": str(hub.spaces[space].root), "path": path, "by": "orchestrator",
                                 "text": None if words == text else words}
        logger.info(f"{self}: saved {len(words)} chars word for word to {path}")
        self._captured_last = True
        where = f" to {path}" if path else ""
        told = f' and they were told "{self._cfg.saved_text}"' if self._cfg.saved_text else ""
        return (f"{text}\n\n(Their words are already saved word for word{where}{told}; do not save them again or "
                f"say that they are saved.)"), True

    def _model_captured(self, entry: TurnEntry, space: str, user_text: str, command: str, result: str) -> None:
        """The model saved words with atlas.py capture itself (the fallback path, TURN_LOG.md: `by: model`): where they
        went, and the bytes it sent on stdin when they are not the person's words as recognised."""
        m = re.search(r"\bto (\S+\.md)\b", result or "")
        body = heredoc_body(command)
        if entry.atlas is not None or not m:
            return   # the orchestrator's own capture is the record; or the capture did not say where it saved
        entry.atlas = {"root": str(self._hub.spaces[space].root), "path": m.group(1), "by": "model",
                       "text": None if body == user_text else body}
        logger.info(f"{self}: the model saved {len(body or '')} chars to {m.group(1)} itself")

    async def _finish_muted(self) -> None:
        """Before the next prompt: let a barged-in run that is finishing silently settle, so the next request extends
        ds4's state. Once it is writing text, give it up to finish_wait_s, then abort it (a long reply costs less to
        replay than to wait for). While it has written nothing it is waiting on the LLM, usually in a prefill, and is
        left alone up to finish_prefill_max_s: an abort during a prefill makes ds4 delete the checkpoint the prefill
        started from ("kv cache discarded reason=prefill-failed", 2026-10-05 14:15:42), and the next request then
        prefilled all 15,738 tokens from zero (11.7 s, after the 2 s wait: "Rome." 18.3 s after the question)."""
        task, st = self._muted_task, self._muted_state
        if task is None or task.done() or st is None:
            return
        t0 = time.monotonic()
        writing_since = t0 if st.chars else None
        why = None
        while not task.done():
            now = time.monotonic()
            if st.chars and writing_since is None:
                writing_since = now
            if writing_since is not None and now - writing_since >= self._cfg.finish_wait_s:
                why = f"still writing {self._cfg.finish_wait_s:.1f} s after the next prompt was ready"
                break
            if now - t0 >= self._cfg.finish_prefill_max_s:
                why = f"no reply after {self._cfg.finish_prefill_max_s:.0f} s"
                break
            await asyncio.wait({task}, timeout=0.1)
        if why:
            logger.info(f"{self}: the silent run is {why}: aborting it")
            await self.cancel_task(task, timeout=2.0)
        else:
            logger.info(f"{self}: waited {1000 * (time.monotonic() - t0):.0f} ms for the silent run to settle")

    def _finish_entry(self, entry: TurnEntry | None, cancelled: bool) -> None:
        """The agent's side of the turn is over. Cancelled (barge-in before the run settled): interrupted. Settled: the
        record waits for the reply to finish playing or be cut off (on_assistant_turn_stopped), with a fallback."""
        if entry is None or self._turn_log is None or entry.written:
            return
        entry.t_end = now_local()
        if cancelled:
            self._entry_interrupted(entry)
            return
        entry.settled = True
        log = self._turn_log

        async def fallback():   # no reply was closed (nothing was spoken, or the pipeline ended)
            await asyncio.sleep(log.settle_fallback_s)
            log.write(entry)
        entry.fallback = self.create_task(fallback(), name="turnlog_fallback")

    def _entry_interrupted(self, entry: TurnEntry) -> None:
        """Barged in or cut off: the client's played_ms says what was heard; wait a moment for it."""
        if entry.written or self._turn_log is None:
            return
        entry.interrupted = True
        if entry.heard_text is None:
            entry.heard_text = ""
        if entry.t_end is None:
            entry.t_end = now_local()
        if self._timeline is None:   # no protocol v1 client: Pipecat's spoken sentences are all there is
            self._turn_log.write(entry)
        else:
            self._turn_log.write_after_heard(entry)

    async def cleanup(self):
        # The connection is gone, so its turn must not run on unobserved, holding the space's lock and Pi's run.
        # Pipecat leaves our tasks alone at cleanup ("dangling tasks detected"); on 2026-10-05 a closed e2e
        # connection's turn ran 10 s more while the next connection's turn waited for the lock. Cancelling ends it
        # like a barge-in: PiChild.turn's cleanup aborts the run, and the next prompt says what was heard.
        for task in (self._turn_task, self._muted_task):
            if task is not None and not task.done():
                await self.cancel_task(task, timeout=2.0)
        entry = self._entry
        if entry is not None and not entry.written and self._turn_log is not None:
            self._turn_log.write(entry)   # the session ended with a turn still open in the log
        await super().cleanup()

    # ------------------------------------------------------------------------- approvals (PROTOCOL.md "Approvals")

    def _ack_when(self, tool: str, space: str) -> str:
        """When a tool call is acknowledged aloud. `now`: at toolcall_start (lookups; calls a trusted space runs without
        asking). kb: by its verb, once the arguments are in (ToolStart). `never`: in an ask or read-only space a call
        that changes things is asked about (the question is the acknowledgement) or refused (the model says so). In an
        early live test "Let me look that up." came before "May I change your knowledge base?" (2026-10-05)."""
        if tool in READ_TOOLS:
            return "now"
        if tool == SPACE_SWITCH:
            return "never"     # the switch says "Switching to ..." itself
        if tool == "kb":
            return "args"
        return "now" if self._hub.spaces[space].tier == "trusted" else "never"

    async def _ask(self, a: Approval, child: PiChild, space: str) -> bool:
        """Ask about one call: the card (confirm_request with the summary, the action and the choices) and the spoken
        question, then the clock. True when this session's grant for the same scope answered it at once instead."""
        hub = self._hub
        s = hub.approvals
        if self._confirm is not None:
            self._confirm_queue.append((a, child, space))   # tool calls run in parallel: one question at a time
            return False
        grant = hub.grants.match(self._session_id, space, a.scope)
        if grant is not None and "allow_session" in a.options:
            grant.uses += 1
            logger.info(f"{self}: {a.tool} allowed by this session's grant for {grant.label}: {a.summary}")
            await self._send_choice(child, a, "allow_session")
            return True
        card = self._client in s.card_clients
        wait = s.card_wait_s if card else s.voice_wait_s
        c = _Confirm(id=a.id, child=child, approval=a, space=space, question=approvals.spoken_question(a, s),
                     card=card, asked_at=time.monotonic())
        self._confirm = c
        hub.approval = {"id": a.id, "space": space, "tool": a.tool, "summary": a.summary, "client": self._client,
                        "asked_at": now_local().isoformat(timespec="seconds"), "wait_s": wait}
        logger.info(f"{self}: asking ({'card and voice' if card else 'voice only'}, {wait:.0f} s): {a.summary}")
        await self._ui(a.request_message(s, int(wait * 1000)))
        await self._say(c.question)
        c.clock = self.create_task(self._approval_clock(c, wait), name="approval_clock")
        return False

    async def _approval_clock(self, c: _Confirm, wait: float) -> None:
        """PROTOCOL.md: with a card on screen, wait at least `card_wait_s` and ask once more aloud before giving up;
        without one, `voice_wait_s`. Never talk over the person, and never give up while they are still speaking (their
        answer may be in those words)."""
        s = self._hub.approvals
        end = c.asked_at + wait
        if c.card and 0 < s.reask_after_s < wait:
            await asyncio.sleep(max(0.0, c.asked_at + s.reask_after_s - time.monotonic()))
            while self._confirm is c and (self._user_speaking or self._bot_speaking) and time.monotonic() < end:
                await asyncio.sleep(0.2)
            if self._confirm is c:
                await self._say(approvals.spoken_again(c.approval, s))
        await asyncio.sleep(max(0.0, end - time.monotonic()))
        grace = time.monotonic() + 10.0
        while self._confirm is c and self._user_speaking and time.monotonic() < grace:
            await asyncio.sleep(0.2)
        if self._confirm is c:
            await self._answer("timeout", by="clock")

    async def _answer(self, choice: str, *, by: str, said: str = "") -> None:
        """Close the open question with this choice (by voice, client or clock): close the card, keep a session grant,
        say what follows from it (what the grant covers, or plainly what was not done), steer the person's own words
        into the run for deny_said, answer the gate, then ask the next queued question."""
        c, self._confirm = self._confirm, None
        if c is None:
            return
        hub = self._hub
        hub.approval = None
        if c.clock is not None and c.clock is not asyncio.current_task() and not c.clock.done():
            await self.cancel_task(c.clock, timeout=1.0)
        logger.info(f"{self}: approval answered by {by}: {choice} ({c.approval.summary})")
        if by != "client":
            await self._ui({"t": "confirm_cancel", "id": c.id, "why": "timeout" if by == "clock" else "answered"})
        if choice == "allow_session" and c.approval.scope:
            hub.grants.add(self._session_id, c.space, c.approval.scope)
        if choice not in approvals.ALLOW and self._turn_state is not None:
            self._turn_state.declined = True    # what was not done is said here, not "I couldn't find that quickly"
        line = approvals.spoken_outcome(c.approval, hub.approvals, choice)
        if line:
            await self._say(line)
        if choice == "deny_said" and said:
            try:   # before the answer, so the words are queued when the run goes on (Pi delivers a steer after the
                await c.child.steer(said)   # tool batch, before the next model call)
            except PiError as e:
                logger.warning(f"{self}: passing the person's reply on failed: {e}")
        await self._send_choice(c.child, c.approval, choice)
        if self._confirm_queue:
            a, child, space = self._confirm_queue.pop(0)
            await self._ask(a, child, space)

    async def _send_choice(self, child: PiChild, a: Approval, choice: str) -> None:
        try:
            if a.kind == "select":
                await child.answer_select(a.id, choice)
            else:
                await child.answer_confirm(a.id, choice in approvals.ALLOW)
        except PiError as e:
            logger.warning(f"{self}: answering the permission question failed: {e}")

    async def _answer_by_voice(self, text: str):
        c = self._confirm
        s = self._hub.approvals
        choice = approvals.heard_answer(text, c.approval.options, f"{c.question} {approvals.spoken_again(c.approval, s)}")
        logger.info(f"{self}: spoken answer {text!r} -> {choice}")
        if choice is None:
            return   # nothing answered yet (an echo of the question, "wait"): the question stays open
        # Only a plain yes approves. Any other reply is not approved and goes to the model as the person's words, as a
        # coding agent's "no, and tell it what to do instead" does: "yes, but call it X" never runs the call as it was.
        await self._answer(choice, by="voice", said=text if choice == "deny_said" else "")

    async def _withdraw_questions(self, why: str) -> None:
        """The run ended with questions open (a barge-in or an abort dismissed them in Pi): close their cards."""
        pending = ([self._confirm.id] if self._confirm is not None else []) + [a.id for a, _, _ in self._confirm_queue]
        c, self._confirm, self._confirm_queue = self._confirm, None, []
        self._hub.approval = None
        if c is not None and c.clock is not None and c.clock is not asyncio.current_task() and not c.clock.done():
            await self.cancel_task(c.clock, timeout=1.0)
        for qid in pending:
            await self._ui({"t": "confirm_cancel", "id": qid, "why": why})

    # ------------------------------------------------------------------------------------ the brain's digests

    def _maybe_speak_digests(self) -> None:
        if self._digests is None or not self._digests.enabled or not self._new_session:
            return
        if not self._digests.pending():
            return
        if self._digest_task is None or self._digest_task.done():
            self._digest_task = self.create_task(self._speak_digests(), name="brain_digests")

    def _idle(self) -> bool:
        running = self._turn_task is not None and not self._turn_task.done()
        return not (running or self._bot_speaking or self._user_speaking or self._confirm is not None)

    async def _wait_idle(self) -> None:
        """Until nothing has happened for digest_idle_s and the hold gate is open (TURN_LOG.md §2)."""
        while True:
            if self._idle() and time.monotonic() - self._busy_since >= self._digest_idle_s:
                st = await self._hub.hold.check()
                if st.allows_gpu:
                    if self._idle():
                        return
                else:
                    await self._hub.hold.wait_open()
                    self._busy_since = time.monotonic()
            await asyncio.sleep(0.1)

    async def _speak_digests(self) -> None:
        """Speak the pending digests oldest first, each as a reply of its own, kept out of the context; then move
        each to delivered/, played or interrupted, so none is ever spoken twice."""
        await self._wait_idle()
        digests = await self._digests.claim()
        for i, d in enumerate(digests):
            if i:
                await self._wait_idle()
            pushed = False
            try:
                self._spoke.clear()
                self._played.clear()
                await self.push_frame(TTSSpeakFrame(d.text, append_to_context=False))
                pushed = True
                await asyncio.wait_for(self._spoke.wait(), 30)
                await asyncio.wait_for(self._played.wait(), 120)
            except asyncio.TimeoutError:
                logger.warning(f"{self}: brain digest {d.id}: no playback seen; counting it as delivered")
            finally:
                if pushed:
                    self._digests.delivered(d)
                else:
                    self._digests.release(d)
