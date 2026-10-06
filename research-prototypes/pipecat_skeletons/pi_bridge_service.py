"""A Pipecat "LLM" service that fronts an external agent (Pi over RPC) which keeps its own conversation.

Construct-only skeleton for the orchestrator (research note 04c). The backend is anything with the
AgentBackend shape below; research-prototypes/pi_rpc_bridge/pi_rpc_bridge.py's PiChild has it (turn(),
interrupt(), busy, drain(); events TextDelta, ToolCallStarted, ToolStart, ToolUpdate, ToolEnd,
ConfirmRequest, Settled, Exited). Events are matched by class name so this file does not import the prototype.

How it maps onto Pipecat 1.12.0 (file:line in the installed package):
- Base class LLMService (services/llm_service.py:249). Handle LLMContextFrame in process_frame exactly as
  BaseOpenAILLMService does (services/openai/base_llm.py:633-658): push LLMFullResponseStartFrame, stream
  LLMTextFrames, push LLMFullResponseEndFrame in `finally`, forward every other frame.
- LLMContextFrame is a plain Frame (frames.py:619), so it runs on the processor's data task. An
  InterruptionFrame (a SystemFrame, frames.py:1221) cancels that task and starts a new one
  (frame_processor.py:819-847, 1129-1156); the turn below sees asyncio.CancelledError. The cancel waits up to
  1 s for the task to finish (frame_processor.py:1256-1260, base_object.py:154-165), so the abort is
  fire-and-forget, as Pipecat's own OpenClaw service does it (services/openclaw/gateway.py:186-189).
- Tools never run inside Pipecat: only LLMService.run_function_calls executes handlers
  (llm_service.py:1582), and this service never calls it.
- The user aggregator writes {"role": "user", "content": text} and pushes the context
  (llm_response_universal.py:900-928). It can also append a "developer" message on an empty interrupted
  turn (llm_response_universal.py:1525-1557, turns/empty_user_turn.py:11-16 and 51); build the aggregator
  with empty_user_turn=None, and this service still ignores non-user roles. It sends Pi only user messages
  it has not sent before, tracked by object identity, so a re-pushed context never re-prompts Pi.
- Only spoken text reaches the record: the TTS marks its aggregated input append_to_context=False
  (tts_service.py:1330) and emits a TTSTextFrame per sentence after that sentence's audio
  (tts_service.py:1435-1454); the output transport releases it only after the audio was written
  (base_output.py:657-663, 955-982); interruption drops what is queued. The assistant aggregator's
  on_assistant_turn_stopped(message.interrupted=True) carries exactly the fully spoken sentences
  (llm_response_universal.py:1902-1904, 2413-2435). note_heard() takes that and the next prompt tells Pi.
- A TTSSpeakFrame pushed mid-response is spoken as its own audio context; with append_to_context=False it
  stays out of the record (tts_service.py:950-984).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MixerEnableFrame,
    OutputTransportMessageUrgentFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings


class AgentBackend(Protocol):
    """What the bridge needs from an agent runtime. PiChild (pi_rpc_bridge.py) has this shape.

    Contract the bridge relies on: after interrupt() returns and `busy` is False, drain() empties every
    event left over from the aborted run, so the next turn() starts on a clean stream.
    """

    @property
    def busy(self) -> bool:
        """True from the start of a turn until the runtime reports it settled."""
        ...

    def turn(self, text: str) -> AsyncIterator[Any]:
        """Send one user turn; yield its events up to and including the settled event."""
        ...

    async def interrupt(self, heard: str | None = None) -> Any:
        """Clear queued steer/follow-up text and abort the run in flight."""
        ...

    def drain(self) -> list[Any]:
        """Take whatever events are waiting, without blocking."""
        ...


def interrupted_note(heard: str) -> str:
    """The prefix that tells the agent where it was cut off (same wording as the 05c prototype)."""
    heard = heard.strip()
    if not heard:
        return "(You were interrupted before the user heard any of your last reply.)\n\n"
    return f'(You were interrupted. The user heard only this much of your last reply: "{heard}")\n\n'


def protocol_v1_message(message: dict) -> Frame:
    """UI events for native clients: wire protocol v1 JSON, sent at once by the serializer."""
    return OutputTransportMessageUrgentFrame(message=message)


class PiBridgeLLMService(LLMService):
    """Streams an external agent's reply into Pipecat as if it were an LLM."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        ack_phrases: dict[str, str] | None = None,
        default_ack: str = "Let me check.",
        tool_labels: dict[str, str] | None = None,
        ui_event: Callable[[dict], Frame] | None = protocol_v1_message,
        use_mixer_for_tools: bool = False,
        pipecat_tool_frames: bool = False,
        settle_timeout_secs: float = 5.0,
        **kwargs,
    ):
        """Initialize.

        Args:
            backend: The agent runtime (a PiChild for the active space).
            ack_phrases: Spoken acknowledgement per tool name, said once per turn when the first tool starts.
            default_ack: Acknowledgement for tools without their own phrase.
            tool_labels: UI label per tool name ("reading your journal").
            ui_event: Turns a protocol v1 message dict into a frame (OutputTransportMessageUrgentFrame for
                native clients; for the browser, wrap it in RTVIServerMessageFrame). None sends no UI events.
            use_mixer_for_tools: Toggle the transport's SoundfileMixer (the "working" loop) while a tool runs.
                Only for transports that have a mixer; see thinking_sound.py for the trade-offs.
            pipecat_tool_frames: Also emit Pipecat's function-call frames, for RTVI clients and the latency
                breakdown. Off by default: they write tool messages into the Pipecat context, keep the user
                idle controller counting until closed, and a FunctionCallResultFrame with a result and no
                run_llm=False re-runs inference, which would re-prompt the agent
                (llm_response_universal.py:1926-2027, turns/user_idle_controller.py:162-167).
            settle_timeout_secs: How long to wait for an aborted run to settle before prompting again.
            **kwargs: Passed to LLMService.
        """
        kwargs.setdefault("settings", LLMSettings(model=None))
        super().__init__(**kwargs)
        self._backend = backend
        self._acks = ack_phrases or {}
        self._default_ack = default_ack
        self._labels = tool_labels or {}
        self._ui_event = ui_event
        self._use_mixer = use_mixer_for_tools
        self._tool_frames = pipecat_tool_frames
        self._settle_timeout = settle_timeout_secs

        # Context messages already seen, keyed by id(). The values keep the objects alive so an id cannot be
        # recycled by a new message while this service runs.
        self._seen: dict[int, Any] = {}
        self._heard: str | None = None  # what the user heard of an interrupted reply
        self._in_turn = False
        self._interrupt_task: asyncio.Task | None = None
        self._needs_quiesce = False
        self._mixer_on = False
        self._mixer_wanted = False
        self._open_calls: dict[str, str] = {}  # call_id -> tool name, for pipecat_tool_frames

    def can_generate_metrics(self) -> bool:
        """Report TTFB (prompt to first text delta) and processing time."""
        return True

    def note_heard(self, text: str) -> None:
        """Record what the user heard of the interrupted reply; the next prompt carries it.

        Wire to the assistant aggregator: on_assistant_turn_stopped with message.interrupted=True.
        """
        self._heard = text or ""

    # ------------------------------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Run turns on LLMContextFrame, abort on interruption, forward everything else."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._run_turn(frame)
        elif isinstance(frame, InterruptionFrame):
            # The data task (and with it any turn in flight) has already been cancelled by the base class.
            await self.push_frame(frame, direction)
            await self._after_interruption()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            if self._mixer_wanted and not self._mixer_on:
                await self._set_mixer(True)  # the acknowledgement has finished; start the loop under silence
        else:
            await self.push_frame(frame, direction)

    async def _after_interruption(self):
        if self._mixer_on or self._mixer_wanted:
            self._mixer_wanted = False
            await self._set_mixer(False)  # queued mixer frames were dropped with everything else
        if self._tool_frames:
            for call_id, name in list(self._open_calls.items()):
                await self.push_frame(FunctionCallCancelFrame(function_name=name, tool_call_id=call_id))
            self._open_calls.clear()

    # -------------------------------------------------------------------------------------------- turn

    def _new_user_text(self, context: LLMContext) -> str:
        parts = []
        for msg in context.get_messages():
            if id(msg) in self._seen:
                continue
            self._seen[id(msg)] = msg
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content.strip())
                elif isinstance(content, list):  # multi-part content: keep the text parts
                    parts += [p.get("text", "").strip() for p in content if isinstance(p, dict)]
        return " ".join(p for p in parts if p)

    async def _quiesce(self):
        """Make sure an aborted run has finished and left nothing in the event stream."""
        if self._interrupt_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._interrupt_task), self._settle_timeout)
            except Exception as e:
                logger.warning(f"{self}: interrupting the agent did not finish cleanly: {e}")
            self._interrupt_task = None
        if self._needs_quiesce:
            deadline = time.monotonic() + self._settle_timeout
            while getattr(self._backend, "busy", False) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            stale = self._backend.drain()
            if stale:
                logger.debug(f"{self}: dropped {len(stale)} events left from the aborted run")
            self._needs_quiesce = False

    async def _run_turn(self, frame: LLMContextFrame):
        if frame.speculation:
            return  # eager end-of-turn speculation is not configured; never prompt the agent speculatively
        text = self._new_user_text(frame.context)
        if not text:
            return  # nothing new from the user (a re-pushed context, a developer note)
        await self._quiesce()
        if self._heard is not None:
            text = interrupted_note(self._heard) + text
            self._heard = None

        acked = False
        first_text = True
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        self._in_turn = True
        try:
            async for ev in self._backend.turn(text):
                kind = type(ev).__name__
                if kind == "TextDelta":
                    if first_text:
                        await self.stop_ttfb_metrics()
                        first_text = False
                    if self._mixer_on or self._mixer_wanted:
                        self._mixer_wanted = False
                        await self._set_mixer(False)
                    await self._push_llm_text(ev.text)  # LLMTextFrame, plus TTFAT (llm_service.py:806-824)
                elif kind in ("ToolCallStarted", "ToolStart"):
                    if not acked:
                        acked = True
                        phrase = self._acks.get(ev.name, self._default_ack)
                        await self.push_frame(TTSSpeakFrame(phrase, append_to_context=False))
                        self._mixer_wanted = self._use_mixer
                    if kind == "ToolStart":
                        await self._ui({"t": "tool", "phase": "start", "name": ev.name,
                                        "label": self._labels.get(ev.name, ev.name)})
                        await self._maybe_tool_frames_start(ev)
                elif kind == "ToolEnd":
                    await self._ui({"t": "tool", "phase": "end", "name": ev.name, "ok": bool(ev.ok)})
                    await self._maybe_tool_frames_end(ev)
                elif kind == "ConfirmRequest":
                    # M3: speak the question, listen for yes/no, answer with PiChild.answer_confirm.
                    await self._ui({"t": "confirm_request", "id": getattr(ev, "id", ""),
                                    "title": getattr(ev, "title", ""), "message": getattr(ev, "message", ""),
                                    "timeout_ms": getattr(ev, "timeout_ms", 0)})
                elif kind in ("Settled", "Exited"):
                    break
        except asyncio.CancelledError:
            # Barge-in. Abort without awaiting (see the module docstring); the next turn waits for it.
            self._needs_quiesce = True
            self._interrupt_task = self.create_task(self._backend.interrupt(None), name="agent_interrupt")
            raise
        except Exception as e:
            await self.push_error(error_msg=f"{self}: agent turn failed: {e}", exception=e)
        finally:
            self._in_turn = False
            if first_text:
                await self.cancel_ttfb_metrics()
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    # ----------------------------------------------------------------------------------------- helpers

    async def _ui(self, message: dict):
        if self._ui_event:
            await self.push_frame(self._ui_event(message))

    async def _set_mixer(self, on: bool):
        self._mixer_on = on
        await self.push_frame(MixerEnableFrame(enable=on))

    async def _maybe_tool_frames_start(self, ev: Any):
        if not self._tool_frames:
            return
        call = FunctionCallFromLLM(function_name=ev.name, tool_call_id=ev.call_id,
                                   arguments=getattr(ev, "args", {}) or {}, context=None)
        self._open_calls[ev.call_id] = ev.name
        await self.push_frame(FunctionCallsStartedFrame(function_calls=[call]))
        await self.push_frame(FunctionCallInProgressFrame(function_name=ev.name, tool_call_id=ev.call_id,
                                                          arguments=call.arguments, cancel_on_interruption=True))

    async def _maybe_tool_frames_end(self, ev: Any):
        if not self._tool_frames or ev.call_id not in self._open_calls:
            return
        del self._open_calls[ev.call_id]
        # run_llm=False is load-bearing: without it the assistant aggregator pushes the context upstream
        # and this service would prompt the agent again (llm_response_universal.py:2001-2027).
        await self.push_frame(FunctionCallResultFrame(function_name=ev.name, tool_call_id=ev.call_id,
                                                      arguments={}, result={"ok": bool(ev.ok)}, run_llm=False))


# --------------------------------------------------------------------------------------- fake (no agent)


class TextDelta:
    """Fake event: a text chunk (same name and field as the prototype's)."""

    def __init__(self, text: str):
        """Store the text."""
        self.text = text


class ToolCallStarted:
    """Fake event: the model began a tool call."""

    def __init__(self, name: str, call_id: str):
        """Store the tool name and call id."""
        self.name, self.call_id = name, call_id


class ToolStart(ToolCallStarted):
    """Fake event: the tool started running."""

    def __init__(self, name: str, call_id: str, args: dict):
        """Store the tool name, call id and arguments."""
        super().__init__(name, call_id)
        self.args = args


class ToolEnd(ToolCallStarted):
    """Fake event: the tool finished."""

    def __init__(self, name: str, call_id: str, ok: bool):
        """Store the tool name, call id and outcome."""
        super().__init__(name, call_id)
        self.ok = ok


class Settled:
    """Fake event: the agent will not continue on its own."""


class FakeAgentBackend:
    """Scripted stand-in for PiChild: one tool call, then a two-sentence answer."""

    def __init__(self):
        """Initialize with an idle state and an empty event queue."""
        self._busy = False
        self.events: asyncio.Queue = asyncio.Queue()
        self.prompts: list[str] = []
        self.interrupts = 0

    @property
    def busy(self) -> bool:
        """True while a scripted turn is running."""
        return self._busy

    async def turn(self, text: str) -> AsyncIterator[Any]:
        """Yield a scripted tool call and reply."""
        self.prompts.append(text)
        self._busy = True
        try:
            for ev in (
                ToolCallStarted("kb", "c1"),
                ToolStart("kb", "c1", {"q": "hold gate"}),
                ToolEnd("kb", "c1", True),
                TextDelta("The hold gate pauses model calls. "),
                TextDelta("It opens when the render ends."),
                Settled(),
            ):
                await asyncio.sleep(0)
                yield ev
        finally:
            self._busy = False

    async def interrupt(self, heard: str | None = None) -> None:
        """Record the abort."""
        self.interrupts += 1
        self._busy = False

    def drain(self) -> list[Any]:
        """Return and clear waiting events."""
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out
