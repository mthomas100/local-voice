"""One pipeline per connection, the same shape for both entry points:

    transport.input() -> STT -> [echo filter] -> [Ears] -> user aggregator (VAD, Smart Turn) -> Pi agent
      -> code-fence splitter -> TTS -> notice player -> [Mouth] -> transport.output() -> assistant aggregator

[Ears] and [Mouth] speak protocol v1 (native clients); the browser gets the same events from RTVI. The speech models
are process-wide (runtime.py); everything here is per connection. The echo filter and the turn start that waits for
words while the agent is busy come from config.yaml's echo: section (echo_guard.py, turn_start.py).

Pipecat 1.12.0 settings that matter (notes 04b, 04c): empty_user_turn=None, or an interrupted turn without a
transcript re-runs the agent with a "developer" note; idle_timeout_secs=None and cancel_runner_on_idle_timeout=False,
or the default 300 s idle timeout cancels the worker and its runner; metrics on, or no latency breakdown; RTVI only
for the browser (its processor swallows other control messages).
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator

from .agent import AgentHub, FlushSpeechFrame, PiAgentService
from .bargein import PausableWebsocketOutput, PauseSettings, ReplyPause, SpeechCueMixin, cue_silero, log_events
from .config import Config
from .digests import DigestOutbox
from .echo_guard import EchoGuard, echo_filter
from .latency import TurnLatencyLog
from .protocol_v1 import ProtocolV1Ears, ProtocolV1Mouth, ReplyTimeline, in_order, urgent
from .recorder import attach_recorder
from .runtime import SpeechRuntime
from .session import HoldCoordinator, NoticePlayer
from .turn_end import WordsTurnStopStrategy
from .turn_start import AgentActivity, BusyHoldStartStrategy, hold_applies
from .turnlog import TurnLog

IN_RATE = 16000
OUT_RATE = 24000


@dataclass
class TurnSetup:
    """How the user's turn starts and ends. Tests pass their own (an energy VAD, a timer) to load no model."""
    vad: VADAnalyzer | None
    stop: list[BaseUserTurnStopStrategy]
    stop_timeout_s: float = 5.0


def turn_setup(cfg: Config, mic: str) -> TurnSetup:
    """Open mic: Silero (CPU ONNX) and Smart Turn v3.2 (CPU ONNX), both bundled with Pipecat and loaded here.
    Push-to-talk: the client's start/stop are the VAD; the turn ends as soon as the transcript is final."""
    if mic == "ptt":
        return TurnSetup(vad=None, stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=cfg.ptt_tail_s)],
                         stop_timeout_s=cfg.user_turn_stop_timeout_s)
    vad = cue_silero(params=VADParams(**cfg.vad))   # Silero that also reports each frame's score (bargein.py)
    if cfg.smart_turn_enabled:
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        w = cfg.smart_turn_words
        stop: list[BaseUserTurnStopStrategy] = [WordsTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=cfg.smart_turn_stop_secs)),
            punctuation_stop_secs=w.get("punctuation_stop_secs"), veto_wait_secs=w.get("veto_wait_secs"),
            veto_unpunctuated=bool(w.get("veto_unpunctuated")), command_stop_secs=w.get("command_stop_secs"),
            hold_secs=cfg.smart_turn_stop_secs if w.get("hold_dangling") else None,
            hold_except_questions=bool(w.get("hold_except_questions", True)))]
    else:
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)]
    return TurnSetup(vad=vad, stop=stop, stop_timeout_s=cfg.user_turn_stop_timeout_s)


class InOrderTextProcessor(LLMTextProcessor):
    """An LLMTextProcessor that lets nothing overtake the text it holds: a TTSSpeakFrame (an acknowledgement, the
    spoken cap's offer) first flushes the sentence the aggregator is holding for lookahead, as the end of a response
    would. Pipecat's TTS speaks a TTSSpeakFrame at once, so the acknowledgement "Let me check." was said before the
    model's own sentence written ahead of its tool call (test runs, 2026-10-05: "Let me check.Let me
    check current prices rather than guessing."), and the spoken cap's offer before the sentence it follows."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, FlushSpeechFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            await self._handle_llm_end()     # the agent's request, at a tool call; the frame goes no further
            return
        if isinstance(frame, TTSSpeakFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_llm_end()
        await super().process_frame(frame, direction)


def code_fence_splitter() -> LLMTextProcessor:
    """Fenced code becomes its own "code" aggregation, which the TTS skips and the record keeps (note 04c §1)."""
    return InOrderTextProcessor(
        text_aggregator=PatternPairAggregator().add_pattern("code", "```", "```", action=MatchAction.AGGREGATE))


@dataclass
class Session:
    """Everything one connection's pipeline is made of, for the server, the status page and the tests."""
    id: str
    device: str
    client: str
    mic: str
    protocol_v1: bool
    pipeline: Pipeline
    worker: PipelineWorker
    agent: PiAgentService
    stt: Any
    tts: Any
    notices: NoticePlayer
    coordinator: HoldCoordinator
    timeline: ReplyTimeline
    mouth: ProtocolV1Mouth | None
    latency: TurnLatencyLog
    context: LLMContext
    reply_pause: ReplyPause | None = None
    echo_guard: EchoGuard | None = None           # echo.guard on: what the agent said, and the echo heard
    turn_start: BusyHoldStartStrategy | None = None   # echo.hold_for_words applies to this connection
    started: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    async def queue(self, frame: Frame) -> None:
        await self.worker.queue_frame(frame)


def build_session(*, cfg: Config, runtime: SpeechRuntime, hub: AgentHub, transport: BaseTransport,
                  protocol_v1: bool, mic: str = "vad", device: str = "", client: str = "", session_id: str | None = None,
                  turn: TurnSetup | None = None, ui_event=None, latency_dir: Path | None = None,
                  observers: Sequence[BaseObserver] = (), mixer: bool = False, turn_log: TurnLog | None = None,
                  digests: DigestOutbox | None = None, new_session: bool = True, tone=None) -> Session:
    sid = session_id or f"s{uuid.uuid4().hex[:8]}"
    timeline = ReplyTimeline()
    turn = turn or turn_setup(cfg, mic)
    notices = NoticePlayer(name=f"notices-{sid}")
    if ui_event is None:
        ui_event = urgent if protocol_v1 else None

    async def send(msg: dict) -> None:
        if ui_event is not None:
            await notices.push_frame(ui_event(msg))

    stt_ref: dict[str, Any] = {}
    coordinator: HoldCoordinator   # created below, after the services it reports for

    async def held_audio(pcm: bytes, why: str):
        await coordinator.held_audio_in(pcm, why)

    async def tts_held(why: str):
        await coordinator.speech_refused(why)

    stt = runtime.make_stt(on_held_audio=held_audio)
    tts = runtime.make_tts(on_held=tts_held, sample_rate=OUT_RATE)
    stt_ref["stt"] = stt
    agent = PiAgentService(hub=hub, ui_event=ui_event, ui_in_order=in_order if protocol_v1 else None,
                           timeline=timeline if protocol_v1 else None, session_id=sid, client=client,
                           turn_log=turn_log, digests=digests if cfg.speak_digests else None, new_session=new_session,
                           digest_idle_s=cfg.digest_idle_s)

    for st in turn.stop:
        if isinstance(st, WordsTurnStopStrategy) and st.is_command is None:
            st.is_command = agent.is_command      # a command the orchestrator answers ends its turn early
    if tone is not None and hasattr(stt, "take_turn_audio"):
        agent.tone = tone
        stt.keep_turn_audio = True
        agent.utterance_audio = stt.take_turn_audio
    # The agent's own voice heard back (config.yaml echo:): the guard learns every sentence the TTS says and the filter
    # after the STT drops echo while the agent is busy; where hold_for_words applies, a VAD start while it is busy
    # waits for the words. Both ask AgentActivity whether it is busy, so they never disagree.
    echo = cfg.echo
    guard = None
    if echo.get("guard"):
        guard = EchoGuard(window_s=echo["window_s"], min_words=echo["min_words"], min_share=echo["min_share"],
                          fragment_sentences=echo["fragment_sentences"])
        tts.echo_guard = guard
        agent.echo_guard = guard     # for EchoGuard.turn_note, when agent.py asks

    def run_state() -> str:
        """The agent's run, for AgentActivity: "confirm" while a spoken yes/no waits for its answer, "running" while
        its run is in progress (not one finishing silently after a barge-in: that one says nothing more), else
        "idle". Reads PiAgentService's private _confirm, _turn_task and _turn_state (agent.py,
        2026-10-05): to be replaced by an accessor of its own."""
        if agent._confirm is not None:
            return "confirm"
        task, st = agent._turn_task, agent._turn_state
        if task is not None and not task.done() and not (st is not None and st.muted):
            return "running"
        return "idle"

    activity = AgentActivity(run_state, tail_s=echo.get("tail_s", 0.0))   # the tail: echo lags "Bot stopped speaking"
    start = None
    if echo and hold_applies(echo["hold_for_words"], protocol_v1=protocol_v1, mic=mic):
        start = BusyHoldStartStrategy(activity=activity, guard=guard, max_hold_s=echo["max_hold_s"],
                                      final_wait_s=echo["final_wait_s"], no_words=echo["no_words"])
    context = LLMContext()
    params = LLMUserAggregatorParams(vad_analyzer=turn.vad,
                                     user_turn_strategies=UserTurnStrategies(start=[start] if start else None,
                                                                             stop=turn.stop),
                                     empty_user_turn=None, user_turn_stop_timeout=turn.stop_timeout_s)
    pair = LLMContextAggregatorPair(context, user_params=params)
    user_agg, assistant_agg = pair.user(), pair.assistant()

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def _reply_over(_: LLMAssistantAggregator, message: AssistantTurnStoppedMessage):
        agent.on_assistant_turn_stopped(message.content or "", bool(message.interrupted))

    mouth = ProtocolV1Mouth(timeline=timeline, out_rate=OUT_RATE) if protocol_v1 else None
    procs = [transport.input(), stt]
    if guard is not None:   # before the Ears: echo dropped here never reaches a protocol v1 client's captions
        echo_gate = echo_filter(guard, busy=activity.busy, on_echo=start.on_echo if start else None, watch=activity.saw)
        procs.append(echo_gate)
        if hasattr(stt, "transcript_gate"):   # judged before the STT pushes: RTVI captions the page from that push
            stt.transcript_gate = echo_gate.gate
    if protocol_v1:
        procs.append(ProtocolV1Ears())
    procs += [user_agg, agent, code_fence_splitter(), tts, notices]
    if mouth is not None:
        procs.append(mouth)
    procs += [transport.output(), assistant_agg]
    pipeline = Pipeline(procs)

    def stages() -> dict[str, Any]:
        return {"stt_ms": _r(getattr(stt, "last_stt_ms", None)), "llm_ttft_ms": _r(agent.last_ttft_ms),
                "llm_idle_s": _r(agent.last_llm_idle_s),
                "tts_first_audio_ms": _r(getattr(tts, "last_first_chunk_ms", None))}

    def on_turn(line: dict[str, Any]) -> None:
        rec = hub.last_turn
        rec.eos_to_first_audio_ms = float(line.get("eos_to_first_audio_ms") or 0)
        st = line.get("stages") or {}
        rec.stt_ms = float(st.get("stt_ms") or 0)
        rec.llm_ttft_ms = float(st.get("llm_ttft_ms") or 0)
        rec.tts_first_audio_ms = float(st.get("tts_first_audio_ms") or 0)

    latency = TurnLatencyLog(latency_dir if cfg.latency_log else None, session=sid, stages=stages, on_turn=on_turn,
                             hold_phase=lambda: hub.hold.state.phase)
    worker = PipelineWorker(
        pipeline, name=f"voice-{sid}",
        params=PipelineParams(audio_in_sample_rate=IN_RATE, audio_out_sample_rate=OUT_RATE, enable_metrics=True,
                              enable_usage_metrics=True),
        idle_timeout_secs=None, cancel_on_idle_timeout=False, cancel_runner_on_idle_timeout=False,
        enable_rtvi=not protocol_v1, observers=[*latency.observers, *observers])

    async def interrupt():
        await agent.broadcast_interruption()
        await agent._on_interruption()

    # The reply pause at the first speech frame (bargein.py): open mic, protocol v1 output, turned on in config.
    pause = None
    out = transport.output()
    if cfg.barge_in_pause and isinstance(turn.vad, SpeechCueMixin) and isinstance(out, PausableWebsocketOutput):
        s = cfg.barge_in_pause
        pause = ReplyPause(settings=PauseSettings(cue_confidence=s["cue_confidence"], cue_frames=int(s["cue_frames"]),
                                                  min_volume=turn.vad.params.min_volume,
                                                  resume_after_s=s["resume_after_s"], max_pause_s=s["max_pause_s"]),
                           on_event=log_events(device or sid))
        out.reply_pause = pause
        turn.vad.cue_sink = pause.frame_threadsafe
    if start is not None:
        start.pause = pause

    coordinator = HoldCoordinator(hold=hub.hold, notice=runtime.busy_notice, player=notices, send=send,
                                  transcribe=runtime.transcribe, queue_frame=worker.queue_frame, interrupt=interrupt)
    agent.on_held_text = coordinator.held_text_in
    agent.on_pi_waiting = coordinator.pi_waiting
    session = Session(id=sid, device=device, client=client, mic=mic, protocol_v1=protocol_v1, pipeline=pipeline,
                      worker=worker, agent=agent, stt=stt, tts=tts, notices=notices, coordinator=coordinator,
                      timeline=timeline, mouth=mouth, latency=latency, context=context, reply_pause=pause,
                      echo_guard=guard, turn_start=start)
    attach_recorder(session, transport=transport, cfg=cfg)   # config.yaml record: (recorder.py), off by default
    return session


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)
