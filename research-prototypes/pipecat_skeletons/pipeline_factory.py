"""One pipeline shape for both entry points: protocol v1 over WebSocket, and the browser over SmallWebRTC.

    transport.input() -> stt -> [Ears] -> user aggregator -> Pi bridge -> code-fence splitter -> tts
                      -> [Mouth] -> transport.output() -> assistant aggregator

[Ears] and [Mouth] are the protocol v1 processors (native clients only); the browser gets the same events
from RTVI, which the worker adds by default (pipeline/worker.py:492-518).

Facts (pipecat-ai 1.12.0):
- Default turn stack: start = [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy], stop =
  [TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())] (turns/user_turn_strategies.py:29-80).
  UserTurnStrategies() builds LocalSmartTurnAnalyzerV3 at construction, and that opens its ONNX session in
  __init__ (local_smart_turn_v3.py:68-78); SileroVADAnalyzer loads its ONNX in __init__ too (silero.py:
  138-169). user_params(load_models=False) avoids both for construct checks and unit tests.
- empty_user_turn=None stops the aggregator from appending a "developer" message and re-running the LLM after
  an interruption that produced no transcript (llm_response_universal.py:1525-1557).
- The pair's assistant aggregator emits on_assistant_turn_stopped(aggregator, message) with
  message.interrupted and message.content = the sentences that were actually played
  (llm_response_universal.py:2413-2435); that is what the bridge tells Pi on the next prompt.
- Code in replies: LLMTextProcessor with PatternPairAggregator AGGREGATE turns ``` blocks into
  AggregatedTextFrame(type "code") (llm_text_processor.py:30-72, pattern_pair_aggregator.py:25-43, 125-131);
  the TTS speaks none of it when built with skip_aggregator_types=["code"] (tts_service.py:1269-1271) and the
  assistant aggregator still records it.
- idle_timeout_secs defaults to 300 s and then cancels the worker and, by default, the whole WorkerRunner
  (pipeline/worker.py:102, 293-300, 537-542, 1605-1629). None disables the monitor.
- PipelineParams defaults are already 16 kHz in / 24 kHz out (pipeline/worker.py:187-188); metrics are off by
  default (190-191).
"""

from __future__ import annotations

from collections.abc import Sequence

from pipecat.frames.frames import Frame
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
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator

from protocol_v1 import ProtocolV1Ears, ProtocolV1Mouth

IN_RATE = 16000
OUT_RATE = 24000


def user_params(
    *,
    mic: str = "vad",
    load_models: bool = True,
    vad_stop_secs: float = 0.2,
    smart_turn_stop_secs: float = 3.0,
) -> LLMUserAggregatorParams:
    """User-turn configuration.

    Args:
        mic: "vad" (open mic, Silero + Smart Turn) or "ptt" (the client sends start/stop; protocol_v1 maps them
            to VAD frames, so no VAD model runs and the turn ends as soon as the transcript is final).
        load_models: False builds a timer-only stop strategy and no VAD, so nothing loads (tests, checks).
        vad_stop_secs: Silero silence before a pause counts; 0.2 is the value Pipecat's p99 tables assume
            (turn_analyzer_user_turn_stop_strategy.py:247-257).
        smart_turn_stop_secs: Silence after which the turn ends whatever Smart Turn says (default 3 s,
            base_smart_turn.py:27-43).
    """
    if mic == "ptt":
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)]
        return LLMUserAggregatorParams(
            vad_analyzer=None, user_turn_strategies=UserTurnStrategies(stop=stop), empty_user_turn=None
        )
    if not load_models:
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)]
        return LLMUserAggregatorParams(
            vad_analyzer=None, user_turn_strategies=UserTurnStrategies(stop=stop), empty_user_turn=None
        )

    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    vad = SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs))  # loads silero_vad.onnx (2.3 MB, CPU)
    turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=smart_turn_stop_secs))  # 8.7 MB ONNX, CPU
    stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)]
    return LLMUserAggregatorParams(
        vad_analyzer=vad, user_turn_strategies=UserTurnStrategies(stop=stop), empty_user_turn=None
    )


def code_fence_splitter() -> LLMTextProcessor:
    """Sentence aggregation that pulls fenced code out as its own "code" aggregation."""
    return LLMTextProcessor(
        text_aggregator=PatternPairAggregator().add_pattern("code", "```", "```", action=MatchAction.AGGREGATE)
    )


def build_pipeline(
    *,
    transport: BaseTransport,
    stt: FrameProcessor,
    llm: FrameProcessor,
    tts: FrameProcessor,
    params: LLMUserAggregatorParams,
    protocol_v1: bool,
) -> tuple[Pipeline, LLMContextAggregatorPair]:
    """Assemble the processors; wires "what the user heard" from the assistant aggregator into the bridge."""
    pair = LLMContextAggregatorPair(LLMContext(), user_params=params)
    user_agg, assistant_agg = pair.user(), pair.assistant()

    procs: list[FrameProcessor] = [transport.input(), stt]
    if protocol_v1:
        procs.append(ProtocolV1Ears())
    procs += [user_agg, llm, code_fence_splitter(), tts]
    if protocol_v1:
        procs.append(ProtocolV1Mouth(out_rate=OUT_RATE))
    procs += [transport.output(), assistant_agg]

    note_heard = getattr(llm, "note_heard", None)
    if note_heard:

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def _on_assistant_turn_stopped(_: LLMAssistantAggregator, message: AssistantTurnStoppedMessage):
            if message.interrupted:
                note_heard(message.content or "")

    return Pipeline(procs), pair


def build_worker(
    pipeline: Pipeline, *, observers: Sequence[BaseObserver] = (), rtvi: bool, name: str = "voice"
) -> PipelineWorker:
    """A worker that never idles out, reports metrics, and has RTVI only for the browser."""
    return PipelineWorker(
        pipeline,
        name=name,
        params=PipelineParams(
            audio_in_sample_rate=IN_RATE,
            audio_out_sample_rate=OUT_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=None,
        enable_rtvi=rtvi,
        observers=list(observers),
    )


def frames_for_busy_notice(text: str) -> list[Frame]:
    """What to queue when the hold gate is held: a spoken notice, kept out of the context."""
    from pipecat.frames.frames import TTSSpeakFrame

    return [TTSSpeakFrame(text, append_to_context=False)]
