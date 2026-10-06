"""Wire protocol v1 (PROTOCOL.md) on Pipecat 1.12.0: a FrameSerializer and two small processors.

Construct-only skeleton (research note 04c). What reaches a FrameSerializer on FastAPIWebsocketTransport:
- serialize(): OutputAudioRawFrame from write_audio_frame (already resampled to audio_out_sample_rate and cut
  to audio_out_10ms_chunks x 10 ms, base_output.py:604-627, websocket/fastapi.py:526-557); the
  InterruptionFrame itself (fastapi.py:499-514); OutputTransportMessageFrame and
  OutputTransportMessageUrgentFrame via send_message (fastapi.py:516-524). Nothing else: write_transport_frame
  is a no-op on this transport (base_output.py:277-287), so state, transcript and reply messages must be
  sent as message frames. Urgent ones (SystemFrame) go out at once (base_output.py:364-365); plain ones
  (DataFrame) wait in the audio queue behind the audio already queued (base_output.py:657-663, 875-876),
  which keeps audio_start / audio_end / end_of_turn in order with the audio, and are dropped on
  interruption.
- deserialize(): its result is pushed downstream by the input transport; InputAudioRawFrame goes through the
  audio queue, InputTransportMessageFrame is broadcast both ways, anything else is pushed downstream
  (fastapi.py:372-396). The input transport does not resample (base_input.py:197-299), so audio must arrive
  at audio_in_sample_rate (16 kHz, which protocol v1 specifies).
- Control messages are returned as ClientMessageFrame, not InputTransportMessageFrame: the worker prepends an
  RTVIProcessor by default (pipeline/worker.py:492-518, 581), and it swallows every non-RTVI
  InputTransportMessageFrame with a warning (rtvi/processor.py:288-294). Native clients run with
  enable_rtvi=False anyway.
- A client "interrupt" becomes InterruptionWorkerFrame; the worker turns it into an InterruptionFrame sent
  from the top of the pipeline (pipeline/worker.py:1465-1471, 1553-1555).
- Push-to-talk "start"/"stop" become VAD frames, which SegmentedSTTService segments on
  (stt_service.py:924-956) and VADUserTurnStartStrategy turns into a user turn plus interruption. Known race:
  "stop" is pushed directly while the last audio frames may still sit in the input transport's audio queue
  (base_input.py:197-204, 269-296), so up to a few 20-40 ms frames can miss the segment. Have the client send
  "stop" after ~100 ms of trailing silence, or add a processor after transport.input() that delays the stop
  until audio has gone quiet for 50 ms.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer

IN_RATE = 16000
OUT_RATE = 24000


@dataclass
class ClientMessageFrame(SystemFrame):
    """A protocol v1 control message from the client (played_ms, space, mode, confirm_response, ping...)."""

    message: dict = field(default_factory=dict)


def urgent(message: dict) -> OutputTransportMessageUrgentFrame:
    """A message sent to the client immediately."""
    return OutputTransportMessageUrgentFrame(message=message)


def in_order(message: dict) -> OutputTransportMessageFrame:
    """A message sent after the audio already queued, and dropped by an interruption."""
    return OutputTransportMessageFrame(message=message)


class ProtocolV1Serializer(FrameSerializer):
    """PCM16 mono 16 kHz up, PCM16 mono 24 kHz down, JSON control messages with a "t" field."""

    def __init__(self, *, in_rate: int = IN_RATE, out_rate: int = OUT_RATE):
        """Initialize; one instance per connection (it tracks the reply being played)."""
        super().__init__(FrameSerializer.InputParams(ignore_rtvi_messages=True))
        self._in_rate = in_rate
        self._out_rate = out_rate
        self.active_reply: str | None = None  # set by audio_start, cleared by end_of_turn or interrupt
        self.client_interrupted = False  # set when the client itself sent "interrupt"

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Frame to wire. Returns None for frames this protocol does not carry."""
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            # The output transport already resampled to audio_out_sample_rate (24 kHz) and chunked it:
            # audio_out_10ms_chunks=4 gives 1,920-byte messages, under the 3,000-byte limit.
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            # Every user turn start broadcasts an interruption (llm_response_universal.py:1362-1363), even
            # with nothing playing; only tell the client when a reply is actually being played.
            if self.active_reply is None or self.client_interrupted:
                self.client_interrupted = False
                self.active_reply = None
                return None
            msg = {"t": "interrupt", "reply_id": self.active_reply}
            self.active_reply = None
            return json.dumps(msg, separators=(",", ":"))
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            msg = frame.message
            if not isinstance(msg, dict) or "t" not in msg:
                return None
            if msg["t"] == "audio_start":
                self.active_reply = msg.get("reply_id")
            elif msg["t"] == "end_of_turn":
                self.active_reply = None
            return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Wire to frame."""
        if isinstance(data, (bytes, bytearray)):
            if len(data) % 2:
                data = data[:-1]
            return InputAudioRawFrame(audio=bytes(data), sample_rate=self._in_rate, num_channels=1)
        try:
            msg = json.loads(data)
        except ValueError:
            logger.warning(f"{self}: dropping a non-JSON text message")
            return None
        if not isinstance(msg, dict):
            return None
        t = msg.get("t")
        if t == "interrupt":
            self.client_interrupted = True
            return InterruptionWorkerFrame()
        if t == "start":
            return VADUserStartedSpeakingFrame(start_secs=0.0, timestamp=time.time())
        if t == "stop":
            return VADUserStoppedSpeakingFrame(stop_secs=0.0, timestamp=time.time())
        if t == "text" and isinstance(msg.get("text"), str):
            # Typed input: append a user message and run the LLM (the user aggregator handles this frame,
            # llm_response_universal.py:853-854).
            return LLMMessagesAppendFrame(messages=[{"role": "user", "content": msg["text"]}], run_llm=True)
        return ClientMessageFrame(message=msg)


class ProtocolV1Ears(FrameProcessor):
    """Between STT and the user aggregator: transcript and listening/thinking state; answers ping.

    It must sit before the user aggregator because the aggregator consumes TranscriptionFrame
    (llm_response_universal.py:836-850). UserStarted/StoppedSpeakingFrame are broadcast by the aggregator
    (llm_response_universal.py:1357-1358, 1443-1444), so their upstream copies pass through here.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Emit protocol messages for the frames that pass, and forward everything."""
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(urgent({"t": "transcript", "final": True, "text": frame.text}))
        elif isinstance(frame, InterimTranscriptionFrame):
            await self.push_frame(urgent({"t": "transcript", "final": False, "text": frame.text}))
        elif isinstance(frame, UserStartedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            await self.push_frame(urgent({"t": "state", "v": "listening"}))
        elif isinstance(frame, UserStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            await self.push_frame(urgent({"t": "state", "v": "thinking"}))
        elif isinstance(frame, ClientMessageFrame) and frame.message.get("t") == "ping":
            await self.push_frame(urgent({"t": "pong", "n": frame.message.get("n", 0)}))
            return
        await self.push_frame(frame, direction)


class ProtocolV1Mouth(FrameProcessor):
    """Between TTS and transport.output(): audio_start, reply_text, audio_end, end_of_turn, speaking state."""

    def __init__(self, *, out_rate: int = OUT_RATE, **kwargs: Any):
        """Initialize."""
        super().__init__(**kwargs)
        self._out_rate = out_rate
        self._reply: str | None = None
        self._in_llm_response = False
        self._audio_open = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Translate TTS output framing into protocol v1 messages, in order with the audio."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._in_llm_response = True
                self._reply = f"r{uuid.uuid4().hex[:8]}"
            elif isinstance(frame, TTSStartedFrame):
                await self._open_audio()
            elif (
                isinstance(frame, AggregatedTextFrame)
                and not isinstance(frame, TTSTextFrame)  # TTSTextFrame subclasses it (frames.py:459) and comes after the audio
                and frame.aggregated_by != "code"
            ):
                # The sentence about to be synthesized. It precedes its own audio, and for the first sentence of a
                # turn it even precedes TTSStartedFrame (tts_service.py:1339-1348 vs 1387-1393), so open here too.
                await self._open_audio()
                await self.push_frame(frame, direction)
                await self.push_frame(in_order({"t": "reply_text", "reply_id": self._reply, "delta": frame.text}))
                return
            elif isinstance(frame, TTSStoppedFrame) and not self._in_llm_response:
                await self.push_frame(frame, direction)
                await self._close_reply()
                return
            elif isinstance(frame, LLMFullResponseEndFrame):
                # TTS releases this only after the turn's audio contexts drained (tts_service.py:933-939).
                await self.push_frame(frame, direction)
                self._in_llm_response = False
                await self._close_reply()
                return
            elif isinstance(frame, InterruptionFrame):
                self._reply, self._in_llm_response, self._audio_open = None, False, False
        else:
            if isinstance(frame, BotStartedSpeakingFrame):
                await self.push_frame(urgent({"t": "state", "v": "speaking"}))
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self.push_frame(urgent({"t": "state", "v": "listening"}))
        await self.push_frame(frame, direction)

    async def _open_audio(self):
        if self._reply is None:  # a TTSSpeakFrame outside any LLM response (busy notice, greeting)
            self._reply = f"r{uuid.uuid4().hex[:8]}"
        if not self._audio_open:
            self._audio_open = True
            await self.push_frame(in_order({"t": "audio_start", "reply_id": self._reply, "rate": self._out_rate}))

    async def _close_reply(self):
        if self._reply is None:
            return
        if self._audio_open:
            await self.push_frame(in_order({"t": "audio_end", "reply_id": self._reply}))
        await self.push_frame(in_order({"t": "end_of_turn", "reply_id": self._reply}))
        self._reply, self._audio_open = None, False
