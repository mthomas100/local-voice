"""Wire protocol v1 (PROTOCOL.md) on Pipecat 1.12.0: the serializer, and the two processors that speak it.

What reaches a FrameSerializer on FastAPIWebsocketTransport (note 04c §2): audio already resampled to 24 kHz and cut
into audio_out_10ms_chunks x 10 ms; the InterruptionFrame itself; OutputTransportMessage(Urgent)Frame. Nothing else,
so state, transcript and reply messages travel as message frames: urgent ones go out at once, plain ones wait behind
the audio already queued (which keeps audio_start / audio_end / end_of_turn in order with the audio) and are dropped
by an interruption.

Client control messages become `ClientMessageFrame` (a SystemFrame), not InputTransportMessageFrame, which the
auto-added RTVI processor would swallow; native pipelines run with enable_rtvi=False anyway.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
MAX_BINARY = 3000          # a reported URLSessionWebSocketTask limit; 40 ms at 24 kHz is 1,920 bytes


@dataclass
class ClientMessageFrame(SystemFrame):
    """A protocol v1 control message from the client (played_ms, space, mode, confirm_response, ping, ...)."""

    message: dict = field(default_factory=dict)


def urgent(message: dict) -> OutputTransportMessageUrgentFrame:
    """A message sent to the client at once."""
    return OutputTransportMessageUrgentFrame(message=message)


def in_order(message: dict) -> OutputTransportMessageFrame:
    """A message sent after the audio already queued; an interruption drops it."""
    return OutputTransportMessageFrame(message=message)


def dumps(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


# Speaking rate assumed for a sentence still being synthesized when no sentence of the reply has finished yet (Qwen3-TTS
# "Ryan" speaks about 2.6 words a second). Only the in-progress sentence of an interrupted reply is estimated this way.
WORDS_PER_MS = 2.6 / 1000


class ReplyTimeline:
    """Where each spoken sentence sits in a reply's audio, so a client's `played_ms` maps to the words it heard.

    Pipecat itself records only sentences whose audio was handed to the transport (note 04c §1); the client's
    playout buffer means less was heard. played_ms counts the reply's audio from its audio_start.

    A sentence is recorded when all its audio has been queued (TTSTextFrame comes after it), but its text is announced
    before its audio (`upcoming`). A barge-in during a long first sentence used to map to no words at all, so the next
    prompt said nothing was heard after 2.4 s of a story (e2e 2026-10-05). The sentence still streaming now counts too,
    at the reply's own speaking rate so far."""

    def __init__(self):
        self.reply_id: str | None = None
        self.sent_ms = 0.0
        self._start_ms = 0.0
        self.sentences: list[tuple[float, float, str]] = []
        self._upcoming: list[str] = []   # announced sentences whose audio is not all queued yet, in order
        self._past: dict[str, tuple[list[tuple[float, float, str]], str | None, float]] = {}
        self.interrupted: set[str] = set()

    def open(self, reply_id: str) -> None:
        if self.reply_id and (self.sentences or self._upcoming):
            self._past[self.reply_id] = (self.sentences, self._upcoming[0] if self._upcoming else None, self._start_ms)
        self.reply_id, self.sent_ms, self._start_ms, self.sentences, self._upcoming = reply_id, 0.0, 0.0, [], []
        while len(self._past) > 8:
            self._past.pop(next(iter(self._past)))

    def audio(self, nbytes: int, rate: int) -> None:
        self.sent_ms += nbytes / 2 / rate * 1000

    def upcoming(self, text: str) -> None:
        """A sentence about to be synthesized: its text arrives before its audio."""
        self._upcoming.append(text)

    def sentence(self, text: str) -> None:
        """A sentence's audio has all been queued (TTSTextFrame comes after it)."""
        self.sentences.append((self._start_ms, self.sent_ms, text))
        self._start_ms = self.sent_ms
        if self._upcoming:
            self._upcoming.pop(0)

    def skip(self) -> None:
        """Audio that belongs to no recorded sentence (an acknowledgement) ends here."""
        self._start_ms = self.sent_ms
        if self._upcoming:
            self._upcoming.pop(0)

    def heard(self, reply_id: str, played_ms: float) -> str | None:
        if reply_id == self.reply_id:
            sentences, partial, partial_start = self.sentences, self._upcoming[0] if self._upcoming else None, self._start_ms
        elif reply_id in self._past:
            sentences, partial, partial_start = self._past[reply_id]
        else:
            return None
        words: list[str] = []
        for start, end, text in sentences:
            if played_ms >= end:
                words += text.split()
            elif played_ms > start and end > start:
                part = text.split()
                words += part[: int(len(part) * (played_ms - start) / (end - start))]
                break
            else:
                break
        else:
            if partial and played_ms > partial_start:   # into the sentence that was still streaming
                done_words = sum(len(t.split()) for _, _, t in sentences)
                rate = done_words / sentences[-1][1] if done_words and sentences[-1][1] > 0 else WORDS_PER_MS
                part = partial.split()
                words += part[: min(len(part), int((played_ms - partial_start) * rate))]
        return " ".join(words)


class ProtocolV1Serializer(FrameSerializer):
    """PCM16 mono 16 kHz up, PCM16 mono 24 kHz down, JSON control messages with a "t" field."""

    def __init__(self, *, in_rate: int = IN_RATE, on_client_message=None):
        super().__init__(FrameSerializer.InputParams(ignore_rtvi_messages=True))
        self._in_rate = in_rate
        self.active_reply: str | None = None   # from audio_start to end_of_turn; only then is `interrupt` sent
        self.client_interrupted = False
        self.on_client_message = on_client_message   # every parsed control message, for keepalive bookkeeping
        self.timeline: ReplyTimeline | None = None    # the session's, set by the server once the pipeline is built
        self.sent_binary = 0
        # The last `state` sent. Several processors report the same change (the Ears when the person stops, the agent
        # when its run starts; the Mouth when the reply's audio ends, the agent after the reply), so every turn sent
        # `thinking` and `listening` twice (a real-server run, 2026-10-05): a repeat is dropped here, at the
        # one place that sees the wire's order.
        self.last_state: str | None = None

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            self.sent_binary += 1
            if len(frame.audio) > MAX_BINARY:
                logger.warning(f"{self}: audio message of {len(frame.audio)} bytes exceeds {MAX_BINARY}")
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            # Every user turn start broadcasts an interruption (note 04c §2); tell the client only while a reply plays,
            # and not when the client asked for it itself.
            if self.active_reply is None or self.client_interrupted:
                self.client_interrupted = False
                self.active_reply = None
                return None
            msg = {"t": "interrupt", "reply_id": self.active_reply}
            self.active_reply = None
            return dumps(msg)
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            msg = frame.message
            if not isinstance(msg, dict) or "t" not in msg:
                return None
            if msg["t"] == "audio_start":
                self.active_reply = msg.get("reply_id")
            elif msg["t"] == "end_of_turn" and msg.get("reply_id") == self.active_reply:
                self.active_reply = None
            elif msg["t"] == "state":
                if msg.get("v") == self.last_state:
                    return None
                self.last_state = msg.get("v")
            return dumps(msg)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
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
        if self.on_client_message:
            self.on_client_message(msg)
        t = msg.get("t")
        if t == "interrupt":
            self.client_interrupted = True
            if self.timeline is not None and msg.get("reply_id"):
                # The client's played_ms follows at once and is a system frame, while the interruption goes round
                # the pipeline's worker before the Mouth marks the reply: without this the played_ms was dropped
                # and the next prompt said nothing was heard (a push-to-talk barge-in, test of 2026-10-05).
                self.timeline.interrupted.add(str(msg["reply_id"]))
            return InterruptionWorkerFrame()
        if t == "start":
            return VADUserStartedSpeakingFrame(start_secs=0.0, timestamp=time.time())
        if t == "stop":
            return VADUserStoppedSpeakingFrame(stop_secs=0.0, timestamp=time.time())
        if t == "text" and isinstance(msg.get("text"), str) and msg["text"].strip():
            # Typed input: a user message that runs the agent (the user aggregator handles this frame).
            # `lv` tells the agent (and the brain's turn log) how and when the words came in.
            lv = {"input": "text", "t_start": datetime.now().astimezone().isoformat()}
            return LLMMessagesAppendFrame(messages=[{"role": "user", "content": msg["text"].strip(), "lv": lv}],
                                          run_llm=True)
        return ClientMessageFrame(message=msg)


class ProtocolV1Ears(FrameProcessor):
    """Between STT and the user aggregator: transcripts, listening/thinking state, pong.

    It sits before the aggregator because the aggregator consumes TranscriptionFrame; the aggregator broadcasts
    User(Started|Stopped)SpeakingFrame, so their upstream copies pass through here."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
    """Between TTS and transport.output(): audio_start, reply_text, audio_end, end_of_turn, speaking state; and the
    reply timeline that turns a client's played_ms into the words it heard."""

    def __init__(self, *, timeline: ReplyTimeline | None = None, out_rate: int = OUT_RATE, **kwargs: Any):
        super().__init__(**kwargs)
        self._out_rate = out_rate
        self.timeline = timeline or ReplyTimeline()
        self._reply: str | None = None
        self._in_llm_response = False
        self._audio_open = False

    @property
    def reply_id(self) -> str | None:
        return self._reply

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._in_llm_response = True
                self._new_reply()
            elif isinstance(frame, TTSStartedFrame):
                await self._open_audio()
            elif isinstance(frame, OutputAudioRawFrame):
                if self._audio_open:
                    self.timeline.audio(len(frame.audio), frame.sample_rate)
            elif isinstance(frame, TTSTextFrame):
                if frame.append_to_context:
                    self.timeline.sentence(frame.text)
                else:
                    self.timeline.skip()
            elif isinstance(frame, AggregatedTextFrame) and frame.aggregated_by != "code" and frame.append_to_context is False \
                    and getattr(frame, "will_be_spoken", False):
                # The sentence about to be synthesized; for the first sentence of a turn it even precedes
                # TTSStartedFrame (tts_service.py), so open the audio here too.
                await self._open_audio()
                self.timeline.upcoming(frame.text)
                await self.push_frame(frame, direction)
                await self.push_frame(in_order({"t": "reply_text", "reply_id": self._reply, "delta": frame.text}))
                return
            elif isinstance(frame, TTSStoppedFrame) and not self._in_llm_response:
                await self.push_frame(frame, direction)
                await self._close_reply()
                return
            elif isinstance(frame, LLMFullResponseEndFrame):
                # The TTS releases this only after the turn's audio contexts drained (tts_service.py).
                await self.push_frame(frame, direction)
                self._in_llm_response = False
                await self._close_reply()
                return
            elif isinstance(frame, InterruptionFrame):
                if self._reply and self._audio_open:
                    self.timeline.interrupted.add(self._reply)
                self._reply, self._in_llm_response, self._audio_open = None, False, False
        else:
            if isinstance(frame, BotStartedSpeakingFrame):
                await self.push_frame(urgent({"t": "state", "v": "speaking"}))
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self.push_frame(urgent({"t": "state", "v": "listening"}))
        await self.push_frame(frame, direction)

    def _new_reply(self):
        self._reply = f"r{uuid.uuid4().hex[:8]}"
        self._audio_open = False

    async def _open_audio(self):
        if self._reply is None:   # a TTSSpeakFrame outside any LLM response (a notice)
            self._new_reply()
        if not self._audio_open:
            self._audio_open = True
            self.timeline.open(self._reply)
            await self.push_frame(in_order({"t": "audio_start", "reply_id": self._reply, "rate": self._out_rate}))

    async def _close_reply(self):
        if self._reply is None:
            return
        if self._audio_open:
            await self.push_frame(in_order({"t": "audio_end", "reply_id": self._reply}))
        await self.push_frame(in_order({"t": "end_of_turn", "reply_id": self._reply}))
        self._reply, self._audio_open = None, False
