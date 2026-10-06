import Foundation

public enum TalkPhase: String, Sendable {
    case idle
    /// Push-to-talk held: audio streams after `start`.
    case talking
    /// Released: the last capture chunk, the 100 ms tail and `stop` are on their way.
    case releasing
}

public enum AudioEngineStatus: Sendable, Equatable {
    case starting
    case running(String)
    case stopped
    case failed(String)
}

/// One push-to-talk utterance as sent: for logs and the e2e checks.
public struct UtteranceStats: Sendable, Equatable {
    public var frames = 0
    public var speechMs = 0
    /// Digital silence sent before `stop` (at least 100 ms).
    public var tailMs = 0
    public var peakDBFS = -120.0

    public init() {}
}

/// Client-side timing of one push-to-talk turn: from the moment `stop` was handed to the socket (what the server's
/// own end-of-speech-to-first-audio starts from), and from the release itself (what the person waits).
public struct TurnLatency: Sendable, Equatable {
    public var reply: String?
    /// `stop` to `audio_start` arriving: the server's end-of-speech-to-first-audio plus the network.
    public var stopToAudioStartMs: Int?
    /// `stop` to the first reply audio going to the player: adds the pre-roll.
    public var stopToPlaybackMs: Int?
    /// The release to the first reply audio going to the player: adds the wait for the last capture chunk before the
    /// tail and `stop` (31-35 ms with the synthetic microphone against the real server, 2026-10-05; up to 150 ms
    /// if a real tap delivers large chunks). The dashboard shows this one.
    public var releaseToPlaybackMs: Int?

    public init(reply: String? = nil, stopToAudioStartMs: Int? = nil, stopToPlaybackMs: Int? = nil,
                releaseToPlaybackMs: Int? = nil) {
        self.reply = reply
        self.stopToAudioStartMs = stopToAudioStartMs
        self.stopToPlaybackMs = stopToPlaybackMs
        self.releaseToPlaybackMs = releaseToPlaybackMs
    }
}

/// A switch the user asked for: a space (`space` message) or a mode (`mode` message).
public struct SwitchRequest: Sendable, Equatable {
    public enum Kind: String, Sendable {
        case space, mode
    }

    public var kind: Kind
    public var name: String

    public init(_ kind: Kind, _ name: String) {
        self.kind = kind
        self.name = name
    }
}

/// What came of a switch. The protocol gives `space` and `mode` no request id: the answer is the server's next `space`
/// message (PROTOCOL.md: "current space and mode, after any switch"), or an `error` while the switch is pending.
public enum SwitchOutcome: Sendable, Equatable {
    /// The server's `space` message shows what was asked for.
    case switched(SwitchRequest, SpaceInfo)
    /// The server answered with an error, or with a `space` message that is not what was asked for.
    case refused(SwitchRequest, reason: String)
    /// Nothing came back in time (the M1 orchestrator ignores both messages; M3 adds them).
    case noAnswer(SwitchRequest)

    public var request: SwitchRequest {
        switch self {
        case let .switched(r, _), let .refused(r, _), let .noAnswer(r): return r
        }
    }
}

/// Everything the client reports, in order, for the UI and the event log.
public enum ClientEvent: Sendable {
    case connection(ConnectionState)
    case server(ServerMessage)
    case malformed(String)
    case talk(TalkPhase)
    case handsFree(Bool)
    case playback(PlaybackNotice)
    /// A control message handed to the socket (audio is summarised by `utterance`).
    case sent(ClientMessage)
    case utterance(UtteranceStats)
    case latency(TurnLatency)
    case gate(armed: Bool)
    case audioEngine(AudioEngineStatus)
    /// Reply audio dropped after a local flush, until the next `audio_start`.
    case discardedAudio(bytes: Int)
    /// Bytes of reply audio received for a reply, logged when the reply finishes.
    case replyAudio(reply: String, bytes: Int, chunks: Int, maxChunkBytes: Int)
    /// How much audio each capture callback delivers, when it changes. A tap may deliver 100 ms at a time (AVAudioNode
    /// documents 100-400 ms), which delays the end of speech by up to that much; this tells from a real run.
    case captureChunk(ms: Int)
    /// `/v1/status` fetched, and why it was.
    case status(StatusMonitor.Reason, ServerStatus)
    case statusFailed(StatusMonitor.Reason, String)
    case switchRequested(SwitchRequest)
    case switchOutcome(SwitchOutcome)
}

public struct ClientSettings: Sendable, Equatable {
    public var mic: MicMode
    public var gate: MicGate
    public var prerollMs: Int
    public var maxPrerollMs: Int
    public var maxPrerollWaitMs: Int
    /// Stop the audio engine (and with it the microphone and voice-processing ducking) after this long with nothing
    /// to do; `nil` keeps it running. A hands-free session never idles out.
    ///
    /// `AVAudioEngine.start()` measured 375-480 ms on this Mac (output only, cold or warm, 2026-10-05), so a press on a
    /// stopped engine loses the first half-second of speech; keeping it warm between turns avoids that.
    public var engineIdleStopSeconds: Int?
    /// Play a short blip when the microphone is live after a push-to-talk press (at once if the engine is warm, when it
    /// has started if not), so a cold first press says when to speak.
    public var pressChirp: Bool

    public init(mic: MicMode, gate: MicGate = MicGate(), prerollMs: Int = 100, maxPrerollMs: Int = 300,
                maxPrerollWaitMs: Int = 400, engineIdleStopSeconds: Int? = 60, pressChirp: Bool = false) {
        self.mic = mic
        self.gate = gate
        self.prerollMs = prerollMs
        self.maxPrerollMs = maxPrerollMs
        self.maxPrerollWaitMs = maxPrerollWaitMs
        self.engineIdleStopSeconds = engineIdleStopSeconds
        self.pressChirp = pressChirp
    }
}
