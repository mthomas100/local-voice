import Foundation

/// Constants of wire protocol v1 (`PROTOCOL.md`, frozen 2026-10-05). A breaking change gets `/v2/voice`, never an
/// edit here; receivers ignore unknown message types, so additive messages are allowed.
public enum ProtocolV1 {
    public static let version = 1
    public static let path = "/v1/voice"
    public static let statusPath = "/v1/status"
    public static let defaultPort = 8770

    /// Client to server: PCM signed 16-bit little-endian, mono, 16 kHz.
    public static let captureSampleRate = 16_000
    /// Server to client: PCM signed 16-bit little-endian, mono, 24 kHz.
    public static let playbackSampleRate = 24_000

    /// One binary message per 20-40 ms. 20 ms keeps end-of-speech latency low; 640 bytes.
    public static let captureFrameDuration: Duration = .milliseconds(20)
    public static var captureFrameSamples: Int { captureSampleRate / 50 }

    /// A reported `URLSessionWebSocketTask` failure for binary frames above about 3,000 bytes (research notes).
    public static let maxBinaryMessageBytes = 3_000

    /// Keepalive: a ping every 15 s; either side closes after 60 s without any message.
    public static let pingInterval: Duration = .seconds(15)
    public static let idleTimeout: Duration = .seconds(60)

    /// Push-to-talk clients send about 100 ms of silence before `stop`, so the last word is flushed through the
    /// server's VAD before the turn closes (Pipecat pushes `stop` while the last frames may still be queued, Pipecat research notes).
    public static let pushToTalkTail: Duration = .milliseconds(100)

    /// After `audio_end` and the playback draining, the client ignores its microphone for 500-800 ms unless new audio
    /// arrives (residual echo tail of Apple's voice processing, research notes).
    public static let micGateRange: ClosedRange<Int> = 500...800

    /// A client reconnecting with the same `device` within 5 minutes resumes the same space and agent session.
    public static let resumeWindow: Duration = .seconds(300)
}

/// WebSocket close codes with protocol meaning (PROTOCOL.md "Close codes").
public enum CloseCode {
    public static let normal = 1000
    public static let goingAway = 1001
    /// A missing or malformed `hello`, an unknown `v`, a binary frame before `hello`.
    public static let protocolError = 1002
    /// No close frame was received (the TCP connection dropped). Never sent on the wire.
    public static let abnormal = 1006
    /// The peer is not one of the owner's tailnet logins.
    public static let notAllowed = 4403
    /// Another session with the same `device` took over.
    public static let replaced = 4409

    /// Whether a client should reconnect after the server closed with this code.
    ///
    /// 4403 will not change until the server's `allowed_logins` does, 4409 means another copy of this device is
    /// connected (fighting it would make both flap), and 1002 is a bug in this client that a retry would repeat.
    public static func shouldReconnect(after code: Int) -> Bool {
        switch code {
        case protocolError, notAllowed, replaced: return false
        default: return true
        }
    }

    public static func describe(_ code: Int) -> String {
        switch code {
        case normal: return "normal close"
        case goingAway: return "going away"
        case protocolError: return "protocol error"
        case abnormal: return "connection lost"
        case notAllowed: return "not allowed: this device is not one of the owner's tailnet logins"
        case replaced: return "another session with the same device name took over"
        default: return "closed (\(code))"
        }
    }
}
