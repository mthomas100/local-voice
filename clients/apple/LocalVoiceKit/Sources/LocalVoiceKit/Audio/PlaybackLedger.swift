import Foundation

/// Counts how much of each reply was actually played, for `played_ms` (PROTOCOL.md).
///
/// `AVAudioPlayerNode`'s own sample time keeps advancing while the player is starved, so it cannot say how much audio
/// was heard. This ledger counts the frames of buffers whose completion fired, plus the elapsed part of the buffer
/// playing now. The head buffer starts when the previous one completed, or when it was scheduled if the player had run
/// dry. With `.dataPlayedBack` completions the count includes the output device's latency; the partial estimate for the
/// buffer in progress can run ahead by that latency (10-60 ms), which is fine for truncating a transcript to the words
/// that were heard.
///
/// Buffers carry their reply id: a canned acknowledgement can be its own reply, queued right behind the answer's first
/// audio, and each reply gets its own `played_ms`.
public struct PlaybackLedger: Sendable, Equatable {
    public struct Item: Sendable, Equatable {
        public var reply: String?
        public var frames: Int
    }

    public let sampleRate: Int
    public private(set) var scheduledFrames = 0
    public private(set) var completedFrames = 0
    private var inFlight: [Item] = []
    private var headStartedAt: Nanos?
    private var completedByReply: [String: Int] = [:]

    public init(sampleRate: Int = ProtocolV1.playbackSampleRate) {
        self.sampleRate = sampleRate
    }

    public var isDrained: Bool { inFlight.isEmpty }
    public var queuedFrames: Int { inFlight.reduce(0) { $0 + $1.frames } }

    public func hasInFlight(reply: String?) -> Bool { inFlight.contains { $0.reply == reply } }

    /// The replies that still have audio queued, in playing order.
    public var queuedReplies: [String?] {
        var seen: [String?] = []
        for item in inFlight where !seen.contains(item.reply) { seen.append(item.reply) }
        return seen
    }

    public mutating func scheduled(frames: Int, reply: String? = nil, at now: Nanos) {
        guard frames > 0 else { return }
        if inFlight.isEmpty { headStartedAt = now }
        inFlight.append(Item(reply: reply, frames: frames))
        scheduledFrames += frames
    }

    /// One buffer finished (completions arrive in scheduling order). Returns the finished item.
    @discardableResult
    public mutating func completed(at now: Nanos) -> Item? {
        guard !inFlight.isEmpty else { return nil }
        let item = inFlight.removeFirst()
        completedFrames += item.frames
        if let reply = item.reply { completedByReply[reply, default: 0] += item.frames }
        headStartedAt = inFlight.isEmpty ? nil : now
        return item
    }

    private func partialFrames(at now: Nanos) -> (Item, Int)? {
        guard let head = inFlight.first, let start = headStartedAt, now > start else { return nil }
        let elapsed = Int(Double(now - start) / 1e9 * Double(sampleRate))
        return (head, min(elapsed, head.frames))
    }

    public func playedFrames(at now: Nanos) -> Int {
        completedFrames + (partialFrames(at: now)?.1 ?? 0)
    }

    public func playedFrames(reply: String, at now: Nanos) -> Int {
        var frames = completedByReply[reply, default: 0]
        if let (head, partial) = partialFrames(at: now), head.reply == reply { frames += partial }
        return frames
    }

    public func playedMs(at now: Nanos) -> Int { playedFrames(at: now) * 1000 / sampleRate }

    public func playedMs(reply: String, at now: Nanos) -> Int { playedFrames(reply: reply, at: now) * 1000 / sampleRate }

    /// Drops everything queued (the player was stopped). Played counts stay until `forget`.
    public mutating func flush(at now: Nanos) {
        if let (head, partial) = partialFrames(at: now) {
            completedFrames += partial
            if let reply = head.reply { completedByReply[reply, default: 0] += partial }
        }
        inFlight.removeAll()
        headStartedAt = nil
    }

    /// Stops tracking a reply whose `played_ms` was reported.
    public mutating func forget(reply: String) {
        completedByReply[reply] = nil
    }

    public mutating func reset() {
        scheduledFrames = 0
        completedFrames = 0
        inFlight.removeAll()
        headStartedAt = nil
        completedByReply.removeAll()
    }
}

/// The post-playback microphone gate (PROTOCOL.md "Mic gate", research notes).
///
/// Apple's voice processing subtracts the device's own output from the microphone but leaves a tail at the end of each
/// utterance, loudest in quiet rooms; heard as the user, it makes the agent answer itself. So after `audio_end`, once
/// playback drains, the client ignores its microphone for 500-800 ms, and disarms the gate the moment new audio
/// arrives so barge-in still works. Push-to-talk clients skip the gate.
///
/// Ignored frames are replaced with digital silence rather than dropped, so the server's audio timeline (VAD, Smart
/// Turn's 8 s window) stays continuous.
///
/// `hard` is the documented behaviour and the default. `twoTier` lets a frame through when it is louder than a
/// threshold, which keeps a quick reply ("no, wait") that starts inside the window; the threshold has not been tuned
/// on real rooms yet (2026-10-05), so it is opt-in.
public struct MicGate: Sendable, Equatable {
    public enum Mode: String, Sendable, CaseIterable {
        case hard, twoTier
    }

    public var duration: Nanos
    public var mode: Mode
    public var openThresholdDBFS: Double
    public private(set) var deadline: Nanos?

    public init(durationMs: Int = 600, mode: Mode = .hard, openThresholdDBFS: Double = -38) {
        self.duration = .ms(durationMs)
        self.mode = mode
        self.openThresholdDBFS = openThresholdDBFS
    }

    public mutating func arm(at now: Nanos) { deadline = now + duration }

    public mutating func disarm() { deadline = nil }

    public func isArmed(at now: Nanos) -> Bool { deadline.map { now < $0 } ?? false }

    /// The frame to send, and whether the gate replaced it with silence.
    public mutating func filter(_ frame: Data, at now: Nanos) -> (frame: Data, gated: Bool) {
        guard let deadline else { return (frame, false) }
        if now >= deadline {
            self.deadline = nil
            return (frame, false)
        }
        if mode == .twoTier, PCM16.rmsDBFS(frame) >= openThresholdDBFS {
            self.deadline = nil
            return (frame, false)
        }
        return (Data(count: frame.count), true)
    }
}
