import Foundation

/// What playback needs from a player node. The real one wraps `AVAudioPlayerNode`; tests use a fake.
public protocol PlayerDriving: AnyObject, Sendable {
    /// Schedules 24 kHz PCM16 audio behind whatever is queued. `completion` runs on any thread once the buffer has
    /// played, or at once if it is flushed. Returns the frames scheduled.
    @discardableResult
    func schedule(_ pcm16: Data, completion: @escaping @Sendable () -> Void) -> Int
    /// Silences the output now and drops everything scheduled; the player stays ready for new buffers.
    func flush()
}

/// The working sound a client plays while a tool runs (Pipecat research notes: native clients play their own loop, which their
/// own engine can duck, instead of Pipecat's mixer).
public protocol CueDriving: AnyObject, Sendable {
    func setActive(_ active: Bool)
    /// A short blip: the microphone is live (push-to-talk pressed, engine running).
    func chirp()
}

public enum PlaybackNotice: Sendable, Equatable {
    /// The first audio of a reply went to the player, after `prerollMs` of buffering.
    case started(reply: String?, prerollMs: Int)
    case finished(reply: String, playedMs: Int, interrupted: Bool)
    /// The player ran dry before the reply's `audio_end`, for `gapMs`, until the reply's audio came back (`resumed`)
    /// or the reply was cut or ended without any. `prerollMs` is the pre-roll after it (it grows only for late audio).
    case underrun(reply: String?, gapMs: Int, resumed: Bool, prerollMs: Int)
}

/// Turns the reply audio stream into scheduled player buffers, and counts what was heard.
///
/// Runs on the client's serial queue; player completions hop back onto it, so all player-node calls come from one
/// queue (AVAudioPlayerNode asks for that when calls are made from its completion handlers).
///
/// Pre-roll: when the player is idle, audio is held until `prerollFrames` are queued (100 ms by default), the reply
/// ended, or `maxPrerollWait` passed, then scheduled in one go. The server paces audio at real time, at most one 40 ms
/// chunk ahead (Pipecat research notes), so without a cushion the phone's Wi-Fi jitter (7-227 ms measured, research notes) would starve the
/// player. Each underrun of late audio adds 40 ms of pre-roll, up to `maxPrerollFrames`, and each reply that plays
/// through without one gives 20 ms back, down to the configured pre-roll (2026-10-05): not every underrun is jitter.
/// An underrun counts only when the reply's audio comes back within a gap the largest pre-roll could have covered.
/// Against the real orchestrator (run 2026-10-05 14:34), both underruns were the server's pauses, not the network:
/// it holds a reply back at the first sound of the user's speech (its bargein.py), so the player runs dry until the
/// `interrupt`, and the TTS pauses between sentences while the LLM writes the next one (and for tools, for seconds).
/// Counting those made the answer to every open-mic barge-in start 40 ms later (pre-roll 120 ms instead of 80).
final class PlaybackController: @unchecked Sendable {
    // Invariant for @unchecked Sendable: confined to `queue`; closures that capture it only run there.
    private let player: any PlayerDriving
    private let queue: DispatchQueue
    private let clock: @Sendable () -> Nanos
    private(set) var ledger = PlaybackLedger()
    private(set) var prerollFrames: Int
    let basePrerollFrames: Int
    let maxPrerollFrames: Int
    let maxPrerollWait: Nanos
    var onNotice: (PlaybackNotice) -> Void = { _ in }

    private var pending: [(reply: String?, data: Data)] = []
    private var pendingFrames = 0
    private var pendingSince: Nanos?
    private var generation = 0
    /// Replies that had `audio_start` and whose `played_ms` is not reported yet, in order.
    private var openReplies: [String] = []
    private var endedReplies: Set<String> = []
    private var startedReplies: Set<String> = []
    private(set) var canPlay = false
    private(set) var underruns = 0
    /// Replies whose late audio grew the pre-roll (they give none back).
    private var underranReplies: Set<String> = []
    /// The reply whose audio ran out before its `audio_end`, and since when; `settleDry` decides what it was.
    private var dry: (reply: String, since: Nanos)?

    init(player: any PlayerDriving, queue: DispatchQueue, clock: @escaping @Sendable () -> Nanos,
         prerollMs: Int = 100, maxPrerollMs: Int = 300, maxPrerollWaitMs: Int = 400) {
        self.player = player
        self.queue = queue
        self.clock = clock
        self.prerollFrames = prerollMs * ProtocolV1.playbackSampleRate / 1000
        self.basePrerollFrames = self.prerollFrames
        self.maxPrerollFrames = maxPrerollMs * ProtocolV1.playbackSampleRate / 1000
        self.maxPrerollWait = .ms(maxPrerollWaitMs)
    }

    /// Nothing playing and nothing waiting.
    var isIdle: Bool { ledger.isDrained && pending.isEmpty }

    var hasOpenReplies: Bool { !openReplies.isEmpty }

    func setCanPlay(_ value: Bool) {
        canPlay = value
        if value { startIfReady(force: false) }
    }

    func audioStart(reply: String) {
        if !openReplies.contains(reply) { openReplies.append(reply) }
    }

    func enqueue(_ data: Data, reply: String?) {
        guard data.count >= 2 else { return }
        if let d = dry { settleDry(resumed: reply == nil || reply == d.reply) }
        if pending.isEmpty && !ledger.isDrained && canPlay {
            schedule(data, reply: reply)
            return
        }
        if pending.isEmpty {
            pendingSince = clock()
            let gen = generation
            queue.asyncAfter(deadline: .now() + .nanoseconds(Int(maxPrerollWait))) { [weak self] in
                guard let self, gen == self.generation else { return }
                self.startIfReady(force: false)
            }
        }
        pending.append((reply, data))
        pendingFrames += data.count / 2
        startIfReady(force: false)
    }

    func audioEnd(reply: String) {
        if dry?.reply == reply { settleDry(resumed: false) }  // all its audio had played; the end marker came after
        endedReplies.insert(reply)
        if !openReplies.contains(reply) { openReplies.append(reply) }
        startIfReady(force: pending.contains { $0.reply == reply })
        finishDrainedReplies()
    }

    /// Stops the sound now and drops everything queued. Returns the played time of every open reply, which the
    /// caller reports as `played_ms`.
    @discardableResult
    func flushAll() -> [(reply: String, playedMs: Int)] {
        settleDry(resumed: false)  // cut while dry: the server stopped sending because the user spoke
        let now = clock()
        generation += 1
        player.flush()
        ledger.flush(at: now)
        pending.removeAll()
        pendingFrames = 0
        pendingSince = nil
        let results = openReplies.map { ($0, ledger.playedMs(reply: $0, at: now)) }
        for reply in openReplies { ledger.forget(reply: reply) }
        openReplies.removeAll()
        endedReplies.removeAll()
        startedReplies.removeAll()
        underranReplies.removeAll()
        for (reply, ms) in results { onNotice(.finished(reply: reply, playedMs: ms, interrupted: true)) }
        return results
    }

    private func startIfReady(force: Bool) {
        guard !pending.isEmpty, canPlay else { return }
        let waited = pendingSince.map { clock() - $0 } ?? 0
        let lastEnded = pending.last?.reply.map { endedReplies.contains($0) } ?? false
        guard force || pendingFrames >= prerollFrames || lastEnded || waited >= maxPrerollWait else { return }
        let prerollMs = pendingFrames * 1000 / ProtocolV1.playbackSampleRate
        let items = pending
        pending.removeAll()
        pendingFrames = 0
        pendingSince = nil
        for (reply, data) in items {
            if let reply, !startedReplies.contains(reply) {
                startedReplies.insert(reply)
                onNotice(.started(reply: reply, prerollMs: prerollMs))
            }
            schedule(data, reply: reply)
        }
    }

    private func schedule(_ data: Data, reply: String?) {
        let gen = generation
        let frames = player.schedule(data) { [weak self] in
            guard let self else { return }
            self.queue.async { self.completed(generation: gen) }
        }
        ledger.scheduled(frames: frames, reply: reply, at: clock())
    }

    private func completed(generation gen: Int) {
        guard gen == generation else { return }
        ledger.completed(at: clock())
        if dry == nil, ledger.isDrained && pending.isEmpty, let open = openReplies.last, !endedReplies.contains(open) {
            underruns += 1
            dry = (open, clock())
        }
        finishDrainedReplies()
    }

    /// Ends a dry spell. Late audio (it `resumed` within the largest pre-roll) grows the pre-roll by 40 ms; a reply cut
    /// or ended while dry, or a longer pause in what the server had to say, leaves it as it is (see the type's comment).
    private func settleDry(resumed: Bool) {
        guard let d = dry else { return }
        dry = nil
        let now = clock()
        let gapMs = now > d.since ? Int((now - d.since) / 1_000_000) : 0
        if resumed && gapMs < maxPrerollFrames * 1000 / ProtocolV1.playbackSampleRate {
            underranReplies.insert(d.reply)
            prerollFrames = min(maxPrerollFrames, prerollFrames + ProtocolV1.playbackSampleRate / 25)
        }
        onNotice(.underrun(reply: d.reply, gapMs: gapMs, resumed: resumed,
                           prerollMs: prerollFrames * 1000 / ProtocolV1.playbackSampleRate))
    }

    private func finishDrainedReplies() {
        let now = clock()
        while let reply = openReplies.first, endedReplies.contains(reply), !ledger.hasInFlight(reply: reply),
              !pending.contains(where: { $0.reply == reply }) {
            openReplies.removeFirst()
            endedReplies.remove(reply)
            startedReplies.remove(reply)
            let ms = ledger.playedMs(reply: reply, at: now)
            ledger.forget(reply: reply)
            if underranReplies.remove(reply) == nil {
                prerollFrames = max(basePrerollFrames, prerollFrames - ProtocolV1.playbackSampleRate / 50)
            }
            onNotice(.finished(reply: reply, playedMs: ms, interrupted: false))
        }
    }
}
