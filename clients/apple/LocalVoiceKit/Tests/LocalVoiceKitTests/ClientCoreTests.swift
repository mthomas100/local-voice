import Foundation
import Synchronization
import Testing
@testable import LocalVoiceKit

final class TestClock: Sendable {
    let value = Mutex<Nanos>(1_000_000_000)
    var now: Nanos { value.withLock { $0 } }
    func advance(ms: Int) { value.withLock { $0 += .ms(ms) } }
}

/// A player whose buffers finish only when the test says so.
final class FakePlayer: PlayerDriving, @unchecked Sendable {
    private let lock = NSLock()
    private var completions: [(frames: Int, done: @Sendable () -> Void)] = []
    private(set) var scheduledFrames: [Int] = []
    private(set) var flushes = 0

    func schedule(_ pcm16: Data, completion: @escaping @Sendable () -> Void) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let frames = pcm16.count / 2
        scheduledFrames.append(frames)
        completions.append((frames, completion))
        return frames
    }

    func flush() {
        lock.lock()
        let pending = completions
        completions.removeAll()
        flushes += 1
        lock.unlock()
        pending.forEach { $0.done() }  // AVAudioPlayerNode calls the handlers of flushed buffers too
    }

    /// Finishes the oldest buffer.
    func completeNext() {
        lock.lock()
        let next = completions.isEmpty ? nil : completions.removeFirst()
        lock.unlock()
        next?.done()
    }

    var queued: Int {
        lock.lock()
        defer { lock.unlock() }
        return completions.count
    }
}

final class FakeCue: CueDriving, @unchecked Sendable {
    let states = Mutex<[Bool]>([])
    let chirps = Mutex(0)
    func setActive(_ active: Bool) { states.withLock { $0.append(active) } }
    func chirp() { chirps.withLock { $0 += 1 } }
}

/// A ClientCore on a private queue, with everything it does recorded.
final class CoreHarness: @unchecked Sendable {
    let queue = DispatchQueue(label: "core-test")
    let clock = TestClock()
    let player = FakePlayer()
    let cue = FakeCue()
    let sent = Mutex<[Outgoing]>([])
    let events = Mutex<[ClientEvent]>([])
    let audioStarts = Mutex(0)
    var core: ClientCore!

    init(settings: ClientSettings) {
        let clock = self.clock
        core = ClientCore(
            settings: settings, queue: queue, clock: { clock.now }, player: player, cue: cue,
            audio: AudioControl(start: { [unowned self] in self.audioStarts.withLock { $0 += 1 } }, stop: {}),
            send: { [unowned self] o in self.sent.withLock { $0.append(o) } },
            emit: { [unowned self] e in self.events.withLock { $0.append(e) } })
    }

    func run(_ body: (ClientCore) -> Void) { queue.sync { body(core) } }

    /// Lets queued work (completions hop onto the queue) run.
    func settle() { queue.sync {} }

    func server(_ m: ServerMessage) { run { $0.connectionEvent(.message(m)) } }
    func audio(_ chunks: Int, bytes: Int = 1920) {
        for _ in 0..<chunks { run { $0.connectionEvent(.audio(Data(count: bytes))) } }
    }

    func complete(_ n: Int) {
        for _ in 0..<n { player.completeNext() }
        settle()
    }

    var controls: [ClientMessage] {
        sent.withLock { $0.compactMap { if case let .message(m) = $0 { m } else { nil } } }
    }

    var audioFrames: [Data] {
        sent.withLock { $0.compactMap { if case let .audio(d) = $0 { d } else { nil } } }
    }

    var sentKinds: [String] {
        sent.withLock { $0.map { if case let .message(m) = $0 { m.type } else { "audio" } } }
    }

    func clearSent() { sent.withLock { $0.removeAll() } }

    var notices: [PlaybackNotice] {
        events.withLock { $0.compactMap { if case let .playback(n) = $0 { n } else { nil } } }
    }
}

@Suite("Client state machine")
struct ClientCoreTests {
    static let ptt = ClientSettings(mic: .ptt, prerollMs: 100, maxPrerollWaitMs: 60)
    static let vad = ClientSettings(mic: .vad, gate: MicGate(durationMs: 600), prerollMs: 100, maxPrerollWaitMs: 60)

    func running(_ settings: ClientSettings) -> CoreHarness {
        let h = CoreHarness(settings: settings)
        h.run { $0.audioStatus(.running("test")) }
        return h
    }

    @Test("push-to-talk: start, 20 ms frames, a padded partial, 100 ms of silence, stop")
    func pushToTalkTurn() async throws {
        let h = CoreHarness(settings: Self.ptt)
        h.run { $0.pressTalk() }
        #expect(h.audioStarts.withLock { $0 } == 1, "the engine starts on the press")
        for _ in 0..<3 { h.run { $0.captured([Int16](repeating: 1000, count: 1024), at: h.clock.now) } }
        h.run { $0.releaseTalk() }
        try await Task.sleep(for: .milliseconds(250))
        h.settle()
        // 3072 samples: 9 full frames + a padded partial, then 5 frames (100 ms) of silence.
        #expect(h.sentKinds == ["start"] + Array(repeating: "audio", count: 15) + ["stop"])
        let frames = h.audioFrames
        #expect(frames.allSatisfy { $0.count == 640 })
        #expect(frames.suffix(5).allSatisfy { $0.allSatisfy { $0 == 0 } }, "the tail is digital silence")
        #expect(!frames[9].allSatisfy { $0 == 0 }, "the partial frame keeps its speech")
        let stats = h.events.withLock { $0.compactMap { if case let .utterance(u) = $0 { u } else { nil } } }
        #expect(stats.first?.tailMs == 100)
    }

    @Test("capture outside a push-to-talk turn is not sent")
    func idleCaptureDropped() {
        let h = CoreHarness(settings: Self.ptt)
        h.run { $0.captured([Int16](repeating: 1000, count: 1024), at: h.clock.now) }
        #expect(h.sent.withLock { $0.isEmpty })
    }

    @Test("a reply plays after the pre-roll and played_ms reports it at the end")
    func replyPlaysAndReports() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(2)  // 80 ms: under the 100 ms pre-roll
        #expect(h.player.scheduledFrames.isEmpty)
        h.audio(1)  // 120 ms: start
        #expect(h.player.scheduledFrames == [960, 960, 960])
        h.audio(2)
        #expect(h.player.scheduledFrames.count == 5, "after the start, chunks go straight to the player")
        h.server(.audioEnd(replyID: "r1"))
        h.complete(5)
        #expect(h.controls.last == .playedMs(replyID: "r1", ms: 200))
        #expect(h.notices.contains(.started(reply: "r1", prerollMs: 120)))
        #expect(h.notices.contains(.finished(reply: "r1", playedMs: 200, interrupted: false)))
    }

    @Test("a short reply that ends inside the pre-roll still plays")
    func shortReply() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(1)
        h.server(.audioEnd(replyID: "r1"))
        #expect(h.player.scheduledFrames == [960])
        h.complete(1)
        #expect(h.controls.last == .playedMs(replyID: "r1", ms: 40))
    }

    @Test("pre-roll gives up waiting after maxPrerollWait")
    func prerollTimeout() async throws {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(1)
        h.clock.advance(ms: 100)
        try await Task.sleep(for: .milliseconds(150))
        h.settle()
        #expect(h.player.scheduledFrames == [960])
    }

    @Test("server interrupt: flush locally first, then played_ms with what was heard")
    func serverInterrupt() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(5)
        h.complete(2)  // 80 ms played
        h.clock.advance(ms: 20)
        h.server(.interrupt(replyID: "r1"))
        #expect(h.player.flushes == 1)
        #expect(h.controls.last == .playedMs(replyID: "r1", ms: 100))
        #expect(!h.controls.contains(.interrupt(replyID: "r1")), "the client does not echo a server interrupt")
        // The next reply plays normally.
        h.server(.audioStart(replyID: "r2", rate: 24_000))
        h.audio(3)
        #expect(h.player.scheduledFrames.count == 8)
    }

    @Test("push-to-talk over a reply: flush, interrupt, played_ms, start; late audio of the old reply is dropped")
    func clientBargeIn() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(5)
        h.complete(1)
        h.clearSent()
        h.run { $0.pressTalk() }
        #expect(h.player.flushes == 1)
        #expect(h.controls == [.interrupt(replyID: "r1"), .playedMs(replyID: "r1", ms: 40), .start])
        let scheduled = h.player.scheduledFrames.count
        h.audio(3)  // in flight before the server read our interrupt
        #expect(h.player.scheduledFrames.count == scheduled)
        h.server(.audioStart(replyID: "r2", rate: 24_000))
        h.audio(3)
        #expect(h.player.scheduledFrames.count == scheduled + 3)
    }

    @Test("stop tapped while the agent speaks")
    func stopTapped() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(4)
        h.clearSent()
        h.run { $0.stopSpeaking() }
        #expect(h.controls == [.interrupt(replyID: "r1"), .playedMs(replyID: "r1", ms: 0)])
    }

    @Test("an acknowledgement reply and the answer queue back to back, each with its own played_ms")
    func backToBackReplies() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "ack", rate: 24_000))
        h.audio(3)
        h.server(.audioEnd(replyID: "ack"))
        h.server(.endOfTurn(replyID: "ack"))
        h.server(.audioStart(replyID: "r2", rate: 24_000))
        h.audio(2)
        #expect(h.player.scheduledFrames.count == 5, "r2 queues behind the ack without a second pre-roll")
        h.complete(3)
        #expect(h.controls.last == .playedMs(replyID: "ack", ms: 120))
        h.server(.audioEnd(replyID: "r2"))
        h.complete(2)
        #expect(h.controls.last == .playedMs(replyID: "r2", ms: 80))
    }

    @Test("open mic: the gate silences the microphone after playback drains, and new audio disarms it")
    func micGate() {
        let h = running(Self.vad)
        h.run { $0.startHandsFree() }
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.server(.audioEnd(replyID: "r1"))
        h.complete(3)
        let speech = [Int16](repeating: 5000, count: 320)
        h.clearSent()
        h.clock.advance(ms: 100)
        h.run { $0.captured(speech, at: h.clock.now) }
        #expect(h.audioFrames.last?.allSatisfy { $0 == 0 } == true, "inside 600 ms: silence")
        h.clock.advance(ms: 600)
        h.run { $0.captured(speech, at: h.clock.now) }
        #expect(h.audioFrames.last == PCM16.data(speech), "after the window: the microphone again")
        // Armed again, then disarmed by the next reply's audio.
        h.server(.audioStart(replyID: "r2", rate: 24_000))
        h.audio(3)
        h.server(.audioEnd(replyID: "r2"))
        h.complete(3)
        h.server(.audioStart(replyID: "r3", rate: 24_000))
        h.run { $0.captured(speech, at: h.clock.now) }
        #expect(h.audioFrames.last == PCM16.data(speech), "new audio disarms the gate so barge-in works")
        let gateEvents = h.events.withLock { $0.compactMap { if case let .gate(a) = $0 { a } else { nil } } }
        #expect(gateEvents.starts(with: [true, false, true, false]))
    }

    @Test("push-to-talk skips the gate")
    func pushToTalkSkipsGate() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.server(.audioEnd(replyID: "r1"))
        h.complete(3)
        h.run { $0.pressTalk() }
        let speech = [Int16](repeating: 5000, count: 320)
        h.run { $0.captured(speech, at: h.clock.now) }
        #expect(h.audioFrames.last == PCM16.data(speech))
    }

    @Test("late audio (an underrun the largest pre-roll would have covered) adds 40 ms of pre-roll")
    func underrun() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.complete(3)
        h.clock.advance(ms: 50)
        h.audio(3)  // the audio comes back 50 ms late: 120 ms < 140 ms, buffering again
        #expect(h.notices.contains(.underrun(reply: "r1", gapMs: 50, resumed: true, prerollMs: 140)))
        #expect(h.player.scheduledFrames.count == 3)
        h.audio(1)
        #expect(h.player.scheduledFrames.count == 7)
    }

    /// The real orchestrator holds a reply back at the first sound of the user's speech and sends `interrupt` when its
    /// VAD confirms (80-90 ms later in the 2026-10-05 run): the player runs dry in between. Not the network.
    @Test("a reply cut while the player is dry (the server paused for a barge-in) leaves the pre-roll alone")
    func dryUntilInterrupt() {
        let h = running(Self.vad)
        h.run { $0.startHandsFree() }
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.complete(3)
        h.clock.advance(ms: 83)
        h.server(.interrupt(replyID: "r1"))
        #expect(h.notices.contains(.underrun(reply: "r1", gapMs: 83, resumed: false, prerollMs: 100)))
        h.server(.audioStart(replyID: "r2", rate: 24_000))
        h.audio(2)
        #expect(h.player.scheduledFrames.count == 3, "80 ms < 100 ms: the next reply waits for the usual pre-roll only")
        h.audio(1)
        #expect(h.player.scheduledFrames.count == 6)
    }

    @Test("a pause longer than the largest pre-roll (a sentence or a tool) and an end marker after the audio leave it alone")
    func serverPauses() {
        let h = running(Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.complete(3)
        h.clock.advance(ms: 900)  // the LLM writes the next sentence
        h.audio(3)
        #expect(h.notices.contains(.underrun(reply: "r1", gapMs: 900, resumed: true, prerollMs: 100)))
        #expect(h.player.scheduledFrames.count == 6, "120 ms >= 100 ms: plays at the usual pre-roll")
        h.complete(3)
        h.clock.advance(ms: 20)
        h.server(.audioEnd(replyID: "r1"))
        #expect(h.notices.contains(.underrun(reply: "r1", gapMs: 20, resumed: false, prerollMs: 100)))
        #expect(h.notices.contains(.finished(reply: "r1", playedMs: 240, interrupted: false)))
        #expect(h.controls.last == .playedMs(replyID: "r1", ms: 240))
    }

    @Test("a reply that plays through without running dry gives 20 ms of the extra pre-roll back")
    func prerollRecovers() {
        let h = running(Self.ptt)
        func preroll() -> Int {
            var frames = 0
            h.run { frames = $0.playback.prerollFrames }
            return frames * 1000 / 24_000
        }
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.complete(3)  // ran dry
        h.audio(4)
        h.server(.audioEnd(replyID: "r1"))
        h.complete(4)
        #expect(preroll() == 140, "the reply that ran dry gives nothing back")
        var after: [Int] = []
        for id in ["r2", "r3", "r4"] {
            h.server(.audioStart(replyID: id, rate: 24_000))
            h.audio(4)
            h.server(.audioEnd(replyID: id))
            h.complete(4)
            after.append(preroll())
        }
        #expect(after == [120, 100, 100], "20 ms back per clean reply, down to the configured 100 ms")
    }

    @Test("losing the connection mid-press restarts the turn on the next session")
    func reconnectWhileTalking() {
        let h = running(Self.ptt)
        h.run { $0.connectionEvent(.state(.ready(session: "s1"))) }
        h.run { $0.pressTalk() }
        h.clearSent()
        h.run {
            $0.connectionEvent(.state(.waiting(nextAttempt: 1, delay: .milliseconds(500),
                                               after: CloseReport(code: 1006, reason: ""))))
        }
        #expect(h.controls == [.start])
    }

    @Test("the working sound plays while a tool runs and nothing is said")
    func cue() {
        let h = running(Self.ptt)
        h.server(.tool(ToolEvent(phase: .start, name: "read", label: "reading")))
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(3)
        h.server(.audioEnd(replyID: "r1"))
        h.complete(3)
        h.server(.tool(ToolEvent(phase: .end, name: "read", ok: true)))
        #expect(h.cue.states.withLock { $0 } == [true, false, true, false])
    }

    @Test("client-side latency from stop (and from the release) to audio_start and to playback")
    func latency() async throws {
        let h = running(Self.ptt)
        h.run { $0.pressTalk() }
        h.run { $0.captured([Int16](repeating: 1000, count: 640), at: h.clock.now) }
        h.run { $0.releaseTalk() }
        h.clock.advance(ms: 35)  // before the grace for the last capture chunk ends (30 ms or more, real time)
        try await Task.sleep(for: .milliseconds(200))
        h.settle()
        h.clock.advance(ms: 700)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(2)
        h.clock.advance(ms: 40)
        h.audio(1)
        let latency = h.events.withLock { $0.compactMap { if case let .latency(l) = $0 { l } else { nil } } }
        #expect(latency == [TurnLatency(reply: "r1", stopToAudioStartMs: 700, stopToPlaybackMs: 740,
                                        releaseToPlaybackMs: 775)])
    }

    @Test("a press chirps once the microphone is live: at once when warm, after the engine starts when cold")
    func pressChirp() {
        var settings = Self.ptt
        settings.pressChirp = true
        let cold = CoreHarness(settings: settings)
        cold.run { $0.pressTalk() }
        #expect(cold.cue.chirps.withLock { $0 } == 0, "not before the engine runs")
        cold.run { $0.audioStatus(.running("test")) }
        #expect(cold.cue.chirps.withLock { $0 } == 1)
        let warm = CoreHarness(settings: settings)
        warm.run { $0.audioStatus(.running("test")) }
        warm.run { $0.pressTalk() }
        #expect(warm.cue.chirps.withLock { $0 } == 1)
        let off = running(Self.ptt)
        off.run { $0.pressTalk() }
        #expect(off.cue.chirps.withLock { $0 } == 0, "off unless asked for")
    }

    @Test("a hands-free session restarts an engine that stopped under it; push-to-talk waits for the next press")
    func handsFreeRestart() async throws {
        let h = running(Self.vad)
        h.run { $0.startHandsFree() }
        h.run { $0.audioStatus(.failed("the audio route or device changed")) }
        #expect(h.audioStarts.withLock { $0 } == 0)
        try await Task.sleep(for: .milliseconds(1700))
        h.settle()
        #expect(h.audioStarts.withLock { $0 } == 1)
        let p = running(Self.ptt)
        p.run { $0.audioStatus(.failed("the audio route or device changed")) }
        try await Task.sleep(for: .milliseconds(1700))
        p.settle()
        #expect(p.audioStarts.withLock { $0 } == 0)
    }

    @Test("audio waits for the engine, then plays")
    func waitsForEngine() {
        let h = CoreHarness(settings: Self.ptt)
        h.server(.audioStart(replyID: "r1", rate: 24_000))
        h.audio(4)
        #expect(h.player.scheduledFrames.isEmpty)
        #expect(h.audioStarts.withLock { $0 } == 1, "a reply arriving starts the engine")
        h.run { $0.audioStatus(.running("test")) }
        #expect(h.player.scheduledFrames.count == 4)
    }
}
