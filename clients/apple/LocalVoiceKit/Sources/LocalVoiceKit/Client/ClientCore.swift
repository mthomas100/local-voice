import Foundation

/// Starts and stops the audio engine on the client's behalf (the engine call blocks, so the facade runs it on its
/// own queue and reports back with `audioStatus`).
struct AudioControl: Sendable {
    var start: @Sendable () -> Void
    var stop: @Sendable () -> Void
}

/// The protocol v1 client state machine: turns, replies, barge-in, the mic gate, `played_ms`.
///
/// Everything runs on one serial queue: server messages and reply audio (in the order they arrived), capture chunks,
/// user actions and player completions. Ordering is the point: `audio_start` for the next reply must be seen before its
/// audio, and a local flush must happen before any message about it is sent (PROTOCOL.md "Barge-in").
final class ClientCore: @unchecked Sendable {
    // Invariant for @unchecked Sendable: confined to `queue`; closures that capture it only run there.
    private(set) var settings: ClientSettings
    private let queue: DispatchQueue
    private let clock: @Sendable () -> Nanos
    private let send: (Outgoing) -> Void
    private let emit: (ClientEvent) -> Void
    private let audio: AudioControl
    private let cue: (any CueDriving)?
    let playback: PlaybackController

    // Connection
    private(set) var ready = false

    // Talking
    private(set) var talkPhase: TalkPhase = .idle
    private(set) var handsFree = false
    var muted = false
    private var framer = PCMFramer()
    private var gate: MicGate
    private var gateWasArmed = false
    private var releaseGeneration = 0
    private var captureChunk: Nanos = .ms(20)
    private var reportedChunkMs = 0
    private var utterance = UtteranceStats()

    // Replies
    private(set) var serverActiveReply: String?
    private(set) var receivingReply: String?
    private var discardAudio = false
    private var discardedBytes = 0
    private var toolActive = false
    private var cueOn = false
    private var replyBytes: [String: (bytes: Int, chunks: Int, max: Int)] = [:]

    // Engine
    private(set) var audioRunning = false
    private var audioStarting = false
    private var chirpWhenRunning = false
    private var restartGeneration = 0
    private var restartDelay: Nanos = .ms(1500)
    private var lastActivity: Nanos

    // Latency of the current push-to-talk turn
    private var stopSentAt: Nanos?
    private var releasedAt: Nanos?
    private var latency: TurnLatency?

    // A space or mode switch waiting for the server's answer
    private var pendingSwitch: (request: SwitchRequest, at: Nanos)?
    /// How long a switch waits for the server's `space` (or an `error`) before it counts as unanswered. The M1
    /// orchestrator never answers one (it ignores `space` and `mode`, 2026-10-05); a server that does answers at once.
    static let switchAnswerWait: Nanos = .ms(5000)

    init(settings: ClientSettings, queue: DispatchQueue, clock: @escaping @Sendable () -> Nanos,
         player: any PlayerDriving, cue: (any CueDriving)?, audio: AudioControl,
         send: @escaping (Outgoing) -> Void, emit: @escaping (ClientEvent) -> Void) {
        self.settings = settings
        self.queue = queue
        self.clock = clock
        self.send = send
        self.emit = emit
        self.audio = audio
        self.cue = cue
        self.gate = settings.gate
        self.lastActivity = clock()
        self.playback = PlaybackController(player: player, queue: queue, clock: clock, prerollMs: settings.prerollMs,
                                           maxPrerollMs: settings.maxPrerollMs,
                                           maxPrerollWaitMs: settings.maxPrerollWaitMs)
        playback.onNotice = { [unowned self] notice in self.playbackNotice(notice) }
    }

    func updateSettings(_ new: ClientSettings) {
        settings = new
        gate = new.gate
    }

    // MARK: User actions

    /// Push-to-talk pressed. If the agent is speaking, this is a barge-in: flush locally first, then tell the server.
    func pressTalk() {
        guard talkPhase != .talking else { return }
        lastActivity = clock()
        if talkPhase == .releasing { finishRelease() }
        bargeIn()
        ensureAudio()
        framer.reset()
        utterance = UtteranceStats()
        stopSentAt = nil
        releasedAt = nil
        latency = nil
        talkPhase = .talking
        sendControl(.start)
        emit(.talk(.talking))
        if settings.pressChirp {
            if audioRunning { cue?.chirp() } else { chirpWhenRunning = true }
        }
    }

    /// Push-to-talk released. The last capture chunk can still be on its way (a tap delivers 20-100 ms at a time), so
    /// the tail and `stop` follow after one chunk period.
    func releaseTalk() {
        guard talkPhase == .talking else { return }
        talkPhase = .releasing
        releasedAt = clock()
        emit(.talk(.releasing))
        releaseGeneration += 1
        let gen = releaseGeneration
        let grace = min(max(captureChunk + .ms(10), .ms(30)), .ms(150))
        queue.asyncAfter(deadline: .now() + .nanoseconds(Int(grace))) { [weak self] in
            guard let self, gen == self.releaseGeneration, self.talkPhase == .releasing else { return }
            self.finishRelease()
        }
    }

    private func finishRelease() {
        if let partial = framer.flushPadded() {
            sendAudio(partial)
            utterance.frames += 1
            utterance.speechMs += 20
        }
        // About 100 ms of silence before stop, so the last word is through the server's VAD (PROTOCOL.md).
        for frame in framer.silenceFrames(covering: ProtocolV1.pushToTalkTail) {
            sendAudio(frame)
            utterance.tailMs += ProtocolV1.captureFrameDuration.milliseconds
        }
        sendControl(.stop)
        stopSentAt = clock()
        lastActivity = clock()
        talkPhase = .idle
        emit(.talk(.idle))
        emit(.utterance(utterance))
    }

    func startHandsFree() {
        guard !handsFree else { return }
        handsFree = true
        lastActivity = clock()
        framer.reset()
        gate.disarm()
        ensureAudio()
        emit(.handsFree(true))
    }

    func stopHandsFree() {
        guard handsFree else { return }
        handsFree = false
        framer.reset()
        lastActivity = clock()
        emit(.handsFree(false))
    }

    /// The user tapped stop while the agent spoke.
    func stopSpeaking() {
        lastActivity = clock()
        bargeIn()
    }

    func sendText(_ text: String) {
        lastActivity = clock()
        bargeIn()
        sendControl(.text(text))
    }

    func sendControl(_ message: ClientMessage) {
        send(.message(message))
        emit(.sent(message))
    }

    /// Asks the server for another space or mode; the outcome follows as `switchOutcome` (a newer request replaces a
    /// pending one).
    func requestSwitch(_ request: SwitchRequest) {
        lastActivity = clock()
        pendingSwitch = (request, clock())
        emit(.switchRequested(request))
        sendControl(request.kind == .space ? .space(name: request.name) : .mode(name: request.name))
    }

    // MARK: Capture

    func captured(_ samples: [Int16], at now: Nanos) {
        guard !samples.isEmpty else { return }
        captureChunk = Nanos(samples.count) * 1_000_000_000 / Nanos(ProtocolV1.captureSampleRate)
        let chunkMs = samples.count * 1000 / ProtocolV1.captureSampleRate
        if abs(chunkMs - reportedChunkMs) > 2 {  // resampling makes sizes wobble by a sample or two
            reportedChunkMs = chunkMs
            emit(.captureChunk(ms: chunkMs))
        }
        switch settings.mic {
        case .ptt:
            guard talkPhase != .idle else { return }
            for frame in framer.append(samples) {
                sendAudio(muted ? Data(count: frame.count) : frame)
                utterance.frames += 1
                utterance.speechMs += ProtocolV1.captureFrameDuration.milliseconds
                utterance.peakDBFS = max(utterance.peakDBFS, PCM16.rmsDBFS(frame))
            }
        case .vad:
            guard handsFree else { return }
            for frame in framer.append(samples) {
                if muted {
                    sendAudio(Data(count: frame.count))
                    continue
                }
                let (out, _) = gate.filter(frame, at: now)
                noteGate(at: now)
                sendAudio(out)
            }
        }
    }

    private func sendAudio(_ frame: Data) { send(.audio(frame)) }

    // MARK: Server

    func connectionEvent(_ event: ConnectionEvent) {
        switch event {
        case let .state(state):
            if case .ready = state {
                ready = true
            } else if ready {
                ready = false
                connectionLost()
            }
            emit(.connection(state))
        case let .message(message):
            serverMessage(message)
        case let .audio(data):
            serverAudio(data)
        case let .malformed(text, error):
            emit(.malformed("\(error): \(text.prefix(200))"))
        }
    }

    private func connectionLost() {
        // Audio in flight is lost; tell the next session what was heard, and restart a turn the user is still holding.
        for (reply, ms) in playback.flushAll() { sendControl(.playedMs(replyID: reply, ms: ms)) }
        serverActiveReply = nil
        receivingReply = nil
        discardAudio = false
        setCue(false)
        if talkPhase == .talking { sendControl(.start) }
    }

    private func serverMessage(_ message: ServerMessage) {
        let now = clock()
        // Reported first, so a log reads "received interrupt, sent played_ms".
        emit(.server(message))
        switch message {
        case let .audioStart(reply, _):
            lastActivity = now
            serverActiveReply = reply
            receivingReply = reply
            discardAudio = false
            disarmGate()
            replyBytes[reply] = (0, 0, 0)
            playback.audioStart(reply: reply)
            ensureAudio()
            setCue(false)
            if let stop = stopSentAt, latency == nil {
                latency = TurnLatency(reply: reply, stopToAudioStartMs: Int((now - stop) / 1_000_000))
            }
        case let .audioEnd(reply):
            if receivingReply == reply { receivingReply = nil }
            playback.audioEnd(reply: reply)
        case let .interrupt(reply):
            // The server stopped the reply: flush before anything else, then report what was heard. The channel is
            // ordered, so no more audio of that reply follows; nothing needs discarding.
            for (r, ms) in playback.flushAll() { sendControl(.playedMs(replyID: r, ms: ms)) }
            if reply == nil || reply == serverActiveReply { serverActiveReply = nil }
            receivingReply = nil
            setCue(false)
        case let .endOfTurn(reply):
            if reply == nil || reply == serverActiveReply { serverActiveReply = nil }
            if receivingReply == reply { receivingReply = nil }
            stopSentAt = nil
            releasedAt = nil
        case let .tool(event):
            switch event.phase {
            case .start: toolActive = true
            case .end: toolActive = false
            default: break
            }
            setCue(toolActive && playback.isIdle)
        case let .space(info):
            guard let pending = pendingSwitch else { break }
            pendingSwitch = nil
            let request = pending.request
            let now = request.kind == .space ? info.name : (info.mode ?? "")
            emit(.switchOutcome(now == request.name ? .switched(request, info)
                                : .refused(request, reason: now.isEmpty ? "no \(request.kind.rawValue) in the answer"
                                                                         : "the server stayed in \(now)")))
        case let .error(code, message):
            // An error while a switch waits is taken as its answer: the protocol gives the request no id.
            guard let pending = pendingSwitch else { break }
            pendingSwitch = nil
            emit(.switchOutcome(.refused(pending.request, reason: message.isEmpty ? code : message)))
        default:
            break
        }
    }

    private func serverAudio(_ data: Data) {
        lastActivity = clock()
        if discardAudio {
            discardedBytes += data.count
            emit(.discardedAudio(bytes: data.count))
            return
        }
        disarmGate()
        if let reply = receivingReply, var stats = replyBytes[reply] {
            stats.bytes += data.count
            stats.chunks += 1
            stats.max = max(stats.max, data.count)
            replyBytes[reply] = stats
        }
        ensureAudio()
        setCue(false)
        playback.enqueue(data, reply: receivingReply)
    }

    /// A local flush of everything the agent is saying (push-to-talk pressed, stop tapped, typed input).
    private func bargeIn() {
        guard !playback.isIdle || serverActiveReply != nil || playback.hasOpenReplies else { return }
        let flushed = playback.flushAll()
        if let active = serverActiveReply { sendControl(.interrupt(replyID: active)) }
        for (reply, ms) in flushed { sendControl(.playedMs(replyID: reply, ms: ms)) }
        // The server keeps sending until it reads our interrupt; drop that audio until the next reply starts.
        if serverActiveReply != nil { discardAudio = true }
        serverActiveReply = nil
        receivingReply = nil
        setCue(false)
    }

    private func playbackNotice(_ notice: PlaybackNotice) {
        let now = clock()
        switch notice {
        case let .started(reply, _):
            if let stop = stopSentAt, var l = latency, l.reply == reply, l.stopToPlaybackMs == nil {
                l.stopToPlaybackMs = Int((now - stop) / 1_000_000)
                l.releaseToPlaybackMs = releasedAt.map { Int((now - $0) / 1_000_000) }
                latency = l
                emit(.latency(l))
            }
        case let .finished(reply, ms, interrupted):
            if let stats = replyBytes.removeValue(forKey: reply) {
                emit(.replyAudio(reply: reply, bytes: stats.bytes, chunks: stats.chunks, maxChunkBytes: stats.max))
            }
            if !interrupted {
                // End of the reply: report what was played, then arm the mic gate once nothing else is queued.
                sendControl(.playedMs(replyID: reply, ms: ms))
                if playback.isIdle && receivingReply == nil {
                    armGate(at: now)
                    setCue(toolActive)
                }
            }
        case .underrun:
            break
        }
        emit(.playback(notice))
    }

    // MARK: Mic gate

    private func armGate(at now: Nanos) {
        guard settings.mic == .vad else { return }  // push-to-talk clients skip the gate
        gate.arm(at: now)
        noteGate(at: now)
    }

    private func disarmGate() {
        gate.disarm()
        noteGate(at: clock())
    }

    private func noteGate(at now: Nanos) {
        let armed = gate.isArmed(at: now)
        if armed != gateWasArmed {
            gateWasArmed = armed
            emit(.gate(armed: armed))
        }
    }

    private func setCue(_ on: Bool) {
        guard on != cueOn else { return }
        cueOn = on
        cue?.setActive(on)
    }

    // MARK: Engine

    func warmUp() {
        lastActivity = clock()
        ensureAudio()
    }

    private func ensureAudio() {
        guard !audioRunning, !audioStarting else { return }
        audioStarting = true
        emit(.audioEngine(.starting))
        audio.start()
    }

    func audioStatus(_ status: AudioEngineStatus) {
        switch status {
        case .running:
            audioRunning = true
            audioStarting = false
            restartDelay = .ms(1500)
            playback.setCanPlay(true)
            if chirpWhenRunning && talkPhase == .talking { cue?.chirp() }
            chirpWhenRunning = false
        case .stopped, .failed:
            audioRunning = false
            audioStarting = false
            playback.setCanPlay(false)
            for (reply, ms) in playback.flushAll() { sendControl(.playedMs(replyID: reply, ms: ms)) }
            if handsFree { scheduleAudioRestart() }
        case .starting:
            break
        }
        emit(.audioEngine(status))
    }

    /// A hands-free session whose engine stopped under it (a route change, an interruption such as a phone call)
    /// gets it back: retry after 1.5 s, doubling to at most 6 s while it keeps failing.
    private func scheduleAudioRestart() {
        restartGeneration += 1
        let gen = restartGeneration
        let delay = restartDelay
        restartDelay = min(restartDelay * 2, .ms(6000))
        queue.asyncAfter(deadline: .now() + .nanoseconds(Int(delay))) { [weak self] in
            guard let self, gen == self.restartGeneration, self.handsFree, !self.audioRunning else { return }
            self.ensureAudio()
        }
    }

    /// Periodic housekeeping: gate expiry for the log, and stopping an idle engine.
    func tick(at now: Nanos) {
        noteGate(at: now)
        if let pending = pendingSwitch, now - pending.at > Self.switchAnswerWait {
            pendingSwitch = nil
            emit(.switchOutcome(.noAnswer(pending.request)))
        }
        guard let idle = settings.engineIdleStopSeconds, audioRunning, talkPhase == .idle, !handsFree,
              playback.isIdle, receivingReply == nil, serverActiveReply == nil, !toolActive,
              now - lastActivity > Nanos(idle) * 1_000_000_000
        else { return }
        audioRunning = false
        playback.setCanPlay(false)
        audio.stop()
        emit(.audioEngine(.stopped))
    }
}
