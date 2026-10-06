import Foundation
import LocalVoiceKit
import Observation

/// The talk screen's model, shared by the iOS app and the Mac menu-bar app: one `VoiceClient`, its events folded into
/// observable state, and the actions the buttons and hotkey call.
@MainActor
@Observable
public final class VoiceSessionModel {
    public struct Entry: Identifiable, Equatable, Sendable {
        public enum Role: Sendable { case user, agent, notice }
        public let id: String
        public let role: Role
        public var text: String
        public var interrupted = false
    }

    public private(set) var entries: [Entry] = []
    public private(set) var partialTranscript: String?
    public private(set) var connection: ConnectionState = .idle
    public private(set) var talk: TalkPhase = .idle
    public private(set) var handsFree = false
    /// Space, mode, tier, model, state, hold, tool and latency: the one place the screens read them from.
    public private(set) var dashboard = Dashboard()
    /// The agent's questions waiting for an answer (PROTOCOL.md "Approvals"); the card shows the first.
    public private(set) var approvals = ApprovalQueue()
    public private(set) var engine: AudioEngineStatus?
    public private(set) var isPlaying = false
    public private(set) var problem: String?
    public private(set) var settings: AppSettings
    public let clientKind: ClientKind

    @ObservationIgnored public nonisolated let microphone: SyntheticMicrophone?
    @ObservationIgnored private var client: VoiceClient
    @ObservationIgnored private var eventTask: Task<Void, Never>?
    @ObservationIgnored private var sessionPolicy: LiveAudioIO.SessionPolicy
    @ObservationIgnored public let tally = EventTally()
    @ObservationIgnored public private(set) var log: EventLog?
    @ObservationIgnored private var micPermission: Bool?
    @ObservationIgnored private var entryCounter = 0
    /// Typed words are shown at once; a server that also sends them back as a transcript is not shown them twice.
    @ObservationIgnored private var typedAwaitingEcho: String?
    /// Called after every state change (the iPhone app drives its Live Activity from it).
    @ObservationIgnored public var onChange: (@MainActor (VoiceSessionModel) -> Void)?
    /// A script's `snapshot:<label>` step (unattended runs): the Mac app draws its panel.
    @ObservationIgnored public var onSnapshot: (@MainActor (String) async -> Void)?
    /// Closes a question whose time ran out when the server sends no `confirm_cancel` (one from before Approvals).
    @ObservationIgnored private var expiryTask: Task<Void, Never>?

    public init(settings: AppSettings, client kind: ClientKind, sessionPolicy: LiveAudioIO.SessionPolicy = .managed) {
        self.settings = settings
        self.clientKind = kind
        self.sessionPolicy = sessionPolicy
        let mic = settings.audio == .microphone ? nil : SyntheticMicrophone()
        self.microphone = mic
        self.micPermission = settings.audio == .microphone ? (MicrophonePermission.isGranted ? true : nil) : true
        self.client = Self.makeClient(settings: settings, kind: kind, microphone: mic, policy: sessionPolicy)
        self.log = try? settings.eventLogURL.map { try EventLog(url: $0) }
        log?.write([("event", "app"), ("client", .string(kind.rawValue)), ("url", .string(settings.serverURL.absoluteString)),
                    ("device", .string(settings.device)), ("audio", .string(settings.audio.rawValue))])
        listen()
    }

    private static func makeClient(settings: AppSettings, kind: ClientKind, microphone: SyntheticMicrophone?,
                                   policy: LiveAudioIO.SessionPolicy) -> VoiceClient {
        let audio: any AudioIO
        switch settings.audio {
        case .microphone:
            audio = LiveAudioIO(options: .init(capture: .microphone(voiceProcessing: true), sessionPolicy: policy))
        case .synthetic:
            audio = LiveAudioIO(options: .init(capture: .synthetic(microphone!), outputVolume: settings.outputVolume,
                                               sessionPolicy: policy))
        case .headless:
            audio = HeadlessAudioIO(microphone: microphone)
        }
        return VoiceClient(
            configuration: .init(url: settings.serverURL,
                                 hello: Hello(client: kind, device: settings.device, mic: settings.mic),
                                 settings: settings.clientSettings),
            audio: audio)
    }

    private func listen() {
        let client = self.client
        eventTask = Task { [weak self] in
            for await event in client.events {
                guard let self else { return }
                self.apply(event)
            }
        }
    }

    // MARK: Actions

    public func connect() { client.connect() }

    /// Press and release must reach the client in order, so neither waits for anything. The first press before the
    /// microphone permission is known only asks for it.
    public func pressTalk() {
        guard settings.audio != .microphone || micPermission == true else {
            Task { _ = await ensureMicrophone() }
            return
        }
        if settings.mic == .vad {
            client.stopSpeaking()  // hands-free: the button barges in; the open microphone carries the words
            return
        }
        client.pressTalk()
    }

    public func releaseTalk() {
        if settings.mic == .ptt { client.releaseTalk() }
    }

    /// Open-microphone session: the server finds the turns; barge-in by talking over the agent.
    public func setHandsFree(_ on: Bool) async {
        if on {
            guard await ensureMicrophone() else { return }
            if settings.mic != .vad { await setMicMode(.vad) }
            client.startHandsFree()
        } else {
            client.stopHandsFree()
            if settings.mic != .ptt { await setMicMode(.ptt) }
        }
    }

    public func stopSpeaking() { client.stopSpeaking() }

    /// Typed words go into the conversation at once: the orchestrator sends no transcript for them (it makes them the
    /// user's message directly, 2026-10-05).
    public func send(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        append(.user, trimmed)
        typedAwaitingEcho = trimmed
        client.sendText(trimmed)
        notify()
    }

    /// Asks the server for another space; the dashboard shows the answer (`Dashboard.switchLine`).
    public func switchSpace(_ name: String) {
        guard name != dashboard.space else { return }
        client.switchSpace(name)
    }

    /// `conversation` or `act`.
    public func switchMode(_ name: String) {
        guard name != dashboard.mode else { return }
        client.switchMode(name)
    }

    /// A status view opened: fetch `/v1/status` now rather than wait for the next turn.
    public func refreshStatus() { client.refreshStatus() }

    /// The question on the card, if any.
    public var pendingApproval: PendingApproval? { approvals.current }

    /// The person picked `choice` on a card (a tap; nothing ever answers by itself). False, with nothing sent, when the
    /// question is no longer waiting or has no such choice.
    @discardableResult
    public func answer(_ choice: String, to id: ScalarID) -> Bool {
        guard let answer = approvals.answer(id: id, choice: choice) else { return false }
        client.answer(answer)
        log?.write(answer.approval.logFields("answered", [("choice", .string(answer.choice.id)),
                                                          ("label", .string(answer.choice.label)),
                                                          ("confirmed", .bool(answer.choice.allows))]))
        tally.note(key: "approval-answered")
        append(.notice, "You chose “\(answer.choice.label)”: \(answer.approval.content.headline)")
        scheduleExpiry()
        notify()
        return true
    }

    public func setMuted(_ muted: Bool) { client.setMuted(muted) }

    public func warmUp() {
        Task {
            guard await ensureMicrophone() else { return }
            client.warmUpAudio()
        }
    }

    /// The server builds its pipeline from the hello's `mic`, so changing it is a new connection (same device, so
    /// the same agent session resumes).
    private func setMicMode(_ mic: MicMode) async {
        settings.mic = mic
        await client.reconfigure(hello: Hello(client: clientKind, device: settings.device, mic: mic),
                                 settings: settings.clientSettings)
    }

    /// Applies edited settings: a new client when the server or device changes.
    public func update(settings new: AppSettings) async {
        let rebuild = new.serverURL != settings.serverURL || new.device != settings.device || new.audio != settings.audio
        settings = new
        new.save()
        if rebuild {
            await rebuildClient()
        } else {
            await client.reconfigure(hello: Hello(client: clientKind, device: new.device, mic: new.mic),
                                     settings: new.clientSettings)
        }
        notify()
    }

    /// CallKit and the Push to Talk framework activate the iOS audio session themselves (`.external`); a new client
    /// is built because the engine's options are fixed at construction.
    public func setSessionPolicy(_ policy: LiveAudioIO.SessionPolicy) async {
        guard policy != sessionPolicy else { return }
        sessionPolicy = policy
        await rebuildClient()
    }

    private func rebuildClient() async {
        await client.shutdown()
        eventTask?.cancel()
        client = Self.makeClient(settings: settings, kind: clientKind, microphone: microphone, policy: sessionPolicy)
        connection = .idle
        handsFree = false
        talk = .idle
        listen()
        client.connect()
        notify()
    }

    public func stopAudio() async {
        await client.stopAudio()
    }

    public func disconnect() async {
        await client.disconnect()
    }

    public func shutdown() async {
        await client.shutdown()
        eventTask?.cancel()
        log?.close()
    }

    private func ensureMicrophone() async -> Bool {
        guard settings.audio == .microphone else { return true }
        if let granted = micPermission { return granted }
        let granted = await MicrophonePermission.request()
        micPermission = granted
        if !granted {
            problem = clientKind == .mac
                ? "Microphone access is off: System Settings > Privacy & Security > Microphone."
                : "Microphone access is off: Settings > Privacy & Security > Microphone."
            notify()
        }
        return granted
    }

    // MARK: Events

    private func apply(_ event: ClientEvent) {
        log?.record(event)
        tally.note(event)
        let before = (space: dashboard.space, mode: dashboard.mode)
        dashboard.apply(event)
        switch event {
        case let .connection(state):
            connection = state
            if case let .stopped(report?) = state { problem = "Disconnected: \(report)" }
            if case .ready = state { problem = nil }
            switch state {
            case .waiting, .stopped:
                closeApprovals(approvals.closeAll(), why: "connection lost", by: "client") {
                    "Question closed when the connection dropped: \($0.content.headline)"
                }
            default: break
            }
        case let .server(message):
            apply(message, before: before)
        case let .talk(phase):
            talk = phase
        case let .handsFree(on):
            handsFree = on
        case let .playback(notice):
            switch notice {
            case .started: isPlaying = true
            case .finished: isPlaying = false
            case .underrun: break
            }
        case .latency, .status, .statusFailed, .switchRequested, .switchOutcome:
            break  // the dashboard has them
        case let .audioEngine(status):
            engine = status
            if case let .failed(reason) = status { problem = "Audio: \(reason)" }
        case let .malformed(text):
            problem = "Server sent something unreadable: \(text.prefix(80))"
        case .sent, .utterance, .gate, .discardedAudio, .replyAudio, .captureChunk:
            return
        }
        notify()
    }

    private func apply(_ message: ServerMessage, before: (space: String?, mode: String?)) {
        switch message {
        case let .transcript(final, text):
            if final {
                partialTranscript = nil
                if let typed = typedAwaitingEcho, typed == text {
                    typedAwaitingEcho = nil
                    return
                }
                typedAwaitingEcho = nil
                append(.user, text)
            } else {
                partialTranscript = text
            }
        case let .replyText(reply, delta):
            if let i = entries.lastIndex(where: { $0.id == reply }) {
                entries[i].text += (entries[i].text.isEmpty || delta.hasPrefix(" ") ? "" : " ") + delta
            } else {
                entries.append(Entry(id: reply, role: .agent, text: delta))
            }
        case let .interrupt(reply):
            if let reply, let i = entries.lastIndex(where: { $0.id == reply }) { entries[i].interrupted = true }
            isPlaying = false
        case let .confirmRequest(c):
            let received = approvals.receive(c, at: Date())
            if let shown = approvals.waiting.first(where: { $0.id == c.id }) {
                log?.write(shown.logFields("shown", [("received", .string(received.rawValue))]))
            }
            tally.note(key: "approval-shown")
            scheduleExpiry()
        case let .confirmCancel(id, why):
            if let gone = approvals.cancel(id: id) {
                // The log keeps the server's code; the note says it in words, a timeout as the client's own does.
                let reason = ApprovalContent.withdrawal(why)
                closeApprovals([gone], why: why, by: "server") {
                    reason.timedOut ? "Not done (no answer in time): \($0.content.headline)"
                        : reason.words.isEmpty ? "Question withdrawn: \($0.content.headline)"
                        : "Question withdrawn (\(reason.words)): \($0.content.headline)"
                }
            }
        case let .space(s):
            // A notice only when something changed: the orchestrator also sends `space` right after every welcome.
            if before.space != nil, before.space != s.name || (s.mode != nil && before.mode != s.mode) {
                append(.notice, "Now in \(s.description ?? s.name)" + (s.mode.map { ", \($0) mode" } ?? ""))
            }
        case let .error(_, text):
            problem = text
        case .welcome, .state, .tool, .hold, .audioStart, .audioEnd, .endOfTurn, .pong, .unknown:
            break  // the dashboard has them
        }
    }

    /// Questions that left the screen without the person's answer: logged, and a note in the conversation saying why.
    private func closeApprovals(_ gone: [PendingApproval], why: String, by: String,
                                notice: (PendingApproval) -> String) {
        for approval in gone {
            log?.write(approval.logFields("closed", [("why", .string(why)), ("by", .string(by))]))
            tally.note(key: "approval-closed")
            append(.notice, notice(approval))
        }
        if !gone.isEmpty { scheduleExpiry() }
    }

    private func scheduleExpiry() {
        expiryTask?.cancel()
        guard let at = approvals.nextExpiry() else { return }
        expiryTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(max(0, at.timeIntervalSinceNow)))
            guard !Task.isCancelled, let self else { return }
            // Silence is no (PROTOCOL.md): the card goes, and nothing is sent.
            self.closeApprovals(self.approvals.expire(now: Date()), why: "no answer in time", by: "client") {
                "Not done (no answer in time): \($0.content.headline)"
            }
            self.notify()
        }
    }

    private func append(_ role: Entry.Role, _ text: String) {
        entryCounter += 1
        entries.append(Entry(id: "e\(entryCounter)", role: role, text: text))
        if entries.count > 200 { entries.removeFirst(entries.count - 200) }
    }

    private func notify() { onChange?(self) }

    // MARK: Presentation

    public var agentState: AgentState { dashboard.state ?? .idle }
    public var hold: HoldStatus { dashboard.hold }
    /// What the agent is doing, for people ("reading your journal").
    public var toolLabel: String? { dashboard.tool.map { $0.label ?? $0.name } }
    /// This client's measurement of its last push-to-talk turn.
    public var latency: TurnLatency? { dashboard.clientLatency }
    public var space: SpaceInfo? {
        dashboard.space.map { SpaceInfo(name: $0, mode: dashboard.mode, tier: dashboard.tier,
                                        description: dashboard.spaceDescription) }
    }

    public var isReady: Bool {
        if case .ready = connection { return true }
        return false
    }

    /// One line for the status pill, the menu bar and the Live Activity.
    public var statusLine: String {
        switch connection {
        case let .stopped(report?): return report.code == CloseCode.normal ? "Disconnected" : "Stopped: \(report)"
        case .stopped, .idle: return "Not connected"
        case .connecting, .handshaking: return "Connecting…"
        case let .waiting(_, delay, _): return "Reconnecting in \(max(1, delay.milliseconds / 1000)) s…"
        case .ready: break
        }
        if hold.phase == .held || agentState == .held {
            return hold.why.isEmpty ? "The Mac is busy" : "The Mac is busy: \(hold.why)"
        }
        if talk == .talking { return "Listening…" }
        if pendingApproval != nil { return "Waiting for your answer" }
        if let toolLabel { return toolLabel.prefix(1).uppercased() + toolLabel.dropFirst() }
        switch agentState {
        case .thinking: return "Thinking…"
        case .speaking: return "Speaking"
        case .listening: return handsFree ? "Listening" : "Ready"
        case .error: return "Something went wrong"
        default: return "Ready"
        }
    }

    public var symbolName: String {
        if !isReady { return "waveform.slash" }
        if hold.phase == .held { return "hourglass" }
        if talk == .talking { return "mic.fill" }
        if pendingApproval != nil { return "hand.raised.fill" }
        if toolLabel != nil { return "gearshape.2" }
        switch agentState {
        case .thinking: return "ellipsis.bubble"
        case .speaking: return "speaker.wave.2.fill"
        default: return handsFree ? "waveform" : "mic"
        }
    }

    // MARK: Unattended test mode

    /// Runs `settings.script` (if any), logging to the event log, then calls `finished` with the session still
    /// open (so the app can take its evidence) before it decides whether to quit.
    public func runScriptIfConfigured(finished: @escaping @MainActor (Bool) async -> Void) {
        guard let script = settings.script, !script.isEmpty else { return }
        let log = self.log
        let runner = ScriptRunner(target: self, tally: tally, log: log)
        Task {
            var ok = true
            do {
                try await runner.run(ScriptStep.parse(script))
            } catch {
                ok = false
                log?.write([("event", "script-failed"), ("error", .string("\(error)"))])
            }
            try? await Task.sleep(for: .milliseconds(400))
            log?.write([("event", "script-done"), ("ok", .bool(ok))])
            await finished(ok)
        }
    }
}

extension VoiceSessionModel: ScriptTarget {
    public func scriptConnect() async { connect() }
    public func scriptDisconnect() async { await disconnect() }
    public func scriptPressTalk() async { pressTalk() }
    public func scriptReleaseTalk() async { releaseTalk() }
    public func scriptHandsFree(_ on: Bool) async { await setHandsFree(on) }
    public func scriptStopSpeaking() async { stopSpeaking() }
    public func scriptText(_ text: String) async { send(text: text) }
    public func scriptAnswer(choice: String) async -> Bool {
        guard let id = pendingApproval?.id else { return false }
        return answer(choice, to: id)
    }
    public func scriptSnapshot(_ label: String) async { await onSnapshot?(label) }
    public func scriptSwitch(_ request: SwitchRequest) async {
        request.kind == .space ? switchSpace(request.name) : switchMode(request.name)
    }
    public func scriptRefreshStatus() async { refreshStatus() }
}
