import Foundation

/// What the status view shows: which space the agent is operating in and which mode it is in, plus what the agent is doing, the GPU hold and how fast the last turn was. A value folded from the
/// client's events, so both apps and their tests share it.
///
/// Sources: the session's messages keep it current between fetches (`welcome`, `space`, `state`, `hold`, `tool`), and
/// `/v1/status` adds what no message carries (the model, the server's latency breakdown, the spaces there are, who is
/// connected). Where both carry a field:
/// - state, hold and tool come from messages once the session is up: the server pushes every change, while a status
///   may have been taken before the latest one (a turn moves through three states in a second);
/// - space, mode and tier come from whichever arrived last: another device may have switched the shared agent without
///   this connection hearing a `space` message, and a status that raced a `space` message is followed by a fresh
///   fetch anyway (`StatusMonitor` fetches after every `space` message).
public struct Dashboard: Sendable, Equatable {
    public private(set) var session: String?
    public private(set) var space: String?
    public private(set) var spaceDescription: String?
    public private(set) var mode: String?
    public private(set) var tier: String?
    public private(set) var model: String?
    public private(set) var state: AgentState?
    public private(set) var hold = HoldStatus(phase: .open)
    public private(set) var tool: ServerStatus.RunningTool?
    /// The server's breakdown of the last turn.
    public private(set) var lastTurn: ServerStatus.LastTurn?
    public private(set) var turns: Int?
    /// What this client measured for its last push-to-talk turn: release to `audio_start` and to the first sound.
    public private(set) var clientLatency: TurnLatency?
    public private(set) var spaces: [ServerStatus.Space] = []
    public private(set) var clients: [ServerStatus.Client] = []
    public private(set) var statusAt: Date?
    public private(set) var statusProblem: String?
    public private(set) var pendingSwitch: SwitchRequest?
    public private(set) var lastSwitch: SwitchOutcome?

    /// The modes of PROTOCOL.md's `mode` message.
    public static let modes = ["conversation", "act"]

    private var sessionUp = false
    private var toolMessageSeen = false

    public init() {}

    public mutating func apply(_ event: ClientEvent, now: Date = Date()) {
        switch event {
        case let .server(message): apply(message)
        case let .status(_, status): apply(status: status, at: now)
        case let .statusFailed(_, why): statusProblem = why
        case let .latency(l): clientLatency = l
        case let .switchRequested(r):
            pendingSwitch = r
            lastSwitch = nil
        case let .switchOutcome(o):
            pendingSwitch = nil
            lastSwitch = o
        case let .connection(state):
            switch state {
            case .ready: break
            default: sessionUp = false
            }
        default: break
        }
    }

    private mutating func apply(_ message: ServerMessage) {
        switch message {
        case let .welcome(w):
            sessionUp = true
            toolMessageSeen = false
            session = w.session
            enter(space: w.space, description: nil)
            if !w.mode.isEmpty { mode = w.mode }
            if !w.tier.isEmpty { tier = w.tier }
            state = w.state
            if let h = w.hold { hold = h }
        case let .space(info):
            enter(space: info.name, description: info.description)
            if let m = info.mode { mode = m }
            if let t = info.tier { tier = t }
        case let .state(s):
            state = s
        case let .hold(h):
            hold = h
        case let .tool(e):
            toolMessageSeen = true
            switch e.phase {
            case .start, .update: tool = ServerStatus.RunningTool(name: e.name, label: e.label ?? tool?.label)
            case .end: tool = nil
            case .other: break
            }
        default:
            break
        }
    }

    private mutating func apply(status s: ServerStatus, at now: Date) {
        statusAt = now
        statusProblem = nil
        if !s.spaces.isEmpty { spaces = s.spaces }
        clients = s.clients
        if let n = s.turns { turns = n }
        if let t = s.lastTurn, !t.isEmpty { lastTurn = t }
        if let name = s.space { enter(space: name, description: nil) }
        if let m = s.mode { mode = m }
        if let t = s.tier { tier = t }
        if let m = s.model, s.space == nil || s.space == space { model = m }
        if !sessionUp {
            if let v = s.state { state = v }
            if let h = s.hold { hold = h }
        }
        if !sessionUp || !toolMessageSeen { tool = s.tool }
    }

    /// The agent is in `name` now: what was known of another space no longer applies.
    private mutating func enter(space name: String, description: String?) {
        guard !name.isEmpty else { return }
        let known = spaces.first { $0.name == name }
        if name != space {
            space = name
            spaceDescription = nil
            model = known?.model
        }
        if let description, !description.isEmpty {
            spaceDescription = description
        } else if spaceDescription == nil {
            spaceDescription = known?.description
        }
        if model == nil { model = known?.model }
    }

    // MARK: Presentation

    /// "atlas · act" for a status line.
    public var whereLine: String? {
        guard let space else { return nil }
        return [space, mode].compactMap { $0 }.joined(separator: " · ")
    }

    /// "your atlas journal", or the space's name when the server gave no description.
    public var spaceTitle: String? { spaceDescription ?? space }

    /// What came of the last switch, in a sentence; nil while none is known.
    public var switchLine: String? {
        if let p = pendingSwitch {
            return p.kind == .space ? "Switching to \(p.name)…" : "Switching to \(p.name) mode…"
        }
        switch lastSwitch {
        case let .switched(r, info)?:
            return r.kind == .space ? "Now in \(info.description ?? info.name)." : "Now in \(r.name) mode."
        case let .refused(r, reason)?:
            return "Could not switch to \(r.name)\(r.kind == .mode ? " mode" : ""): \(reason)."
        case let .noAnswer(r)?:
            return "No answer to switching to \(r.name)\(r.kind == .mode ? " mode" : ""); the server may not support "
                + "switching yet."
        case nil:
            return nil
        }
    }

    /// The last turn in one line: the server's end of speech to first audio and where the time went, then what the
    /// client measured. "844 ms to first audio (STT 40, LLM 300, TTS 130); 905 ms from release to sound here".
    public var latencyLine: String? {
        var parts: [String] = []
        if let t = lastTurn, let eos = t.eosToFirstAudioMs, eos > 0 {
            let split = [("STT", t.sttMs), ("LLM", t.llmFirstTokenMs), ("TTS", t.ttsFirstAudioMs)]
                .compactMap { name, v in v.flatMap { $0 > 0 ? "\(name) \(Int($0.rounded()))" : nil } }
            let detail = split.isEmpty ? "" : " (\(split.joined(separator: ", ")))"
            parts.append("\(Int(eos.rounded())) ms to first audio" + detail)
        }
        if let ms = clientLatency?.releaseToPlaybackMs {
            parts.append("\(ms) ms from release to sound here")
        }
        return parts.isEmpty ? nil : parts.joined(separator: "; ")
    }

    /// "The GPU is free", "The GPU is held: film render"; draining is the hold gate waiting for work in flight to
    /// finish before a GPU job takes over.
    public var holdLine: String {
        switch hold.phase {
        case .open: return "The GPU is free"
        case .held: return hold.why.isEmpty ? "The GPU is held" : "The GPU is held: \(hold.why)"
        case .draining:
            return hold.why.isEmpty ? "The GPU is about to be held" : "The GPU is about to be held: \(hold.why)"
        case let .other(s): return "GPU: \(s)"
        }
    }
}
