import Foundation

// Control messages of protocol v1 (PROTOCOL.md "Control messages"). Every message is a JSON object with "t".

public enum ClientKind: String, Sendable, CaseIterable {
    case iphone, mac, test
}

/// `vad`: the server detects turns on an open microphone. `ptt`: the client brackets each turn with start/stop.
public enum MicMode: String, Sendable, CaseIterable {
    case vad, ptt
}

public struct Hello: Sendable, Equatable {
    public var version: Int
    public var client: ClientKind
    public var device: String
    public var mic: MicMode
    public var space: String?

    public init(version: Int = ProtocolV1.version, client: ClientKind, device: String, mic: MicMode, space: String? = nil) {
        self.version = version
        self.client = client
        self.device = device
        self.mic = mic
        self.space = space
    }
}

/// An id that arrives as a JSON string or number and must go back in the same JSON type (`confirm_request.id`; the
/// protocol does not fix its type, and Pi's dialog ids are strings while a test server may use numbers).
public enum ScalarID: Sendable, Hashable, CustomStringConvertible {
    case string(String)
    case int(Int)

    public var json: JSONValue {
        switch self {
        case let .string(s): return .string(s)
        case let .int(i): return .int(i)
        }
    }

    public var description: String {
        switch self {
        case let .string(s): return s
        case let .int(i): return String(i)
        }
    }

    init?(_ value: JSONValue?) {
        switch value {
        case let .string(s)?: self = .string(s)
        case let v? where v.intValue != nil: self = .int(v.intValue!)
        default: return nil
        }
    }
}

public enum ClientMessage: Sendable, Equatable {
    case hello(Hello)
    /// Push-to-talk pressed: start of a user turn.
    case start
    /// Push-to-talk released: end of the user turn.
    case stop
    /// The client stopped playback itself (stop tapped, or client-side barge-in).
    case interrupt(replyID: String)
    /// How much of a reply's audio was actually played; after any interrupt and at the end of every reply.
    case playedMs(replyID: String, ms: Int)
    /// Typed input, treated as a user turn.
    case text(String)
    case space(name: String)
    case mode(name: String)
    /// The answer to a `confirm_request`: `choice` is one of the offered ids (PROTOCOL.md "Approvals"), nil for a
    /// request that offered none (a server before 2026-10-05 reads only `confirmed`).
    case confirmResponse(id: ScalarID, confirmed: Bool, choice: String? = nil)
    case ping(n: Int)

    public var type: String {
        switch self {
        case .hello: return "hello"
        case .start: return "start"
        case .stop: return "stop"
        case .interrupt: return "interrupt"
        case .playedMs: return "played_ms"
        case .text: return "text"
        case .space: return "space"
        case .mode: return "mode"
        case .confirmResponse: return "confirm_response"
        case .ping: return "ping"
        }
    }
}

public enum AgentState: Sendable, Equatable, CustomStringConvertible {
    case idle, listening, thinking, speaking, held, error
    case other(String)

    public init(wire: String) {
        switch wire {
        case "idle": self = .idle
        case "listening": self = .listening
        case "thinking": self = .thinking
        case "speaking": self = .speaking
        case "held": self = .held
        case "error": self = .error
        default: self = .other(wire)
        }
    }

    public var wire: String {
        switch self {
        case .idle: return "idle"
        case .listening: return "listening"
        case .thinking: return "thinking"
        case .speaking: return "speaking"
        case .held: return "held"
        case .error: return "error"
        case let .other(s): return s
        }
    }

    public var description: String { wire }
}

public enum HoldPhase: Sendable, Equatable, CustomStringConvertible {
    case open, draining, held
    case other(String)

    public init(wire: String) {
        switch wire {
        case "open": self = .open
        case "draining": self = .draining
        case "held": self = .held
        default: self = .other(wire)
        }
    }

    public var wire: String {
        switch self {
        case .open: return "open"
        case .draining: return "draining"
        case .held: return "held"
        case let .other(s): return s
        }
    }

    public var description: String { wire }
}

public struct HoldStatus: Sendable, Equatable {
    public var phase: HoldPhase
    public var why: String

    public init(phase: HoldPhase, why: String = "") {
        self.phase = phase
        self.why = why
    }
}

public struct Welcome: Sendable, Equatable {
    public var version: Int
    public var session: String
    public var space: String
    public var mode: String
    public var tier: String
    public var state: AgentState
    /// The example in PROTOCOL.md sends `"hold":"open"`; the `hold` message and `/v1/status` use an object. Both parse.
    public var hold: HoldStatus?

    public init(version: Int = ProtocolV1.version, session: String, space: String, mode: String, tier: String,
                state: AgentState, hold: HoldStatus?) {
        self.version = version
        self.session = session
        self.space = space
        self.mode = mode
        self.tier = tier
        self.state = state
        self.hold = hold
    }
}

public enum ToolPhase: Sendable, Equatable {
    case start, update, end
    case other(String)

    public init(wire: String) {
        switch wire {
        case "start": self = .start
        case "update": self = .update
        case "end": self = .end
        default: self = .other(wire)
        }
    }

    public var wire: String {
        switch self {
        case .start: return "start"
        case .update: return "update"
        case .end: return "end"
        case let .other(s): return s
        }
    }
}

public struct ToolEvent: Sendable, Equatable {
    public var phase: ToolPhase
    public var name: String
    /// What the agent is doing, for people ("reading your journal").
    public var label: String?
    /// Present on `end`.
    public var ok: Bool?

    public init(phase: ToolPhase, name: String, label: String? = nil, ok: Bool? = nil) {
        self.phase = phase
        self.name = name
        self.label = label
        self.ok = ok
    }
}

/// The agent asks permission. A server since PROTOCOL.md "Approvals" (2026-10-05) also says exactly what will happen
/// (`summary`, `action`) and which answers it takes (`choices`); an older one sends only `title` and `message`.
public struct ConfirmRequest: Sendable, Equatable {
    public var id: ScalarID
    public var title: String
    public var message: String
    public var timeoutMs: Int?
    /// One plain sentence: exactly what will happen, and to what.
    public var summary: String?
    public var action: ApprovalAction?
    /// In the server's order; empty when the server offered none (then it takes a plain yes or no).
    public var choices: [ApprovalChoice]

    public init(id: ScalarID, title: String, message: String, timeoutMs: Int?, summary: String? = nil,
                action: ApprovalAction? = nil, choices: [ApprovalChoice] = []) {
        self.id = id
        self.title = title
        self.message = message
        self.timeoutMs = timeoutMs
        self.summary = summary
        self.action = action
        self.choices = choices
    }
}

/// What a permission request would do, field by field (PROTOCOL.md "Approvals"). Every field may be missing.
public struct ApprovalAction: Sendable, Equatable {
    /// `write`, `edit`, `bash`, `kb`, ...
    public var tool: String?
    public var effect: ApprovalEffect?
    /// The exact command line, for a shell or kb call.
    public var command: String?
    /// The absolute path of the file written or edited.
    public var path: String?
    public var cwd: String?
    public var space: String?
    public var mode: String?
    /// The text to be written, or a unified diff for an edit; the server cuts it at 4,000 characters with a marker.
    public var preview: String?

    public init(tool: String? = nil, effect: ApprovalEffect? = nil, command: String? = nil, path: String? = nil,
                cwd: String? = nil, space: String? = nil, mode: String? = nil, preview: String? = nil) {
        self.tool = tool
        self.effect = effect
        self.command = command
        self.path = path
        self.cwd = cwd
        self.space = space
        self.mode = mode
        self.preview = preview
    }
}

public enum ApprovalEffect: Sendable, Equatable, CustomStringConvertible {
    case create, modify, delete, run, network
    case other(String)

    public init(wire: String) {
        switch wire {
        case "create": self = .create
        case "modify": self = .modify
        case "delete": self = .delete
        case "run": self = .run
        case "network": self = .network
        default: self = .other(wire)
        }
    }

    public var wire: String {
        switch self {
        case .create: return "create"
        case .modify: return "modify"
        case .delete: return "delete"
        case .run: return "run"
        case .network: return "network"
        case let .other(s): return s
        }
    }

    public var description: String { wire }
}

/// One answer the server offers, shown as one button.
public struct ApprovalChoice: Sendable, Equatable {
    public static let allowOnce = "allow_once"
    public static let allowSession = "allow_session"
    public static let deny = "deny"

    public var id: String
    /// The button's words ("Do it"); nil when the server sent none.
    public var label: String?

    public init(id: String, label: String? = nil) {
        self.id = id
        self.label = label
    }

    /// What `confirm_response.confirmed` says for this choice: true for any allow, false for `deny`. An id this client
    /// does not know counts as an allow only when it says so (`allow_…`), so nothing unknown is ever sent as a yes.
    public var allows: Bool { id == "allow" || id.hasPrefix("allow_") }
}

public struct SpaceInfo: Sendable, Equatable {
    public var name: String
    public var mode: String?
    public var tier: String?
    public var description: String?

    public init(name: String, mode: String? = nil, tier: String? = nil, description: String? = nil) {
        self.name = name
        self.mode = mode
        self.tier = tier
        self.description = description
    }
}

public enum ServerMessage: Sendable, Equatable {
    case welcome(Welcome)
    case state(AgentState)
    /// What the user said; partials (`final == false`) may be replaced.
    case transcript(final: Bool, text: String)
    /// Assistant text as it streams (captions).
    case replyText(replyID: String, delta: String)
    /// Binary audio for this reply follows.
    case audioStart(replyID: String, rate: Int)
    case audioEnd(replyID: String)
    /// Flush local playback now: the user barged in or the reply was cancelled.
    case interrupt(replyID: String?)
    /// The agent is done; the client may arm its mic gate.
    case endOfTurn(replyID: String?)
    case tool(ToolEvent)
    case confirmRequest(ConfirmRequest)
    /// The question is withdrawn (answered by voice, timed out, or overtaken): close its card.
    case confirmCancel(id: ScalarID, why: String)
    case space(SpaceInfo)
    case hold(HoldStatus)
    /// Non-fatal; fatal problems close the socket. The protocol does not fix the code's JSON type.
    case error(code: String, message: String)
    case pong(n: Int)
    /// A type this client does not know. Receivers ignore unknown types (additive messages are allowed in v1).
    case unknown(type: String)

    public var type: String {
        switch self {
        case .welcome: return "welcome"
        case .state: return "state"
        case .transcript: return "transcript"
        case .replyText: return "reply_text"
        case .audioStart: return "audio_start"
        case .audioEnd: return "audio_end"
        case .interrupt: return "interrupt"
        case .endOfTurn: return "end_of_turn"
        case .tool: return "tool"
        case .confirmRequest: return "confirm_request"
        case .confirmCancel: return "confirm_cancel"
        case .space: return "space"
        case .hold: return "hold"
        case .error: return "error"
        case .pong: return "pong"
        case let .unknown(t): return t
        }
    }
}
