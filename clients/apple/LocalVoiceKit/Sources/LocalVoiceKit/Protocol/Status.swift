import Foundation

/// `GET /v1/status` (PROTOCOL.md "Status endpoint"): what the dashboard shows.
///
/// Decoding is tolerant, like the message codec: any field may be missing (the orchestrator serves only `v`, `state`,
/// `hold` and `clients` before its agent hub is up), `hold` may be a bare phase string, numbers may be floats, and the
/// orchestrator's additions beyond PROTOCOL.md's example (`spaces`, `turns`, `uptime_s`, as of 2026-10-05) are read
/// when present.
public struct ServerStatus: Sendable, Equatable {
    /// The last turn's latency, as the server measured it.
    public struct LastTurn: Sendable, Equatable {
        public var eosToFirstAudioMs: Double?
        public var sttMs: Double?
        public var llmFirstTokenMs: Double?
        public var ttsFirstAudioMs: Double?
        public var tools: [String]
        public var space: String?

        public init(eosToFirstAudioMs: Double? = nil, sttMs: Double? = nil, llmFirstTokenMs: Double? = nil,
                    ttsFirstAudioMs: Double? = nil, tools: [String] = [], space: String? = nil) {
            self.eosToFirstAudioMs = eosToFirstAudioMs
            self.sttMs = sttMs
            self.llmFirstTokenMs = llmFirstTokenMs
            self.ttsFirstAudioMs = ttsFirstAudioMs
            self.tools = tools
            self.space = space
        }

        /// The orchestrator reports zeros until the first turn.
        public var isEmpty: Bool { (eosToFirstAudioMs ?? 0) <= 0 }
    }

    /// A tool the agent is running ("reading your journal").
    public struct RunningTool: Sendable, Equatable {
        public var name: String
        public var label: String?

        public init(name: String, label: String? = nil) {
            self.name = name
            self.label = label
        }
    }

    /// A space the server can switch to: the orchestrator's `spaces` map (spaces.yaml), not in PROTOCOL.md's example.
    public struct Space: Sendable, Equatable, Identifiable {
        public var name: String
        public var description: String?
        public var tier: String?
        public var model: String?
        public var id: String { name }

        public init(name: String, description: String? = nil, tier: String? = nil, model: String? = nil) {
            self.name = name
            self.description = description
            self.tier = tier
            self.model = model
        }
    }

    public struct Client: Sendable, Equatable {
        public var device: String
        public var kind: String?
        public var connectedSeconds: Double?

        public init(device: String, kind: String? = nil, connectedSeconds: Double? = nil) {
            self.device = device
            self.kind = kind
            self.connectedSeconds = connectedSeconds
        }
    }

    public var version: Int?
    public var state: AgentState?
    public var space: String?
    public var mode: String?
    public var tier: String?
    public var model: String?
    public var hold: HoldStatus?
    public var tool: RunningTool?
    public var lastTurn: LastTurn?
    public var turns: Int?
    /// Sorted by name (JSON objects arrive unordered).
    public var spaces: [Space]
    public var clients: [Client]
    public var uptimeSeconds: Double?

    public init(version: Int? = ProtocolV1.version, state: AgentState? = nil, space: String? = nil, mode: String? = nil,
                tier: String? = nil, model: String? = nil, hold: HoldStatus? = nil, tool: RunningTool? = nil,
                lastTurn: LastTurn? = nil, turns: Int? = nil, spaces: [Space] = [], clients: [Client] = [],
                uptimeSeconds: Double? = nil) {
        self.version = version
        self.state = state
        self.space = space
        self.mode = mode
        self.tier = tier
        self.model = model
        self.hold = hold
        self.tool = tool
        self.lastTurn = lastTurn
        self.turns = turns
        self.spaces = spaces
        self.clients = clients
        self.uptimeSeconds = uptimeSeconds
    }

    public enum DecodeError: Error, Equatable, CustomStringConvertible {
        case notAnObject
        public var description: String { "the status is not a JSON object" }
    }

    public static func decode(_ data: Data) throws -> ServerStatus {
        try decode(JSONValue.parse(data))
    }

    public static func decode(_ o: JSONValue) throws -> ServerStatus {
        guard case .object = o else { throw DecodeError.notAnObject }
        var s = ServerStatus(version: o["v"]?.intValue)
        s.state = o["state"]?.stringValue.map(AgentState.init(wire:))
        s.space = o["space"]?.stringValue
        s.mode = o["mode"]?.stringValue
        s.tier = o["tier"]?.stringValue
        s.model = o["model"]?.stringValue
        switch o["hold"] {
        case let .string(phase)?: s.hold = HoldStatus(phase: HoldPhase(wire: phase))
        case let h? where h["phase"]?.stringValue != nil:
            s.hold = HoldStatus(phase: HoldPhase(wire: h["phase"]!.stringValue!), why: h["why"]?.stringValue ?? "")
        default: break
        }
        if let t = o["tool"], let name = t["name"]?.stringValue {
            s.tool = RunningTool(name: name, label: t["label"]?.stringValue)
        }
        if let t = o["last_turn"], case .object = t {
            s.lastTurn = LastTurn(eosToFirstAudioMs: t["eos_to_first_audio_ms"]?.doubleValue,
                                  sttMs: t["stt_ms"]?.doubleValue, llmFirstTokenMs: t["llm_ttft_ms"]?.doubleValue,
                                  ttsFirstAudioMs: t["tts_first_audio_ms"]?.doubleValue,
                                  tools: t["tools"]?.arrayValue?.compactMap(\.stringValue) ?? [],
                                  space: t["space"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 })
        }
        s.turns = o["turns"]?.intValue
        s.spaces = (o["spaces"]?.objectPairs ?? []).map { key, v in
            Space(name: v["name"]?.stringValue ?? key, description: v["description"]?.stringValue,
                  tier: v["tier"]?.stringValue, model: v["model"]?.stringValue)
        }.sorted { $0.name < $1.name }
        s.clients = (o["clients"]?.arrayValue ?? []).compactMap { c in
            c["device"]?.stringValue.map {
                Client(device: $0, kind: c["client"]?.stringValue, connectedSeconds: c["connected_s"]?.doubleValue)
            }
        }
        s.uptimeSeconds = o["uptime_s"]?.doubleValue
        return s
    }

    /// The fields this client read, in the wire's names (for the event log).
    public var json: JSONValue {
        func opt(_ s: String?) -> JSONValue { s.map(JSONValue.string) ?? .null }
        func num(_ d: Double?) -> JSONValue { d.map(JSONValue.double) ?? .null }
        var o: [(String, JSONValue)] = [
            ("v", version.map(JSONValue.int) ?? .null), ("state", opt(state?.wire)), ("space", opt(space)),
            ("mode", opt(mode)), ("tier", opt(tier)), ("model", opt(model)),
            ("hold", hold.map { .object([("phase", .string($0.phase.wire)), ("why", .string($0.why))]) } ?? .null),
            ("tool", tool.map { .object([("name", .string($0.name)), ("label", opt($0.label))]) } ?? .null),
        ]
        if let t = lastTurn {
            o.append(("last_turn", .object([
                ("eos_to_first_audio_ms", num(t.eosToFirstAudioMs)), ("stt_ms", num(t.sttMs)),
                ("llm_ttft_ms", num(t.llmFirstTokenMs)), ("tts_first_audio_ms", num(t.ttsFirstAudioMs)),
                ("tools", .array(t.tools.map(JSONValue.string))), ("space", opt(t.space)),
            ])))
        }
        o.append(("turns", turns.map(JSONValue.int) ?? .null))
        o.append(("spaces", .array(spaces.map {
            .object([("name", .string($0.name)), ("description", opt($0.description)), ("tier", opt($0.tier)),
                     ("model", opt($0.model))])
        })))
        o.append(("clients", .array(clients.map {
            .object([("device", .string($0.device)), ("client", opt($0.kind)),
                     ("connected_s", num($0.connectedSeconds))])
        })))
        return .object(o)
    }
}

extension ProtocolV1 {
    /// `/v1/status` on the voice WebSocket's host and port (PROTOCOL.md: the same server, the same access rules):
    /// `ws` becomes `http`, `wss` becomes `https`.
    public static func statusURL(forVoiceURL url: URL) -> URL? {
        guard var c = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        switch c.scheme?.lowercased() {
        case "ws": c.scheme = "http"
        case "wss": c.scheme = "https"
        case "http", "https": break
        default: return nil
        }
        c.path = statusPath
        c.query = nil
        c.fragment = nil
        return c.url
    }
}

/// Where the dashboard's status comes from: HTTP in the apps, a fake in tests.
public protocol StatusFetching: Sendable {
    func fetchStatus() async throws -> ServerStatus
}

public enum StatusFetchError: Error, Equatable, CustomStringConvertible {
    case http(Int)
    case notHTTP

    public var description: String {
        switch self {
        case .http(403): return "not allowed (403)"
        case .http(404): return "the server has no /v1/status (404)"
        case let .http(code): return "HTTP \(code)"
        case .notHTTP: return "not an HTTP answer"
        }
    }
}

public struct HTTPStatusFetcher: StatusFetching {
    public let url: URL
    private let session: URLSession

    /// 4 s: the tailnet's round trip measured 7-227 ms (research notes); longer means the Mac is unreachable.
    public init(url: URL, timeout: TimeInterval = 4) {
        self.url = url
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: config)
    }

    public func fetchStatus() async throws -> ServerStatus {
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse else { throw StatusFetchError.notHTTP }
        guard http.statusCode == 200 else { throw StatusFetchError.http(http.statusCode) }
        return try ServerStatus.decode(data)
    }
}
