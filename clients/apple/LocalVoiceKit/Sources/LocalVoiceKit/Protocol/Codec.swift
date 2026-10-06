import Foundation

/// Encodes and decodes every protocol v1 control message, in both directions.
///
/// The client only needs to encode `ClientMessage` and decode `ServerMessage`; the other two directions exist so the
/// tests can round-trip every message and so a Swift test server can speak the same protocol.
///
/// Decoding is strict about what the protocol requires (a `reply_id` on `audio_start`) and tolerant about the rest:
/// unknown fields are ignored, unknown types decode to `.unknown`, and ids or codes may be strings or numbers.
public enum ProtocolCodec {
    public enum DecodeError: Error, Equatable, CustomStringConvertible {
        case notJSON
        case notAnObject
        case missingType
        case missingField(type: String, field: String)

        public var description: String {
            switch self {
            case .notJSON: return "not JSON"
            case .notAnObject: return "not a JSON object"
            case .missingType: return "no \"t\" field"
            case let .missingField(type, field): return "\(type) without \(field)"
            }
        }
    }

    // MARK: Client messages

    public static func encode(_ message: ClientMessage) -> String {
        json(message).serialized()
    }

    public static func json(_ message: ClientMessage) -> JSONValue {
        var o: [(String, JSONValue)] = [("t", .string(message.type))]
        switch message {
        case let .hello(h):
            o += [("v", .int(h.version)), ("client", .string(h.client.rawValue)), ("device", .string(h.device)),
                  ("mic", .string(h.mic.rawValue))]
            if let space = h.space { o.append(("space", .string(space))) }
        case .start, .stop:
            break
        case let .interrupt(replyID):
            o.append(("reply_id", .string(replyID)))
        case let .playedMs(replyID, ms):
            o += [("reply_id", .string(replyID)), ("ms", .int(ms))]
        case let .text(text):
            o.append(("text", .string(text)))
        case let .space(name), let .mode(name):
            o.append(("name", .string(name)))
        case let .confirmResponse(id, confirmed, choice):
            o += [("id", id.json), ("confirmed", .bool(confirmed))]
            if let choice { o.append(("choice", .string(choice))) }
        case let .ping(n):
            o.append(("n", .int(n)))
        }
        return .object(o)
    }

    /// Decodes a client message (the server's side of the protocol). `nil` for an unknown type.
    public static func decodeClient(_ text: String) throws -> ClientMessage? {
        let (t, o) = try envelope(text)
        func str(_ field: String) throws -> String {
            guard let s = o[field]?.stringValue else { throw DecodeError.missingField(type: t, field: field) }
            return s
        }
        func int(_ field: String) throws -> Int {
            guard let i = o[field]?.intValue else { throw DecodeError.missingField(type: t, field: field) }
            return i
        }
        switch t {
        case "hello":
            guard let client = ClientKind(rawValue: try str("client")) else {
                throw DecodeError.missingField(type: t, field: "client")
            }
            guard let mic = MicMode(rawValue: try str("mic")) else {
                throw DecodeError.missingField(type: t, field: "mic")
            }
            return .hello(Hello(version: try int("v"), client: client, device: try str("device"), mic: mic,
                                space: o["space"]?.stringValue))
        case "start": return .start
        case "stop": return .stop
        case "interrupt": return .interrupt(replyID: try str("reply_id"))
        case "played_ms": return .playedMs(replyID: try str("reply_id"), ms: try int("ms"))
        case "text": return .text(try str("text"))
        case "space": return .space(name: try str("name"))
        case "mode": return .mode(name: try str("name"))
        case "confirm_response":
            guard let id = ScalarID(o["id"]) else { throw DecodeError.missingField(type: t, field: "id") }
            guard let confirmed = o["confirmed"]?.boolValue else {
                throw DecodeError.missingField(type: t, field: "confirmed")
            }
            return .confirmResponse(id: id, confirmed: confirmed, choice: o["choice"]?.stringValue)
        case "ping": return .ping(n: o["n"]?.intValue ?? 0)
        default: return nil
        }
    }

    // MARK: Server messages

    public static func encode(_ message: ServerMessage) -> String {
        json(message).serialized()
    }

    public static func json(_ message: ServerMessage) -> JSONValue {
        var o: [(String, JSONValue)] = [("t", .string(message.type))]
        switch message {
        case let .welcome(w):
            o += [("v", .int(w.version)), ("session", .string(w.session)), ("space", .string(w.space)),
                  ("mode", .string(w.mode)), ("tier", .string(w.tier)), ("state", .string(w.state.wire))]
            if let hold = w.hold {
                o.append(("hold", .object([("phase", .string(hold.phase.wire)), ("why", .string(hold.why))])))
            }
        case let .state(s):
            o.append(("v", .string(s.wire)))
        case let .transcript(final, text):
            o += [("final", .bool(final)), ("text", .string(text))]
        case let .replyText(replyID, delta):
            o += [("reply_id", .string(replyID)), ("delta", .string(delta))]
        case let .audioStart(replyID, rate):
            o += [("reply_id", .string(replyID)), ("rate", .int(rate))]
        case let .audioEnd(replyID):
            o.append(("reply_id", .string(replyID)))
        case let .interrupt(replyID), let .endOfTurn(replyID):
            if let replyID { o.append(("reply_id", .string(replyID))) }
        case let .tool(e):
            o += [("phase", .string(e.phase.wire)), ("name", .string(e.name))]
            if let label = e.label { o.append(("label", .string(label))) }
            if let ok = e.ok { o.append(("ok", .bool(ok))) }
        case let .confirmRequest(c):
            o += [("id", c.id.json), ("title", .string(c.title)), ("message", .string(c.message))]
            if let timeout = c.timeoutMs { o.append(("timeout_ms", .int(timeout))) }
            if let summary = c.summary { o.append(("summary", .string(summary))) }
            if let action = c.action { o.append(("action", json(action))) }
            if !c.choices.isEmpty {
                o.append(("choices", .array(c.choices.map { choice in
                    .object([("id", .string(choice.id))] + (choice.label.map { [("label", .string($0))] } ?? []))
                })))
            }
        case let .confirmCancel(id, why):
            o += [("id", id.json), ("why", .string(why))]
        case let .space(s):
            o.append(("name", .string(s.name)))
            if let mode = s.mode { o.append(("mode", .string(mode))) }
            if let tier = s.tier { o.append(("tier", .string(tier))) }
            if let description = s.description { o.append(("description", .string(description))) }
        case let .hold(h):
            o += [("phase", .string(h.phase.wire)), ("why", .string(h.why))]
        case let .error(code, message):
            o += [("code", .string(code)), ("message", .string(message))]
        case let .pong(n):
            o.append(("n", .int(n)))
        case .unknown:
            break
        }
        return .object(o)
    }

    public static func decodeServer(_ text: String) throws -> ServerMessage {
        let (t, o) = try envelope(text)
        func str(_ field: String) throws -> String {
            guard let s = o[field]?.stringValue else { throw DecodeError.missingField(type: t, field: field) }
            return s
        }
        switch t {
        case "welcome":
            let hold: HoldStatus?
            switch o["hold"] {
            case let .string(phase)?: hold = HoldStatus(phase: HoldPhase(wire: phase))
            case let obj? where obj["phase"]?.stringValue != nil:
                hold = HoldStatus(phase: HoldPhase(wire: obj["phase"]!.stringValue!), why: obj["why"]?.stringValue ?? "")
            default: hold = nil
            }
            return .welcome(Welcome(
                version: o["v"]?.intValue ?? ProtocolV1.version,
                session: o["session"]?.stringValue ?? "",
                space: o["space"]?.stringValue ?? "",
                mode: o["mode"]?.stringValue ?? "",
                tier: o["tier"]?.stringValue ?? "",
                state: AgentState(wire: o["state"]?.stringValue ?? "idle"),
                hold: hold))
        case "state":
            return .state(AgentState(wire: try str("v")))
        case "transcript":
            return .transcript(final: o["final"]?.boolValue ?? false, text: try str("text"))
        case "reply_text":
            return .replyText(replyID: try str("reply_id"), delta: try str("delta"))
        case "audio_start":
            return .audioStart(replyID: try str("reply_id"), rate: o["rate"]?.intValue ?? ProtocolV1.playbackSampleRate)
        case "audio_end":
            return .audioEnd(replyID: try str("reply_id"))
        case "interrupt":
            return .interrupt(replyID: o["reply_id"]?.stringValue)
        case "end_of_turn":
            return .endOfTurn(replyID: o["reply_id"]?.stringValue)
        case "tool":
            return .tool(ToolEvent(phase: ToolPhase(wire: try str("phase")), name: o["name"]?.stringValue ?? "",
                                   label: o["label"]?.stringValue, ok: o["ok"]?.boolValue))
        case "confirm_request":
            guard let id = ScalarID(o["id"]) else { throw DecodeError.missingField(type: t, field: "id") }
            return .confirmRequest(ConfirmRequest(id: id, title: o["title"]?.stringValue ?? "",
                                                  message: o["message"]?.stringValue ?? "",
                                                  timeoutMs: o["timeout_ms"]?.intValue,
                                                  summary: o["summary"]?.stringValue,
                                                  action: o["action"].flatMap(approvalAction),
                                                  choices: o["choices"]?.arrayValue.map(approvalChoices) ?? []))
        case "confirm_cancel":
            guard let id = ScalarID(o["id"]) else { throw DecodeError.missingField(type: t, field: "id") }
            return .confirmCancel(id: id, why: o["why"]?.stringValue ?? "")
        case "space":
            return .space(SpaceInfo(name: try str("name"), mode: o["mode"]?.stringValue, tier: o["tier"]?.stringValue,
                                    description: o["description"]?.stringValue))
        case "hold":
            return .hold(HoldStatus(phase: HoldPhase(wire: try str("phase")), why: o["why"]?.stringValue ?? ""))
        case "error":
            let code = o["code"].flatMap { $0.stringValue ?? $0.intValue.map(String.init) } ?? ""
            return .error(code: code, message: o["message"]?.stringValue ?? "")
        case "pong":
            return .pong(n: o["n"]?.intValue ?? 0)
        default:
            return .unknown(type: t)
        }
    }

    // MARK: Approvals (PROTOCOL.md, 2026-10-05)

    static func json(_ a: ApprovalAction) -> JSONValue {
        let fields: [(String, String?)] = [("tool", a.tool), ("effect", a.effect?.wire), ("command", a.command),
                                           ("path", a.path), ("cwd", a.cwd), ("space", a.space), ("mode", a.mode),
                                           ("preview", a.preview)]
        return .object(fields.compactMap { key, value in value.map { (key, .string($0)) } })
    }

    /// Anything but an object is no action; a field that is missing, null or not a string is nil.
    static func approvalAction(_ value: JSONValue) -> ApprovalAction? {
        guard value.objectPairs != nil else { return nil }
        return ApprovalAction(tool: value["tool"]?.stringValue,
                              effect: value["effect"]?.stringValue.map(ApprovalEffect.init(wire:)),
                              command: value["command"]?.stringValue, path: value["path"]?.stringValue,
                              cwd: value["cwd"]?.stringValue, space: value["space"]?.stringValue,
                              mode: value["mode"]?.stringValue, preview: value["preview"]?.stringValue)
    }

    /// The offered choices in order; an entry without an id cannot be answered and is left out.
    static func approvalChoices(_ items: [JSONValue]) -> [ApprovalChoice] {
        items.compactMap { item in
            guard let id = item["id"].flatMap({ $0.stringValue ?? $0.intValue.map(String.init) }), !id.isEmpty else {
                return nil
            }
            return ApprovalChoice(id: id, label: item["label"]?.stringValue)
        }
    }

    private static func envelope(_ text: String) throws -> (String, JSONValue) {
        let value: JSONValue
        do {
            value = try JSONValue.parse(text)
        } catch {
            throw DecodeError.notJSON
        }
        guard case .object = value else { throw DecodeError.notAnObject }
        guard let t = value["t"]?.stringValue else { throw DecodeError.missingType }
        return (t, value)
    }
}
