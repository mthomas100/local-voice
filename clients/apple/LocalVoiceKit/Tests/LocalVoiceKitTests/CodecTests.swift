import Foundation
import Testing
@testable import LocalVoiceKit

/// Every PROTOCOL.md v1 control message, both directions (definition of done M2.1).
@Suite("Protocol v1 codec")
struct CodecTests {
    // MARK: Client -> server

    static let clientCases: [(ClientMessage, String)] = [
        (.hello(Hello(client: .iphone, device: "iphone", mic: .vad)),
         #"{"t":"hello","v":1,"client":"iphone","device":"iphone","mic":"vad"}"#),
        (.hello(Hello(client: .mac, device: "mac", mic: .ptt, space: "atlas")),
         #"{"t":"hello","v":1,"client":"mac","device":"mac","mic":"ptt","space":"atlas"}"#),
        (.start, #"{"t":"start"}"#),
        (.stop, #"{"t":"stop"}"#),
        (.interrupt(replyID: "r1"), #"{"t":"interrupt","reply_id":"r1"}"#),
        (.playedMs(replyID: "r1", ms: 2140), #"{"t":"played_ms","reply_id":"r1","ms":2140}"#),
        (.text("what did I write yesterday?"), #"{"t":"text","text":"what did I write yesterday?"}"#),
        (.space(name: "journal"), #"{"t":"space","name":"journal"}"#),
        (.mode(name: "act"), #"{"t":"mode","name":"act"}"#),
        (.confirmResponse(id: .string("c7"), confirmed: true), #"{"t":"confirm_response","id":"c7","confirmed":true}"#),
        (.confirmResponse(id: .int(3), confirmed: false), #"{"t":"confirm_response","id":3,"confirmed":false}"#),
        // PROTOCOL.md "Approvals": the offered id goes back with `confirmed` (true for any allow, false for deny).
        (.confirmResponse(id: .string("c8"), confirmed: true, choice: "allow_session"),
         #"{"t":"confirm_response","id":"c8","confirmed":true,"choice":"allow_session"}"#),
        (.confirmResponse(id: .string("c9"), confirmed: false, choice: "deny"),
         #"{"t":"confirm_response","id":"c9","confirmed":false,"choice":"deny"}"#),
        (.ping(n: 4), #"{"t":"ping","n":4}"#),
    ]

    @Test("client messages encode to the documented JSON", arguments: clientCases)
    func clientEncodes(message: ClientMessage, expected: String) throws {
        let wire = ProtocolCodec.encode(message)
        #expect(try JSONValue.parse(wire) == JSONValue.parse(expected))
        #expect(wire.hasPrefix(#"{"t":""#), "the type comes first, for readable logs")
    }

    @Test("client messages round-trip", arguments: clientCases.map(\.0))
    func clientRoundTrips(message: ClientMessage) throws {
        #expect(try ProtocolCodec.decodeClient(ProtocolCodec.encode(message)) == message)
    }

    @Test("every client message type is covered")
    func clientCoverage() {
        let covered = Set(Self.clientCases.map { $0.0.type })
        #expect(covered == ["hello", "start", "stop", "interrupt", "played_ms", "text", "space", "mode",
                            "confirm_response", "ping"])
    }

    // MARK: Server -> client

    /// The server side, with the exact lines of PROTOCOL.md's example session where it has one.
    static let serverCases: [(String, ServerMessage)] = [
        (#"{"t":"welcome","v":1,"session":"s1","space":"home","mode":"conversation","tier":"ask","state":"listening","hold":"open"}"#,
         .welcome(Welcome(session: "s1", space: "home", mode: "conversation", tier: "ask", state: .listening,
                          hold: HoldStatus(phase: .open)))),
        (#"{"t":"state","v":"thinking"}"#, .state(.thinking)),
        (#"{"t":"state","v":"held"}"#, .state(.held)),
        (#"{"t":"transcript","final":true,"text":"what did I write in my journal yesterday"}"#,
         .transcript(final: true, text: "what did I write in my journal yesterday")),
        (#"{"t":"transcript","final":false,"text":"what did"}"#, .transcript(final: false, text: "what did")),
        (#"{"t":"reply_text","reply_id":"r1","delta":"Yesterday you wrote about"}"#,
         .replyText(replyID: "r1", delta: "Yesterday you wrote about")),
        (#"{"t":"audio_start","reply_id":"r1","rate":24000}"#, .audioStart(replyID: "r1", rate: 24000)),
        (#"{"t":"audio_end","reply_id":"r1"}"#, .audioEnd(replyID: "r1")),
        (#"{"t":"interrupt","reply_id":"r1"}"#, .interrupt(replyID: "r1")),
        (#"{"t":"end_of_turn","reply_id":"r1"}"#, .endOfTurn(replyID: "r1")),
        (#"{"t":"tool","phase":"start","name":"read","label":"reading your journal"}"#,
         .tool(ToolEvent(phase: .start, name: "read", label: "reading your journal"))),
        (#"{"t":"tool","phase":"end","name":"read","ok":true}"#, .tool(ToolEvent(phase: .end, name: "read", ok: true))),
        (#"{"t":"confirm_request","id":"c7","title":"Run a command","message":"Run ls in your home?","timeout_ms":20000}"#,
         .confirmRequest(ConfirmRequest(id: .string("c7"), title: "Run a command", message: "Run ls in your home?",
                                        timeoutMs: 20000))),
        (Self.approvalWire, .confirmRequest(Self.approval)),
        (#"{"t":"confirm_cancel","id":"c8","why":"answered by voice"}"#,
         .confirmCancel(id: .string("c8"), why: "answered by voice")),
        (#"{"t":"space","name":"atlas","mode":"conversation","tier":"trusted","description":"the journal"}"#,
         .space(SpaceInfo(name: "atlas", mode: "conversation", tier: "trusted", description: "the journal"))),
        (#"{"t":"hold","phase":"held","why":"film render"}"#, .hold(HoldStatus(phase: .held, why: "film render"))),
        (#"{"t":"error","code":"stt","message":"speech recognition failed"}"#,
         .error(code: "stt", message: "speech recognition failed")),
        (#"{"t":"pong","n":4}"#, .pong(n: 4)),
    ]

    @Test("server messages decode", arguments: serverCases)
    func serverDecodes(wire: String, expected: ServerMessage) throws {
        #expect(try ProtocolCodec.decodeServer(wire) == expected)
    }

    @Test("server messages round-trip", arguments: serverCases.map(\.1))
    func serverRoundTrips(message: ServerMessage) throws {
        #expect(try ProtocolCodec.decodeServer(ProtocolCodec.encode(message)) == message)
    }

    @Test("every server message type is covered")
    func serverCoverage() {
        let covered = Set(Self.serverCases.map { $0.1.type })
        #expect(covered == ["welcome", "state", "transcript", "reply_text", "audio_start", "audio_end", "interrupt",
                            "end_of_turn", "tool", "confirm_request", "confirm_cancel", "space", "hold", "error",
                            "pong"])
    }

    // MARK: Approvals (PROTOCOL.md, added 2026-10-05)

    /// A request with every field the Approvals section defines, in the shape of its own example.
    static let approvalWire = #"""
        {"t":"confirm_request","id":"c8","title":"May I change your knowledge base?","message":"kb new Analysis tea-brewing-notes","timeout_ms":120000,"summary":"Create a new page in your knowledge base titled 'Tea brewing notes'.","action":{"tool":"kb","effect":"create","command":"kb new Analysis tea-brewing-notes --title 'Tea brewing notes'","path":"/Users/owner/kb/wiki/analyses/tea-brewing-notes.md","cwd":"/Users/owner/kb","space":"home","mode":"act","preview":"# Tea brewing notes\n\nThe agent keeps the steeping times here.\n"},"choices":[{"id":"allow_once","label":"Do it"},{"id":"allow_session","label":"Allow this for the rest of the session"},{"id":"deny","label":"Don't"}]}
        """#

    static let approval = ConfirmRequest(
        id: .string("c8"), title: "May I change your knowledge base?",
        message: "kb new Analysis tea-brewing-notes", timeoutMs: 120_000,
        summary: "Create a new page in your knowledge base titled 'Tea brewing notes'.",
        action: ApprovalAction(tool: "kb", effect: .create,
                               command: "kb new Analysis tea-brewing-notes --title 'Tea brewing notes'",
                               path: "/Users/owner/kb/wiki/analyses/tea-brewing-notes.md", cwd: "/Users/owner/kb",
                               space: "home", mode: "act",
                               preview: "# Tea brewing notes\n\nThe agent keeps the steeping times here.\n"),
        choices: [ApprovalChoice(id: "allow_once", label: "Do it"),
                  ApprovalChoice(id: "allow_session", label: "Allow this for the rest of the session"),
                  ApprovalChoice(id: "deny", label: "Don't")])

    @Test("an approval request keeps every field, and the choices in the server's order")
    func approvalFields() throws {
        guard case let .confirmRequest(c) = try ProtocolCodec.decodeServer(Self.approvalWire) else {
            Issue.record("not a confirm_request")
            return
        }
        #expect(c.summary?.hasPrefix("Create a new page") == true)
        #expect(c.action?.effect == .create && c.action?.tool == "kb" && c.action?.cwd == "/Users/owner/kb")
        #expect(c.action?.preview?.contains("\n\nThe agent") == true, "the preview's newlines survive")
        #expect(c.choices.map(\.id) == ["allow_once", "allow_session", "deny"])
        #expect(c.choices.map(\.allows) == [true, true, false])
        #expect(c.timeoutMs == 120_000)
    }

    @Test("an old-style request (title and message only) decodes as before, with nothing new")
    func oldStyleRequest() throws {
        // An old-style request (title and message only), as early servers sent it.
        let wire = #"{"t":"confirm_request","id":"q1","title":"May I change your knowledge base?","message":"kb new Issue tea-brewing-notes --title Tea brewing notes","timeout_ms":20000}"#
        guard case let .confirmRequest(c) = try ProtocolCodec.decodeServer(wire) else {
            Issue.record("not a confirm_request")
            return
        }
        #expect(c.title == "May I change your knowledge base?" && c.message.hasPrefix("kb new Issue"))
        #expect(c.summary == nil && c.action == nil && c.choices.isEmpty)
        // And it encodes back without the new fields.
        let back = try JSONValue.parse(ProtocolCodec.encode(.confirmRequest(c)))
        #expect(back["summary"] == nil && back["action"] == nil && back["choices"] == nil)
    }

    @Test("approval fields are read tolerantly: wrong types are missing, unknown words are kept")
    func approvalTolerance() throws {
        func request(_ extra: String) throws -> ConfirmRequest {
            let m = try ProtocolCodec.decodeServer(#"{"t":"confirm_request","id":5,"title":"t","message":"m","# + extra + "}")
            guard case let .confirmRequest(c) = m else { throw ProtocolCodec.DecodeError.notAnObject }
            return c
        }
        #expect(try request(#""action":null"#).action == nil)
        #expect(try request(#""action":"rm -rf""#).action == nil, "a string is not an action")
        let partial = try request(#""action":{"tool":"bash","effect":"teleport","command":null,"cwd":7}"#).action
        #expect(partial == ApprovalAction(tool: "bash", effect: .other("teleport")))
        let choices = try request(#""choices":[{"id":"allow_once"},{"label":"no id"},"deny",{"id":3,"label":"Three"},{"id":"deny","label":"Don't"}]"#).choices
        #expect(choices == [ApprovalChoice(id: "allow_once"), ApprovalChoice(id: "3", label: "Three"),
                            ApprovalChoice(id: "deny", label: "Don't")])
        #expect(try request(#""choices":{"id":"allow_once"}"#).choices.isEmpty, "not a list: none offered")
        #expect(try request(#""summary":12"#).summary == nil)
        #expect(try request(#""timeout_ms":120000.0"#).timeoutMs == 120_000)
    }

    @Test("an unknown choice is never sent as a yes")
    func choiceAllows() {
        #expect(ApprovalChoice(id: "allow_once").allows && ApprovalChoice(id: "allow_session").allows)
        #expect(!ApprovalChoice(id: "deny").allows)
        #expect(!ApprovalChoice(id: "always").allows && !ApprovalChoice(id: "allowance").allows)
        #expect(ApprovalChoice(id: "allow_folder").allows, "a future allow_… is an allow")
    }

    @Test("confirm_cancel: numeric ids, a missing why, and no id at all")
    func confirmCancel() throws {
        #expect(try ProtocolCodec.decodeServer(#"{"t":"confirm_cancel","id":12}"#) == .confirmCancel(id: .int(12), why: ""))
        #expect(throws: ProtocolCodec.DecodeError.missingField(type: "confirm_cancel", field: "id")) {
            try ProtocolCodec.decodeServer(#"{"t":"confirm_cancel","why":"timed out"}"#)
        }
    }

    @Test("the server's side reads the choice, and a response without one")
    func choiceDecodes() throws {
        #expect(try ProtocolCodec.decodeClient(#"{"t":"confirm_response","id":"c1","confirmed":true,"choice":"allow_once"}"#)
            == .confirmResponse(id: .string("c1"), confirmed: true, choice: "allow_once"))
        #expect(try ProtocolCodec.decodeClient(#"{"t":"confirm_response","id":"c1","confirmed":false}"#)
            == .confirmResponse(id: .string("c1"), confirmed: false, choice: nil))
    }

    @Test("a diff preview with tabs, quotes, unicode and a cut marker survives both ways")
    func previewEscaping() throws {
        let diff = "--- a/todo.md\n+++ b/todo.md\n@@ -1,2 +1,2 @@\n-buy milk\n+buy \"oat\" milk\té 🙂\n\n… [cut: 2,180 more characters]"
        let request = ConfirmRequest(id: .string("c2"), title: "", message: "", timeoutMs: nil, summary: "s",
                                     action: ApprovalAction(tool: "edit", effect: .modify, path: "/tmp/todo.md",
                                                            preview: diff))
        let wire = ProtocolCodec.encode(.confirmRequest(request))
        #expect(!wire.contains("\n"), "one JSON message per line")
        #expect(try ProtocolCodec.decodeServer(wire) == .confirmRequest(request))
    }

    // MARK: Tolerance

    @Test("welcome accepts hold as an object, as the status endpoint shapes it")
    func welcomeHoldObject() throws {
        let m = try ProtocolCodec.decodeServer(
            #"{"t":"welcome","v":1,"session":"s9","space":"home","mode":"act","tier":"ask","state":"held","hold":{"phase":"held","why":"render"}}"#)
        guard case let .welcome(w) = m else { Issue.record("not a welcome: \(m)"); return }
        #expect(w.hold == HoldStatus(phase: .held, why: "render"))
        #expect(w.state == .held)
        #expect(w.session == "s9")
    }

    @Test("unknown types and fields are ignored, unknown enum values kept")
    func unknownsTolerated() throws {
        #expect(try ProtocolCodec.decodeServer(#"{"t":"emotion","v":"calm"}"#) == .unknown(type: "emotion"))
        #expect(try ProtocolCodec.decodeServer(#"{"t":"state","v":"dreaming","extra":[1,2]}"#) == .state(.other("dreaming")))
        #expect(try ProtocolCodec.decodeServer(#"{"t":"tool","phase":"retry","name":"bash"}"#)
            == .tool(ToolEvent(phase: .other("retry"), name: "bash")))
        #expect(try ProtocolCodec.decodeClient(#"{"t":"wave"}"#) == nil)
    }

    @Test("numeric ids and codes keep working")
    func numericScalars() throws {
        #expect(try ProtocolCodec.decodeServer(#"{"t":"confirm_request","id":12,"title":"t","message":"m"}"#)
            == .confirmRequest(ConfirmRequest(id: .int(12), title: "t", message: "m", timeoutMs: nil)))
        #expect(try ProtocolCodec.decodeServer(#"{"t":"error","code":503,"message":"busy"}"#)
            == .error(code: "503", message: "busy"))
        #expect(try ProtocolCodec.decodeClient(#"{"t":"played_ms","reply_id":"r2","ms":1234.0}"#)
            == .playedMs(replyID: "r2", ms: 1234))
        // The id goes back in the type it came in.
        #expect(ProtocolCodec.encode(.confirmResponse(id: .int(12), confirmed: true)).contains(#""id":12"#))
    }

    @Test("interrupt and end_of_turn without a reply id still decode")
    func optionalReplyID() throws {
        #expect(try ProtocolCodec.decodeServer(#"{"t":"interrupt"}"#) == .interrupt(replyID: nil))
        #expect(try ProtocolCodec.decodeServer(#"{"t":"end_of_turn"}"#) == .endOfTurn(replyID: nil))
    }

    @Test("malformed messages are errors, not crashes", arguments: [
        ("not json", ProtocolCodec.DecodeError.notJSON),
        ("[1,2]", .notAnObject),
        (#"{"v":1}"#, .missingType),
        (#"{"t":"audio_start","rate":24000}"#, .missingField(type: "audio_start", field: "reply_id")),
        (#"{"t":"transcript","final":true}"#, .missingField(type: "transcript", field: "text")),
        (#"{"t":"state"}"#, .missingField(type: "state", field: "v")),
    ])
    func malformed(wire: String, expected: ProtocolCodec.DecodeError) {
        #expect(throws: expected) { try ProtocolCodec.decodeServer(wire) }
    }

    @Test("text with quotes, newlines, control characters and emoji survives")
    func escaping() throws {
        let text = "she said \"hi\"\nthen\t\\left \u{01} 🙂 é"
        let wire = ProtocolCodec.encode(.text(text))
        #expect(!wire.contains("\n"), "one JSON message per line in logs")
        #expect(try ProtocolCodec.decodeClient(wire) == .text(text))
        // And the encoder agrees with Foundation's parser.
        let parsed = try JSONSerialization.jsonObject(with: Data(wire.utf8)) as? [String: Any]
        #expect(parsed?["text"] as? String == text)
    }

    @Test("close codes: retry only when it can help")
    func closeCodes() {
        #expect(!CloseCode.shouldReconnect(after: CloseCode.protocolError))
        #expect(!CloseCode.shouldReconnect(after: CloseCode.notAllowed))
        #expect(!CloseCode.shouldReconnect(after: CloseCode.replaced))
        #expect(CloseCode.shouldReconnect(after: CloseCode.abnormal))
        #expect(CloseCode.shouldReconnect(after: CloseCode.goingAway))
        #expect(CloseCode.shouldReconnect(after: CloseCode.normal))
    }
}
