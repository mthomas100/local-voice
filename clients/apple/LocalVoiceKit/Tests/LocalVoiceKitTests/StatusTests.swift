import Foundation
import Synchronization
import Testing
@testable import LocalVoiceKit

/// The shape the orchestrator serves (orchestrator/local_voice/server.py `status()` and agent.py, read 2026-10-05):
/// PROTOCOL.md's fields plus `turns`, `children`, `spaces`, `digest_pending`, `speech`, `uptime_s`; `tool` an object
/// with `since`; the latencies floats.
let orchestratorStatus = """
    {"v": 1, "state": "thinking", "hold": {"phase": "open", "why": ""}, "space": "atlas", "mode": "conversation",
     "tier": "trusted", "model": "local/qwen38",
     "tool": {"name": "read", "label": "reading your journal", "since": 1791234567.5},
     "last_turn": {"eos_to_first_audio_ms": 843.7, "stt_ms": 41.2, "llm_ttft_ms": 301.0, "tts_first_audio_ms": 129.9,
                   "tools": ["read"], "space": "atlas"},
     "turns": 7, "children": {"atlas": {"alive": true, "busy": true}},
     "spaces": {"home": {"name": "home", "description": "your Mac", "root": "/Users/x", "tier": "ask",
                         "model": "local/qwen38", "tools": ["read"], "act_tools": ["bash"], "skills": ["wiki-query"]},
                "atlas": {"name": "atlas", "description": "your atlas journal", "root": "/x/atlas", "tier": "trusted",
                          "model": "local/qwen38", "tools": ["read", "bash"], "act_tools": [], "skills": "auto"}},
     "clients": [{"device": "mac", "client": "mac", "connected_s": 120}, {"device": "browser-1", "client": "browser",
                  "connected_s": 4}],
     "digest_pending": false, "speech": {"stt": {"impl": "X", "model": "m", "load_s": 1.2}}, "uptime_s": 3600}
    """

/// A status source that counts fetches and can hold one until released.
actor FakeStatusSource: StatusFetching {
    private(set) var count = 0
    private var gate: CheckedContinuation<Void, Never>?
    private var holding = false
    var answer: Result<ServerStatus, StatusFetchError> = .success(ServerStatus(space: "home"))

    func hold() { holding = true }

    func release() {
        holding = false
        gate?.resume()
        gate = nil
    }

    func set(_ answer: Result<ServerStatus, StatusFetchError>) { self.answer = answer }

    nonisolated func fetchStatus() async throws -> ServerStatus { try await fetch() }

    private func fetch() async throws -> ServerStatus {
        count += 1
        if holding { await withCheckedContinuation { gate = $0 } }
        return try answer.get()
    }
}

@Suite("Status endpoint and dashboard")
struct StatusTests {
    @Test("PROTOCOL.md's example decodes")
    func protocolExample() throws {
        let s = try ServerStatus.decode(JSONValue.parse("""
            {"v": 1, "state": "idle", "space": "home", "mode": "conversation", "tier": "ask",
             "model": "local/qwen38", "hold": {"phase": "open", "why": ""},
             "tool": null, "last_turn": {"eos_to_first_audio_ms": 0, "stt_ms": 0, "llm_ttft_ms": 0, "tts_first_audio_ms": 0},
             "clients": [{"device": "iphone", "connected_s": 0}]}
            """))
        #expect(s.version == 1 && s.state == .idle && s.space == "home" && s.mode == "conversation" && s.tier == "ask")
        #expect(s.model == "local/qwen38" && s.hold == HoldStatus(phase: .open) && s.tool == nil)
        #expect(s.lastTurn?.isEmpty == true, "zeros before the first turn")
        #expect(s.clients == [ServerStatus.Client(device: "iphone", connectedSeconds: 0)])
        #expect(s.spaces.isEmpty)
    }

    @Test("the orchestrator's status decodes, with its spaces sorted by name")
    func orchestratorShape() throws {
        let s = try ServerStatus.decode(JSONValue.parse(orchestratorStatus))
        #expect(s.state == .thinking && s.space == "atlas" && s.tier == "trusted" && s.turns == 7)
        #expect(s.tool == ServerStatus.RunningTool(name: "read", label: "reading your journal"))
        let t = try #require(s.lastTurn)
        #expect(t.eosToFirstAudioMs == 843.7 && t.sttMs == 41.2 && t.llmFirstTokenMs == 301 && t.ttsFirstAudioMs == 129.9)
        #expect(t.tools == ["read"] && t.space == "atlas" && !t.isEmpty)
        #expect(s.spaces.map(\.name) == ["atlas", "home"])
        #expect(s.spaces[0] == ServerStatus.Space(name: "atlas", description: "your atlas journal", tier: "trusted",
                                                  model: "local/qwen38"))
        #expect(s.clients.map(\.kind) == ["mac", "browser"] && s.uptimeSeconds == 3600)
        // The log keeps what was read, in the wire's names.
        let logged = s.json
        #expect(logged["last_turn"]?["llm_ttft_ms"]?.doubleValue == 301)
        #expect(logged["spaces"]?.arrayValue?.count == 2 && logged["tool"]?["label"]?.stringValue == "reading your journal")
    }

    @Test("before its agent hub is up the orchestrator serves less; a bare hold phase reads; garbage does not")
    func partialAndTolerant() throws {
        let early = try ServerStatus.decode(JSONValue.parse("""
            {"v": 1, "state": "held", "hold": {"phase": "held", "why": "film render"}, "clients": [],
             "digest_pending": false, "uptime_s": 2}
            """))
        #expect(early.state == .held && early.hold == HoldStatus(phase: .held, why: "film render"))
        #expect(early.space == nil && early.lastTurn == nil && early.model == nil)
        let bare = try ServerStatus.decode(JSONValue.parse(#"{"v": 1, "hold": "draining"}"#))
        #expect(bare.hold == HoldStatus(phase: .draining))
        #expect(throws: ServerStatus.DecodeError.self) { try ServerStatus.decode(JSONValue.parse("[1, 2]")) }
        #expect(throws: JSONValue.ParseError.self) { try ServerStatus.decode(Data("not json".utf8)) }
    }

    @Test("the status URL is the voice URL's host and port, over HTTP")
    func statusURL() {
        func status(_ s: String) -> String? { ProtocolV1.statusURL(forVoiceURL: URL(string: s)!)?.absoluteString }
        #expect(status("ws://127.0.0.1:8770/v1/voice") == "http://127.0.0.1:8770/v1/status")
        #expect(status("wss://mac.example-tailnet.ts.net/v1/voice?x=1") == "https://mac.example-tailnet.ts.net/v1/status")
        #expect(status("ws://[fd7a:115c:a1e0::1]:8770/v1/voice") == "http://[fd7a:115c:a1e0::1]:8770/v1/status")
        #expect(status("ftp://x/v1/voice") == nil)
    }

    @Test("what calls for a fresh status: connected, a turn ended, the space changed, a switch unanswered")
    func reasons() {
        let request = SwitchRequest(.space, "atlas")
        #expect(StatusMonitor.reason(for: .connection(.ready(session: "s1"))) == .connected)
        #expect(StatusMonitor.reason(for: .server(.endOfTurn(replyID: "r1"))) == .turnEnded)
        #expect(StatusMonitor.reason(for: .server(.space(SpaceInfo(name: "atlas")))) == .spaceChanged)
        #expect(StatusMonitor.reason(for: .switchOutcome(.noAnswer(request))) == .switchUnanswered)
        // Pushed live, so no fetch: state, tool, hold; nor the status itself.
        #expect(StatusMonitor.reason(for: .server(.state(.thinking))) == nil)
        #expect(StatusMonitor.reason(for: .server(.tool(ToolEvent(phase: .start, name: "read")))) == nil)
        #expect(StatusMonitor.reason(for: .server(.hold(HoldStatus(phase: .held)))) == nil)
        #expect(StatusMonitor.reason(for: .status(.connected, ServerStatus())) == nil)
        #expect(StatusMonitor.reason(for: .switchOutcome(.refused(request, reason: "no"))) == nil)
    }

    @Test("requests close together share a fetch; one during a fetch makes exactly one more")
    func monitorCoalesces() async throws {
        let source = FakeStatusSource()
        let delivered = Mutex<[StatusMonitor.Reason]>([])
        let monitor = StatusMonitor(fetcher: source, debounce: .milliseconds(30)) { reason, _ in
            delivered.withLock { $0.append(reason) }
        }
        await monitor.request(.connected)
        await monitor.request(.spaceChanged)  // welcome and space back to back
        try await Task.sleep(for: .milliseconds(150))
        #expect(await source.count == 1)
        #expect(delivered.withLock { $0 } == [.connected])

        await source.hold()
        await monitor.request(.turnEnded)
        try await Task.sleep(for: .milliseconds(80))  // the fetch is running and held
        await monitor.request(.spaceChanged)
        await monitor.request(.requested)
        await source.release()
        try await Task.sleep(for: .milliseconds(200))
        #expect(await source.count == 3, "the two requests during the held fetch made one more fetch")
        #expect(delivered.withLock { $0 } == [.connected, .turnEnded, .spaceChanged])
    }

    @Test("a failed fetch is delivered as a failure")
    func monitorFailure() async throws {
        let source = FakeStatusSource()
        await source.set(.failure(.http(404)))
        let failures = Mutex<[String]>([])
        let monitor = StatusMonitor(fetcher: source, debounce: .milliseconds(5)) { _, result in
            if case let .failure(e) = result { failures.withLock { $0.append(VoiceClient.describe(e)) } }
        }
        await monitor.request(.requested)
        try await Task.sleep(for: .milliseconds(100))
        #expect(failures.withLock { $0 } == ["the server has no /v1/status (404)"])
    }
}

@Suite("Dashboard")
struct DashboardTests {
    let welcome = ClientEvent.server(.welcome(Welcome(session: "s1", space: "home", mode: "conversation", tier: "ask",
                                                      state: .listening, hold: HoldStatus(phase: .open))))
    var status: ServerStatus { try! ServerStatus.decode(JSONValue.parse(orchestratorStatus)) }

    @Test("welcome and space say where; the status adds the model, the latency breakdown and the spaces")
    func foldsMessagesAndStatus() {
        var d = Dashboard()
        d.apply(welcome)
        d.apply(.server(.space(SpaceInfo(name: "home", mode: "conversation", tier: "ask", description: "your Mac"))))
        #expect(d.space == "home" && d.spaceTitle == "your Mac" && d.whereLine == "home · conversation")
        #expect(d.model == nil && d.lastTurn == nil)
        var s = status
        s.space = "home"
        s.tier = "ask"
        d.apply(.status(.connected, s), now: Date(timeIntervalSince1970: 100))
        #expect(d.model == "local/qwen38" && d.spaces.count == 2 && d.clients.count == 2 && d.turns == 7)
        #expect(d.statusAt == Date(timeIntervalSince1970: 100) && d.statusProblem == nil)
        #expect(d.latencyLine == "844 ms to first audio (STT 41, LLM 301, TTS 130)")
        d.apply(.latency(TurnLatency(reply: "r1", stopToAudioStartMs: 870, stopToPlaybackMs: 935,
                                     releaseToPlaybackMs: 968)))
        #expect(d.latencyLine == "844 ms to first audio (STT 41, LLM 301, TTS 130); 968 ms from release to sound here")
    }

    @Test("once the session is up, state, hold and tool follow messages, not an older status")
    func messagesWinForLiveFields() {
        var d = Dashboard()
        d.apply(welcome)
        d.apply(.server(.state(.speaking)))
        d.apply(.server(.hold(HoldStatus(phase: .held, why: "film render"))))
        d.apply(.server(.tool(ToolEvent(phase: .end, name: "read", ok: true))))
        d.apply(.status(.turnEnded, status))  // says thinking, open, reading
        #expect(d.state == .speaking && d.hold == HoldStatus(phase: .held, why: "film render") && d.tool == nil)
        #expect(d.holdLine == "The GPU is held: film render")
        // Space and mode follow the latest word: another device may have switched the shared agent.
        #expect(d.space == "atlas" && d.spaceTitle == "your atlas journal" && d.tier == "trusted")
    }

    @Test("before any session the status is all there is; a tool running at connect shows until its end")
    func statusBeforeSession() {
        var d = Dashboard()
        d.apply(.status(.requested, status))
        #expect(d.state == .thinking && d.tool?.label == "reading your journal" && d.space == "atlas")
        d.apply(welcome)
        d.apply(.status(.connected, status))
        #expect(d.tool?.name == "read", "no tool message yet on this connection: the status's tool stands")
        d.apply(.server(.tool(ToolEvent(phase: .end, name: "read", ok: true))))
        d.apply(.status(.turnEnded, status))
        #expect(d.tool == nil)
    }

    @Test("entering another space drops what belonged to the old one until the server says")
    func spaceChange() {
        var d = Dashboard()
        d.apply(welcome)
        var s = status
        s.space = "home"
        d.apply(.status(.connected, s))
        d.apply(.server(.space(SpaceInfo(name: "kb", mode: "conversation", tier: "readonly"))))
        #expect(d.space == "kb" && d.spaceDescription == nil && d.model == nil && d.spaceTitle == "kb")
        d.apply(.server(.space(SpaceInfo(name: "atlas", mode: "act", tier: "trusted"))))
        #expect(d.spaceDescription == "your atlas journal" && d.model == "local/qwen38", "known from the status's spaces")
        #expect(d.whereLine == "atlas · act")
    }

    @Test("a switch shows while pending, then what came of it")
    func switchLines() {
        var d = Dashboard()
        let atlas = SwitchRequest(.space, "atlas")
        let act = SwitchRequest(.mode, "act")
        d.apply(.switchRequested(atlas))
        #expect(d.pendingSwitch == atlas && d.switchLine == "Switching to atlas…")
        d.apply(.switchOutcome(.switched(atlas, SpaceInfo(name: "atlas", description: "your atlas journal"))))
        #expect(d.pendingSwitch == nil && d.switchLine == "Now in your atlas journal.")
        d.apply(.switchRequested(act))
        #expect(d.switchLine == "Switching to act mode…")
        d.apply(.switchOutcome(.refused(act, reason: "act mode needs a trusted space")))
        #expect(d.switchLine == "Could not switch to act mode: act mode needs a trusted space.")
        d.apply(.switchOutcome(.noAnswer(atlas)))
        #expect(d.switchLine == "No answer to switching to atlas; the server may not support switching yet.")
        d.apply(.statusFailed(.connected, "not allowed (403)"))
        #expect(d.statusProblem == "not allowed (403)")
    }
}
