import Foundation
import Synchronization
import Testing
@testable import LocalVoiceKit

/// A scripted socket: records what the client sent, and lets the test deliver server messages or close it.
final class FakeTransport: WebSocketTransport, @unchecked Sendable {
    // Invariant: `iterator` is only touched by the connection's single receive loop.
    private let stream: AsyncStream<Result<WireMessage, TransportClosed>>
    private let feed: AsyncStream<Result<WireMessage, TransportClosed>>.Continuation
    private nonisolated(unsafe) var iterator: AsyncStream<Result<WireMessage, TransportClosed>>.Iterator?
    let sent = Mutex<[WireMessage]>([])
    let closedWith = Mutex<Int?>(nil)

    init() {
        (stream, feed) = AsyncStream.makeStream()
    }

    func send(_ message: WireMessage) async throws {
        if let code = closedWith.withLock({ $0 }) { throw TransportClosed(code: code, reason: "closed") }
        sent.withLock { $0.append(message) }
    }

    func receive() async throws -> WireMessage {
        if iterator == nil { iterator = stream.makeAsyncIterator() }
        guard let next = await iterator!.next() else { throw TransportClosed(code: CloseCode.abnormal, reason: "") }
        switch next {
        case let .success(m): return m
        case let .failure(closed): throw closed
        }
    }

    func close(code: Int, reason: String) {
        closedWith.withLock { $0 = $0 ?? code }
        feed.yield(.failure(TransportClosed(code: code, reason: reason)))
    }

    // Test side
    func deliver(_ message: ServerMessage) { feed.yield(.success(.text(ProtocolCodec.encode(message)))) }
    func deliverAudio(_ data: Data) { feed.yield(.success(.binary(data))) }
    func feedRaw(_ text: String) { feed.yield(.success(.text(text))) }
    func serverClose(_ code: Int) {
        closedWith.withLock { $0 = $0 ?? code }
        feed.yield(.failure(TransportClosed(code: code, reason: "server")))
    }

    var sentMessages: [WireMessage] { sent.withLock { $0 } }
    var sentTypes: [String] {
        sentMessages.map {
            switch $0 {
            case let .text(t): return (try? ProtocolCodec.decodeClient(t))??.type ?? "?"
            case let .binary(d): return "audio:\(d.first ?? 0)"
            }
        }
    }
}

final class FakeConnector: WebSocketConnecting, @unchecked Sendable {
    let transports = Mutex<[FakeTransport]>([])
    let failuresLeft = Mutex(0)

    func connect(to url: URL, timeout: Duration) async throws -> any WebSocketTransport {
        if failuresLeft.withLock({ n in defer { n = max(0, n - 1) }; return n > 0 }) {
            throw URLError(.cannotConnectToHost)
        }
        let t = FakeTransport()
        transports.withLock { $0.append(t) }
        return t
    }

    var count: Int { transports.withLock { $0.count } }
    var latest: FakeTransport? { transports.withLock { $0.last } }
}

final class EventRecorder: Sendable {
    let events = Mutex<[ConnectionEvent]>([])
    func record(_ e: ConnectionEvent) { events.withLock { $0.append(e) } }
    var states: [ConnectionState] {
        events.withLock { $0.compactMap { if case let .state(s) = $0 { s } else { nil } } }
    }
}

func waitUntil(_ timeout: Duration = .seconds(3), _ condition: @escaping @Sendable () async -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + timeout
    while ContinuousClock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return await condition()
}

@Suite("Connection: hello, pre-connect buffering, reconnect")
struct ConnectionTests {
    let url = URL(string: "ws://127.0.0.1:1/v1/voice")!

    func make(mic: MicMode = .ptt, pingInterval: Duration = .seconds(15), idle: Duration = .seconds(60),
              welcomeTimeout: Duration = .seconds(10), buffered: Duration? = nil)
        -> (VoiceConnection, FakeConnector, EventRecorder) {
        let connector = FakeConnector()
        let recorder = EventRecorder()
        let config = ConnectionConfig(url: url, hello: Hello(client: .test, device: "t1", mic: mic),
                                      welcomeTimeout: welcomeTimeout, pingInterval: pingInterval, idleTimeout: idle,
                                      backoff: ReconnectBackoff(initial: .milliseconds(20), maximum: .milliseconds(50)),
                                      maxBufferedAudio: buffered)
        let connection = VoiceConnection(config: config, connector: connector) { recorder.record($0) }
        return (connection, connector, recorder)
    }

    static let welcome = ServerMessage.welcome(Welcome(session: "s1", space: "home", mode: "conversation", tier: "ask",
                                                       state: .listening, hold: HoldStatus(phase: .open)))

    func frame(_ tag: UInt8) -> Data { Data([tag]) + Data(count: 639) }

    @Test("hello goes first; start and audio captured before the socket was up follow welcome, in order")
    func preConnectBuffering() async throws {
        let (connection, connector, _) = make()
        connection.send(.start)
        for i in 1...3 { connection.sendAudio(frame(UInt8(i))) }
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        let t = try #require(connector.latest)
        #expect(await waitUntil { t.sentTypes == ["hello"] })
        try await Task.sleep(for: .milliseconds(50))
        #expect(t.sentTypes == ["hello"], "nothing but hello before welcome")
        connection.send(.stop)
        t.deliver(Self.welcome)
        #expect(await waitUntil { t.sentTypes.count == 6 })
        #expect(t.sentTypes == ["hello", "start", "audio:1", "audio:2", "audio:3", "stop"])
        guard case let .text(hello) = t.sentMessages[0] else { Issue.record("hello not text"); return }
        #expect(try ProtocolCodec.decodeClient(hello) == .hello(Hello(client: .test, device: "t1", mic: .ptt)))
        await connection.shutdown()
    }

    @Test("an abnormal close reconnects with the same hello, and the next session flushes what queued meanwhile")
    func reconnects() async throws {
        let (connection, connector, recorder) = make()
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        connector.latest!.deliver(Self.welcome)
        #expect(await waitUntil { if case .ready = await connection.state { true } else { false } })
        connector.latest!.serverClose(CloseCode.abnormal)
        connection.send(.text("typed while down"))
        #expect(await waitUntil { connector.count == 2 })
        let second = connector.latest!
        #expect(await waitUntil { second.sentTypes == ["hello"] })
        second.deliver(Self.welcome)
        #expect(await waitUntil { second.sentTypes == ["hello", "text"] })
        #expect(recorder.states.contains { if case .waiting(_, _, let r) = $0 { r.code == 1006 } else { false } })
        await connection.shutdown()
    }

    @Test("1002, 4403 and 4409 stop reconnecting", arguments: [CloseCode.protocolError, CloseCode.notAllowed,
                                                               CloseCode.replaced])
    func permanentCloses(code: Int) async throws {
        let (connection, connector, _) = make()
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        connector.latest!.serverClose(code)
        #expect(await waitUntil { if case let .stopped(r) = await connection.state { r?.code == code } else { false } })
        try await Task.sleep(for: .milliseconds(150))
        #expect(connector.count == 1, "no second connection")
        await connection.shutdown()
    }

    @Test("connect failures back off and retry")
    func connectFailures() async throws {
        let (connection, connector, recorder) = make()
        connector.failuresLeft.withLock { $0 = 2 }
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        #expect(recorder.states.filter { if case .waiting = $0 { true } else { false } }.count == 2)
        await connection.shutdown()
    }

    @Test("pings once ready; silence from the server closes and reconnects")
    func keepaliveAndIdle() async throws {
        let (connection, connector, _) = make(pingInterval: .milliseconds(300), idle: .milliseconds(1200))
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        let t = connector.latest!
        t.deliver(Self.welcome)
        #expect(await waitUntil { t.sentTypes.filter { $0 == "ping" }.count >= 2 })
        #expect(await waitUntil(.seconds(4)) { connector.count == 2 }, "idle timeout reconnects")
        #expect(t.closedWith.withLock { $0 } == CloseCode.normal)
        await connection.shutdown()
    }

    @Test("no welcome within the limit closes and retries")
    func welcomeTimeout() async throws {
        let (connection, connector, _) = make(welcomeTimeout: .milliseconds(300))
        await connection.start()
        #expect(await waitUntil(.seconds(3)) { connector.count == 2 })
        await connection.shutdown()
    }

    @Test("an open microphone keeps only the last moments while the socket is down")
    func openMicBufferLimit() async throws {
        let (connection, connector, _) = make(mic: .vad, buffered: .milliseconds(100))
        for i in 1...20 { connection.sendAudio(frame(UInt8(i))) }  // 400 ms
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        let t = connector.latest!
        t.deliver(Self.welcome)
        #expect(await waitUntil { t.sentTypes.count == 6 })
        #expect(t.sentTypes == ["hello", "audio:16", "audio:17", "audio:18", "audio:19", "audio:20"])
        await connection.shutdown()
    }

    @Test("server audio and messages reach the event handler in order")
    func eventsInOrder() async throws {
        let (connection, connector, recorder) = make()
        await connection.start()
        #expect(await waitUntil { connector.count == 1 })
        let t = connector.latest!
        t.deliver(Self.welcome)
        t.deliver(.audioStart(replyID: "r1", rate: 24000))
        t.deliverAudio(Data([1, 2]))
        t.deliver(.audioEnd(replyID: "r1"))
        t.feedRaw("{not json")
        #expect(await waitUntil { recorder.events.withLock { $0.count } >= 8 })
        let kinds = recorder.events.withLock { events in
            events.map { e -> String in
                switch e {
                case let .state(s): return "state:\(s)"
                case let .message(m): return m.type
                case .audio: return "audio"
                case .malformed: return "malformed"
                }
            }
        }
        let tail = Array(kinds.drop { !$0.hasPrefix("state:ready") })
        #expect(tail == ["state:ready(session: \"s1\")", "welcome", "audio_start", "audio", "audio_end", "malformed"])
        await connection.shutdown()
    }

    @Test func backoffGrowsAndCaps() {
        let b = ReconnectBackoff(initial: .milliseconds(500), maximum: .seconds(15), multiplier: 2, jitter: 0)
        #expect(b.delay(beforeAttempt: 2, random: 0.5) == .milliseconds(500))
        #expect(b.delay(beforeAttempt: 3, random: 0.5) == .seconds(1))
        #expect(b.delay(beforeAttempt: 4, random: 0.5) == .seconds(2))
        #expect(b.delay(beforeAttempt: 20, random: 0.5) == .seconds(15))
        let j = ReconnectBackoff(jitter: 0.2)
        #expect(j.delay(beforeAttempt: 2, random: 0) == .milliseconds(400))
        #expect(j.delay(beforeAttempt: 2, random: 1) == .milliseconds(600))
    }
}
