import Foundation

public struct ReconnectBackoff: Sendable, Equatable {
    public var initial: Duration
    public var maximum: Duration
    public var multiplier: Double
    /// Each delay is scaled by a random factor in 1 ± jitter, so a fleet of clients (or a phone and a Mac) do not
    /// reconnect in lockstep after the server restarts.
    public var jitter: Double

    public init(initial: Duration = .milliseconds(500), maximum: Duration = .seconds(15), multiplier: Double = 2,
                jitter: Double = 0.2) {
        self.initial = initial
        self.maximum = maximum
        self.multiplier = multiplier
        self.jitter = jitter
    }

    /// The wait before attempt `attempt` (2 is the first retry). `random` is in 0...1.
    public func delay(beforeAttempt attempt: Int, random: Double) -> Duration {
        let retries = max(0, attempt - 2)
        let base = min(Double(maximum.nanos), Double(initial.nanos) * pow(multiplier, Double(retries)))
        let scaled = base * (1 + jitter * (2 * random - 1))
        return .nanoseconds(Int64(max(0, min(scaled, Double(maximum.nanos)))))
    }
}

public struct ConnectionConfig: Sendable {
    public var url: URL
    public var hello: Hello
    public var connectTimeout: Duration
    public var welcomeTimeout: Duration
    public var pingInterval: Duration
    public var idleTimeout: Duration
    public var backoff: ReconnectBackoff
    /// Audio kept while the socket is down (pre-connect buffering). Push-to-talk keeps the whole utterance; an open
    /// microphone keeps only the last moments, since stale audio would be heard as a late turn.
    public var maxBufferedAudio: Duration

    public init(url: URL, hello: Hello, connectTimeout: Duration = .seconds(5), welcomeTimeout: Duration = .seconds(10),
                pingInterval: Duration = ProtocolV1.pingInterval, idleTimeout: Duration = ProtocolV1.idleTimeout,
                backoff: ReconnectBackoff = ReconnectBackoff(), maxBufferedAudio: Duration? = nil) {
        self.url = url
        self.hello = hello
        self.connectTimeout = connectTimeout
        self.welcomeTimeout = welcomeTimeout
        self.pingInterval = pingInterval
        self.idleTimeout = idleTimeout
        self.backoff = backoff
        self.maxBufferedAudio = maxBufferedAudio ?? (hello.mic == .ptt ? .seconds(30) : .seconds(2))
    }
}

public struct CloseReport: Sendable, Equatable, CustomStringConvertible {
    public var code: Int
    public var reason: String
    /// True when this client closed the socket (stop, idle timeout, no welcome).
    public var local: Bool

    public init(code: Int, reason: String, local: Bool = false) {
        self.code = code
        self.reason = reason
        self.local = local
    }

    public var description: String {
        let what = CloseCode.describe(code)
        return reason.isEmpty ? what : "\(what) (\(reason))"
    }
}

public enum ConnectionState: Sendable, Equatable {
    case idle
    case connecting(attempt: Int)
    /// The socket is open and `hello` was sent; waiting for `welcome`.
    case handshaking
    case ready(session: String)
    /// Waiting before the next attempt.
    case waiting(nextAttempt: Int, delay: Duration, after: CloseReport)
    /// Not reconnecting: stopped by the client, or closed with 1002, 4403 or 4409.
    case stopped(CloseReport?)
}

public enum ConnectionEvent: Sendable {
    case state(ConnectionState)
    case message(ServerMessage)
    case audio(Data)
    /// A text message that did not decode; logged and otherwise ignored.
    case malformed(text: String, error: String)
}

public enum Outgoing: Sendable, Equatable {
    case message(ClientMessage)
    case audio(Data)

    var wire: WireMessage {
        switch self {
        case let .message(m): return .text(ProtocolCodec.encode(m))
        case let .audio(d): return .binary(d)
        }
    }
}

/// One protocol v1 connection that stays up: hello/welcome, ordered sending with pre-connect buffering, keepalive,
/// and reconnect with backoff.
///
/// Everything the client sends goes through one ordered stream (`send`, `sendAudio`), drained by a single sender. Until
/// `welcome` arrives the sender keeps items in an outbox, so a push-to-talk `start` and the audio captured while the
/// socket was still connecting go out first, in order, once the server is ready (pre-connect buffering, from the
/// research notes; a binary frame before `hello` is a protocol error, so nothing is sent before it). Events reach `onEvent`
/// in order from one receive loop.
public actor VoiceConnection {
    private enum PumpItem: Sendable {
        case outgoing(Outgoing)
        case ready(any WebSocketTransport, generation: Int)
        case down(generation: Int)
        case reset
    }

    public nonisolated let url: URL
    private var config: ConnectionConfig
    private let connector: any WebSocketConnecting
    private let onEvent: @Sendable (ConnectionEvent) -> Void
    private let pump: AsyncStream<PumpItem>.Continuation
    private var pumpStream: AsyncStream<PumpItem>?
    private var senderTask: Task<Void, Never>?
    private var runTask: Task<Void, Never>?
    private var transport: (any WebSocketTransport)?
    private var generation = 0
    private var stopRequested = false
    private var reconnectImmediately = false
    private var localClose: CloseReport?
    private var lastReceived: Nanos = 0
    private var pingCount = 0
    public private(set) var state: ConnectionState = .idle

    public init(config: ConnectionConfig, connector: any WebSocketConnecting = URLSessionConnector(),
                onEvent: @escaping @Sendable (ConnectionEvent) -> Void) {
        self.url = config.url
        self.config = config
        self.connector = connector
        self.onEvent = onEvent
        let (stream, continuation) = AsyncStream<PumpItem>.makeStream()
        self.pumpStream = stream
        self.pump = continuation
    }

    // MARK: Sending (any thread, ordered)

    public nonisolated func send(_ message: ClientMessage) {
        pump.yield(.outgoing(.message(message)))
    }

    public nonisolated func sendAudio(_ frame: Data) {
        pump.yield(.outgoing(.audio(frame)))
    }

    // MARK: Lifecycle

    public func start() {
        if senderTask == nil, let stream = pumpStream {
            pumpStream = nil
            senderTask = Task { await self.runSender(stream) }
        }
        guard runTask == nil else { return }
        stopRequested = false
        runTask = Task { await self.runLoop() }
    }

    /// Closes with 1000 and stops reconnecting. Queued items are dropped.
    public func stop() async {
        stopRequested = true
        localClose = CloseReport(code: CloseCode.normal, reason: "client stop", local: true)
        transport?.close(code: CloseCode.normal, reason: "client stop")
        runTask?.cancel()
        _ = await runTask?.value
        runTask = nil
        pump.yield(.reset)
        setState(.stopped(nil))
    }

    /// Stops for good and ends the sender.
    public func shutdown() async {
        await stop()
        pump.finish()
        _ = await senderTask?.value
        senderTask = nil
    }

    /// Changes the hello (for example `mic: vad` to `ptt`) and reconnects at once; the server builds its pipeline from
    /// the hello, so a mode change is a new connection.
    public func reconfigure(hello: Hello) {
        config.hello = hello
        config.maxBufferedAudio = hello.mic == .ptt ? .seconds(30) : .seconds(2)
        guard let transport, !stopRequested else { return }
        reconnectImmediately = true
        localClose = CloseReport(code: CloseCode.normal, reason: "reconfigure", local: true)
        transport.close(code: CloseCode.normal, reason: "reconfigure")
    }

    // MARK: Connect, receive, reconnect

    private func runLoop() async {
        var attempt = 0
        while !stopRequested && !Task.isCancelled {
            attempt += 1
            generation += 1
            let gen = generation
            localClose = nil
            setState(.connecting(attempt: attempt))
            var report: CloseReport
            var welcomed = false
            do {
                let t = try await connector.connect(to: config.url, timeout: config.connectTimeout)
                if stopRequested {
                    t.close(code: CloseCode.normal, reason: "client stop")
                    break
                }
                transport = t
                try await t.send(.text(ProtocolCodec.encode(.hello(config.hello))))
                setState(.handshaking)
                (report, welcomed) = await receiveLoop(t, generation: gen)
            } catch let closed as TransportClosed {
                report = localClose ?? CloseReport(code: closed.code, reason: closed.reason)
            } catch is CancellationError {
                report = localClose ?? CloseReport(code: CloseCode.normal, reason: "cancelled", local: true)
            } catch {
                report = CloseReport(code: CloseCode.abnormal, reason: (error as NSError).localizedDescription)
            }
            transport = nil
            pump.yield(.down(generation: gen))
            if welcomed { attempt = 0 }
            if stopRequested || Task.isCancelled { break }
            if !report.local && !CloseCode.shouldReconnect(after: report.code) {
                stopRequested = true
                setState(.stopped(report))
                break
            }
            if reconnectImmediately {
                reconnectImmediately = false
                continue
            }
            let delay = config.backoff.delay(beforeAttempt: attempt + 1, random: Double.random(in: 0...1))
            setState(.waiting(nextAttempt: attempt + 1, delay: delay, after: report))
            try? await Task.sleep(for: delay)
        }
        if case .stopped = state {} else if stopRequested { setState(.stopped(nil)) }
    }

    private func receiveLoop(_ t: any WebSocketTransport, generation gen: Int) async -> (CloseReport, Bool) {
        var welcomed = false
        lastReceived = MonotonicClock.now()
        let openedAt = lastReceived
        let watchdog = Task { await self.watch(t, generation: gen, openedAt: openedAt) }
        defer { watchdog.cancel() }
        while true {
            let message: WireMessage
            do {
                message = try await t.receive()
            } catch let closed as TransportClosed {
                return (localClose ?? CloseReport(code: closed.code, reason: closed.reason), welcomed)
            } catch {
                return (localClose ?? CloseReport(code: CloseCode.abnormal, reason: "\(error)"), welcomed)
            }
            lastReceived = MonotonicClock.now()
            switch message {
            case let .binary(data):
                onEvent(.audio(data))
            case let .text(text):
                let decoded: ServerMessage
                do {
                    decoded = try ProtocolCodec.decodeServer(text)
                } catch {
                    onEvent(.malformed(text: text, error: "\(error)"))
                    continue
                }
                if case let .welcome(w) = decoded, !welcomed {
                    welcomed = true
                    // The sender flushes the outbox before anything sent after this point.
                    pump.yield(.ready(t, generation: gen))
                    setState(.ready(session: w.session))
                }
                onEvent(.message(decoded))
            }
        }
    }

    /// Keepalive and timeouts for one socket: a ping every 15 s once ready, no welcome within the limit, and 60 s
    /// without any message (PROTOCOL.md "Keepalive").
    private func watch(_ t: any WebSocketTransport, generation gen: Int, openedAt: Nanos) async {
        var nextPing = openedAt + config.pingInterval.nanos
        while !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled, gen == generation else { return }
            let now = MonotonicClock.now()
            if case .handshaking = state, now - openedAt > config.welcomeTimeout.nanos {
                closeLocally(t, reason: "no welcome within \(config.welcomeTimeout.milliseconds) ms")
                return
            }
            if now - lastReceived > config.idleTimeout.nanos {
                closeLocally(t, reason: "nothing received for \(config.idleTimeout.milliseconds) ms")
                return
            }
            if case .ready = state, now >= nextPing {
                pingCount += 1
                pump.yield(.outgoing(.message(.ping(n: pingCount))))
                nextPing = now + config.pingInterval.nanos
            }
        }
    }

    private func closeLocally(_ t: any WebSocketTransport, reason: String) {
        localClose = CloseReport(code: CloseCode.normal, reason: reason, local: true)
        t.close(code: CloseCode.normal, reason: reason)
    }

    private func setState(_ new: ConnectionState) {
        state = new
        onEvent(.state(new))
    }

    // MARK: The single sender

    private func runSender(_ stream: AsyncStream<PumpItem>) async {
        var current: (any WebSocketTransport)?
        var currentGeneration = -1
        var outbox: [Outgoing] = []
        var bufferedAudioBytes = 0

        func keep(_ item: Outgoing) {
            if case .message(.ping) = item { return }  // a ping for a dead socket means nothing
            outbox.append(item)
            if case let .audio(d) = item {
                bufferedAudioBytes += d.count
                let limit = Int(Double(config.maxBufferedAudio.nanos) / 1e9 * Double(ProtocolV1.captureSampleRate * 2))
                while bufferedAudioBytes > limit, let i = outbox.firstIndex(where: { if case .audio = $0 { true } else { false } }) {
                    if case let .audio(old) = outbox.remove(at: i) { bufferedAudioBytes -= old.count }
                }
            }
        }

        for await item in stream {
            switch item {
            case let .ready(t, gen):
                current = t
                currentGeneration = gen
                let pending = outbox
                outbox.removeAll()
                bufferedAudioBytes = 0
                for o in pending {
                    do {
                        try await t.send(o.wire)
                    } catch {
                        current = nil
                        break
                    }
                }
            case let .down(gen):
                if gen == currentGeneration { current = nil }
            case .reset:
                current = nil
                outbox.removeAll()
                bufferedAudioBytes = 0
            case let .outgoing(o):
                if let t = current {
                    do {
                        try await t.send(o.wire)
                    } catch {
                        // The socket died under this item. Audio in flight is lost (PROTOCOL.md "Reconnect"); a control
                        // message (typed text, stop, a confirmation) waits for the next session instead.
                        current = nil
                        if case .message = o { keep(o) }
                    }
                } else {
                    keep(o)
                }
            }
        }
    }
}
