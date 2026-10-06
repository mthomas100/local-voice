import Foundation

/// The protocol v1 client: a connection, an audio engine and the state machine between them.
///
/// Methods may be called from any thread (the apps call them from the main actor). Events arrive in order on
/// `events`. Internally everything that touches client state runs on one serial queue; the audio engine starts and
/// stops on another, because those calls block for up to a few hundred milliseconds with voice processing on.
public final class VoiceClient: @unchecked Sendable {
    // Invariant for @unchecked Sendable: `core` is only touched on `queue`; `audio` start/stop only on `setupQueue`;
    // the rest is immutable after init.

    public struct Configuration: Sendable {
        public var url: URL
        public var hello: Hello
        public var settings: ClientSettings
        public var connectTimeout: Duration
        public var backoff: ReconnectBackoff
        /// `/v1/status` for the dashboard (derived from `url`); nil fetches nothing.
        public var statusURL: URL?

        public init(url: URL, hello: Hello, settings: ClientSettings? = nil, connectTimeout: Duration = .seconds(5),
                    backoff: ReconnectBackoff = ReconnectBackoff(), fetchStatus: Bool = true) {
            self.url = url
            self.hello = hello
            self.settings = settings ?? ClientSettings(mic: hello.mic)
            self.connectTimeout = connectTimeout
            self.backoff = backoff
            self.statusURL = fetchStatus ? ProtocolV1.statusURL(forVoiceURL: url) : nil
        }
    }

    public let events: AsyncStream<ClientEvent>
    public let configuration: Configuration
    private let eventSink: AsyncStream<ClientEvent>.Continuation
    private let queue = DispatchQueue(label: "lv.client", qos: .userInteractive)
    private let setupQueue = DispatchQueue(label: "lv.audio.setup", qos: .userInitiated)
    private let connection: VoiceConnection
    private let audio: any AudioIO
    private let box: CoreBox
    private let ticker: DispatchSourceTimer
    private let statusMonitor: StatusMonitor?

    private final class CoreBox: @unchecked Sendable {
        // Written once in init, then only read on the client queue.
        var core: ClientCore?
    }

    /// `statusFetcher` replaces the HTTP fetch of `configuration.statusURL` (tests).
    public init(configuration: Configuration, audio: any AudioIO,
                connector: any WebSocketConnecting = URLSessionConnector(),
                statusFetcher: (any StatusFetching)? = nil) {
        self.configuration = configuration
        self.audio = audio
        let (events, sink) = AsyncStream<ClientEvent>.makeStream(bufferingPolicy: .unbounded)
        self.events = events
        self.eventSink = sink
        let queue = self.queue
        let box = CoreBox()
        self.box = box
        let connection = VoiceConnection(
            config: ConnectionConfig(url: configuration.url, hello: configuration.hello,
                                     connectTimeout: configuration.connectTimeout, backoff: configuration.backoff),
            connector: connector,
            onEvent: { event in queue.async { box.core?.connectionEvent(event) } })
        self.connection = connection
        // The dashboard's status: fetched when an event calls for it, delivered on the client queue like every event.
        let fetcher: (any StatusFetching)? = statusFetcher
            ?? configuration.statusURL.map { HTTPStatusFetcher(url: $0) as any StatusFetching }
        let monitor = fetcher.map { fetcher in
            StatusMonitor(fetcher: fetcher) { reason, result in
                queue.async {
                    switch result {
                    case let .success(status): sink.yield(.status(reason, status))
                    case let .failure(error): sink.yield(.statusFailed(reason, VoiceClient.describe(error)))
                    }
                }
            }
        }
        self.statusMonitor = monitor
        let setupQueue = self.setupQueue
        let control = AudioControl(
            start: {
                setupQueue.async {
                    let status: AudioEngineStatus
                    do {
                        status = .running(try audio.start { samples, at in
                            queue.async { box.core?.captured(samples, at: at) }
                        })
                    } catch {
                        status = .failed("\(error)")
                    }
                    queue.async { box.core?.audioStatus(status) }
                }
            },
            stop: { setupQueue.async { audio.stop() } })
        box.core = ClientCore(
            settings: configuration.settings, queue: queue, clock: { MonotonicClock.now() },
            player: audio.player, cue: audio.cue, audio: control,
            send: { outgoing in
                switch outgoing {
                case let .message(m): connection.send(m)
                case let .audio(d): connection.sendAudio(d)
                }
            },
            emit: { event in
                sink.yield(event)
                if let monitor, let reason = StatusMonitor.reason(for: event) {
                    Task { await monitor.request(reason) }
                }
            })
        audio.onStopped = { reason in
            queue.async { box.core?.audioStatus(.failed(reason)) }
        }
        ticker = DispatchSource.makeTimerSource(queue: queue)
        ticker.schedule(deadline: .now() + .milliseconds(250), repeating: .milliseconds(250))
        ticker.setEventHandler { box.core?.tick(at: MonotonicClock.now()) }
        ticker.resume()
    }

    deinit {
        ticker.cancel()
    }

    private func onCore(_ body: @escaping @Sendable (ClientCore) -> Void) {
        let box = self.box
        queue.async { box.core.map(body) }
    }

    // MARK: Connection

    public func connect() {
        let connection = self.connection
        Task { await connection.start() }
    }

    /// Closes with 1000 and stops reconnecting; the audio engine stops too.
    public func disconnect() async {
        await connection.stop()
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let box = self.box
            queue.async {
                box.core?.stopHandsFree()
                c.resume()
            }
        }
        await stopAudio()
    }

    public func shutdown() async {
        await disconnect()
        await connection.shutdown()
        eventSink.finish()
    }

    /// Switches between open microphone and push-to-talk: a new hello, so a new connection (same device, so the server
    /// resumes the same agent session).
    public func reconfigure(hello: Hello, settings: ClientSettings) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let box = self.box
            queue.async {
                box.core?.updateSettings(settings)
                c.resume()
            }
        }
        await connection.reconfigure(hello: hello)
    }

    // MARK: Talking

    public func pressTalk() {
        connect()
        onCore { $0.pressTalk() }
    }

    public func releaseTalk() { onCore { $0.releaseTalk() } }

    public func startHandsFree() {
        connect()
        onCore { $0.startHandsFree() }
    }

    public func stopHandsFree() { onCore { $0.stopHandsFree() } }

    /// Stop the agent talking (a tap on stop): local flush, `interrupt`, `played_ms`.
    public func stopSpeaking() { onCore { $0.stopSpeaking() } }

    public func setMuted(_ muted: Bool) { onCore { $0.muted = muted } }

    public func sendText(_ text: String) {
        connect()
        onCore { $0.sendText(text) }
    }

    /// The person's answer to a question, worked out by `ApprovalQueue.answer` (PROTOCOL.md "Approvals").
    public func answer(_ answer: ApprovalQueue.Answer) {
        onCore { $0.sendControl(answer.message) }
    }

    /// Asks for another space; `switchOutcome` tells what came of it.
    public func switchSpace(_ name: String) {
        connect()
        onCore { $0.requestSwitch(SwitchRequest(.space, name)) }
    }

    /// Asks for another mode (`conversation` or `act`); `switchOutcome` tells what came of it.
    public func switchMode(_ name: String) {
        connect()
        onCore { $0.requestSwitch(SwitchRequest(.mode, name)) }
    }

    /// Fetches `/v1/status` now (a status view opened); the result arrives as a `status` event.
    public func refreshStatus() {
        guard let monitor = statusMonitor else { return }
        Task { await monitor.request(.requested) }
    }

    static func describe(_ error: any Error) -> String {
        switch error {
        case let e as StatusFetchError: return e.description
        case let e as URLError: return e.localizedDescription
        case is JSONValue.ParseError, is ServerStatus.DecodeError: return "the status is not JSON the client can read"
        default: return "\(error)"
        }
    }

    /// Starts the audio engine ahead of the first press, so the first words are not lost to its start-up time.
    public func warmUpAudio() { onCore { $0.warmUp() } }

    public func stopAudio() async {
        let audio = self.audio
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            setupQueue.async {
                audio.stop()
                c.resume()
            }
        }
        let box = self.box
        queue.async { box.core?.audioStatus(.stopped) }
    }
}
