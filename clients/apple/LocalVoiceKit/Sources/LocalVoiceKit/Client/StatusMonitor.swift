import Foundation

/// Fetches `/v1/status` when something happened that it would show, never on a timer: the connection came up, a turn
/// ended (the server's latency breakdown is new), the space changed (its model may differ), a switch went unanswered
/// (to show where the server really is), or a status view opened. Everything else the dashboard shows (state, hold,
/// tool, space and mode) the server pushes as messages anyway.
///
/// One fetch at a time: requests within `debounce` share a fetch (on connect the orchestrator sends `welcome` and
/// `space` back to back), and requests while a fetch runs make exactly one more fetch after it.
public actor StatusMonitor {
    public enum Reason: String, Sendable {
        case connected
        case turnEnded = "turn-ended"
        case spaceChanged = "space-changed"
        case switchUnanswered = "switch-unanswered"
        case requested
    }

    private let fetcher: any StatusFetching
    private let debounce: Duration
    private let deliver: @Sendable (Reason, Result<ServerStatus, any Error>) -> Void
    private var scheduled: Reason?
    private var running = false
    private var again: Reason?
    public private(set) var fetches = 0

    public init(fetcher: any StatusFetching, debounce: Duration = .milliseconds(300),
                deliver: @escaping @Sendable (Reason, Result<ServerStatus, any Error>) -> Void) {
        self.fetcher = fetcher
        self.debounce = debounce
        self.deliver = deliver
    }

    /// Why an event calls for a fresh status, if it does.
    public static func reason(for event: ClientEvent) -> Reason? {
        switch event {
        case .connection(.ready): return .connected
        case .server(.endOfTurn): return .turnEnded
        case .server(.space): return .spaceChanged
        case .switchOutcome(.noAnswer): return .switchUnanswered
        default: return nil
        }
    }

    public func request(_ reason: Reason) {
        if running {
            again = again ?? reason
            return
        }
        guard scheduled == nil else { return }
        scheduled = reason
        Task { await self.fetchAfterDebounce() }
    }

    private func fetchAfterDebounce() async {
        try? await Task.sleep(for: debounce)
        guard let reason = scheduled else { return }
        scheduled = nil
        running = true
        fetches += 1
        let result: Result<ServerStatus, any Error>
        do {
            result = .success(try await fetcher.fetchStatus())
        } catch {
            result = .failure(error)
        }
        running = false
        deliver(reason, result)
        if let next = again {
            again = nil
            request(next)
        }
    }
}
