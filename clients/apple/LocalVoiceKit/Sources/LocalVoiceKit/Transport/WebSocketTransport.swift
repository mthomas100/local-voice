import Foundation
import Synchronization

public enum WireMessage: Sendable, Equatable {
    case text(String)
    case binary(Data)
}

/// The socket closed. `code` is the peer's close code, or 1006 when no close frame arrived.
public struct TransportClosed: Error, Sendable, Equatable, CustomStringConvertible {
    public let code: Int
    public let reason: String

    public init(code: Int, reason: String) {
        self.code = code
        self.reason = reason
    }

    public var description: String { reason.isEmpty ? CloseCode.describe(code) : "\(CloseCode.describe(code)): \(reason)" }
}

/// One open WebSocket. A seam so the connection logic can be tested without a network.
public protocol WebSocketTransport: AnyObject, Sendable {
    func send(_ message: WireMessage) async throws
    /// The next message; throws `TransportClosed` once the socket is closed.
    func receive() async throws -> WireMessage
    func close(code: Int, reason: String)
}

public protocol WebSocketConnecting: Sendable {
    func connect(to url: URL, timeout: Duration) async throws -> any WebSocketTransport
}

public struct URLSessionConnector: WebSocketConnecting {
    public init() {}

    public func connect(to url: URL, timeout: Duration) async throws -> any WebSocketTransport {
        try await URLSessionWebSocketTransport.open(url: url, timeout: timeout)
    }
}

public struct ConnectTimeout: Error, CustomStringConvertible {
    public var description: String { "connect timed out" }
}

/// `URLSessionWebSocketTask`, with the close code captured from the delegate.
///
/// When the server closes (4403, 4409, 1002), `receive()` fails with a socket error and the code arrives separately on
/// the delegate, before or after that error. `receive()` therefore waits up to 250 ms for the delegate before deciding
/// the close was abnormal (1006). Binary messages stay at 640-1,920 bytes, under the reported ~3,000-byte failure of
/// this API (research notes).
final class URLSessionWebSocketTransport: NSObject, WebSocketTransport, URLSessionWebSocketDelegate, @unchecked Sendable {
    // Invariant for @unchecked Sendable: `session` and `task` are written once in `open` before the task is resumed and
    // only read afterwards; all other mutable state lives in `state`.
    private var session: URLSession!
    private var task: URLSessionWebSocketTask!

    private struct State {
        var openContinuation: CheckedContinuation<Void, any Error>?
        var closed: TransportClosed?
        var closeWaiters: [UUID: CheckedContinuation<Void, Never>] = [:]
    }

    private let state = Mutex(State())

    static func open(url: URL, timeout: Duration) async throws -> URLSessionWebSocketTransport {
        let transport = URLSessionWebSocketTransport()
        let configuration = URLSessionConfiguration.ephemeral
        // URLSession's own idle limit must sit above the protocol's 60 s so the connection's keepalive decides.
        configuration.timeoutIntervalForRequest = 90
        configuration.waitsForConnectivity = false
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.name = "lv.websocket.delegate"
        transport.session = URLSession(configuration: configuration, delegate: transport, delegateQueue: queue)
        var request = URLRequest(url: url)
        request.timeoutInterval = Double(timeout.nanos) / 1e9
        transport.task = transport.session.webSocketTask(with: request)
        transport.task.maximumMessageSize = 1 << 20

        let timer = Task {
            try await Task.sleep(for: timeout)
            transport.failOpen(ConnectTimeout())
        }
        defer { timer.cancel() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                transport.state.withLock { $0.openContinuation = continuation }
                transport.task.resume()
            }
        } onCancel: {
            transport.failOpen(CancellationError())
        }
        return transport
    }

    private func failOpen(_ error: any Error) {
        let continuation = state.withLock { s -> CheckedContinuation<Void, any Error>? in
            defer { s.openContinuation = nil }
            return s.openContinuation
        }
        guard let continuation else { return }
        task.cancel()
        session.invalidateAndCancel()
        continuation.resume(throwing: error)
    }

    func send(_ message: WireMessage) async throws {
        do {
            switch message {
            case let .text(s): try await task.send(.string(s))
            case let .binary(d): try await task.send(.data(d))
            }
        } catch {
            throw await closeReport(fallback: error)
        }
    }

    func receive() async throws -> WireMessage {
        do {
            switch try await task.receive() {
            case let .string(s): return .text(s)
            case let .data(d): return .binary(d)
            @unknown default: return .text("")
            }
        } catch {
            throw await closeReport(fallback: error)
        }
    }

    func close(code: Int, reason: String) {
        let closeCode = URLSessionWebSocketTask.CloseCode(rawValue: code) ?? .normalClosure
        task.cancel(with: closeCode, reason: reason.data(using: .utf8))
        session.finishTasksAndInvalidate()
    }

    /// The close code from the delegate, waiting briefly because it can arrive after the receive error.
    private func closeReport(fallback error: any Error) async -> TransportClosed {
        if let closed = state.withLock({ $0.closed }) { return closed }
        let id = UUID()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let alreadyClosed = state.withLock { s -> Bool in
                if s.closed != nil { return true }
                s.closeWaiters[id] = continuation
                return false
            }
            if alreadyClosed {
                continuation.resume()
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(250)) { [weak self] in
                let waiter = self?.state.withLock { $0.closeWaiters.removeValue(forKey: id) }
                waiter?.resume()
            }
        }
        if let closed = state.withLock({ $0.closed }) { return closed }
        let code = task.closeCode.rawValue
        if code != 0 { return TransportClosed(code: code, reason: "") }
        return TransportClosed(code: CloseCode.abnormal, reason: (error as NSError).localizedDescription)
    }

    private func markClosed(_ closed: TransportClosed) {
        let waiters = state.withLock { s -> [CheckedContinuation<Void, Never>] in
            if s.closed == nil { s.closed = closed }
            defer { s.closeWaiters.removeAll() }
            return Array(s.closeWaiters.values)
        }
        waiters.forEach { $0.resume() }
    }

    // MARK: URLSessionWebSocketDelegate

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        let continuation = state.withLock { s -> CheckedContinuation<Void, any Error>? in
            defer { s.openContinuation = nil }
            return s.openContinuation
        }
        continuation?.resume()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        markClosed(TransportClosed(code: closeCode.rawValue,
                                   reason: reason.flatMap { String(data: $0, encoding: .utf8) } ?? ""))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let openContinuation = state.withLock { s -> CheckedContinuation<Void, any Error>? in
            defer { s.openContinuation = nil }
            return s.openContinuation
        }
        if let openContinuation {
            // The handshake failed: refused, timed out, blocked by App Transport Security, or an HTTP error (a server
            // that closes before accepting surfaces as HTTP 403 here).
            openContinuation.resume(throwing: error ?? TransportClosed(code: CloseCode.abnormal, reason: "handshake failed"))
        }
        let code = self.task.closeCode.rawValue
        markClosed(TransportClosed(code: code != 0 ? code : CloseCode.abnormal,
                                   reason: error.map { ($0 as NSError).localizedDescription } ?? ""))
        session.finishTasksAndInvalidate()
    }
}
