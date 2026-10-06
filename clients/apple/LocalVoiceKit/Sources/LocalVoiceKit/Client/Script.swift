import Foundation
import Synchronization

/// Counts events by key, so a script can wait for "the next end_of_turn" without racing the event stream.
public final class EventTally: Sendable {
    private let counts = Mutex<[String: Int]>([:])

    public init() {}

    public func note(_ event: ClientEvent) {
        counts.withLock { counts in
            for key in event.tallyKeys { counts[key, default: 0] += 1 }
        }
    }

    public func note(key: String) { counts.withLock { $0[key, default: 0] += 1 } }

    public func count(_ key: String) -> Int { counts.withLock { $0[key, default: 0] } }
}

/// What a script drives: the CLI's client, or an app's session model in its unattended test mode.
public protocol ScriptTarget: AnyObject, Sendable {
    var microphone: SyntheticMicrophone? { get }
    func scriptConnect() async
    func scriptDisconnect() async
    func scriptPressTalk() async
    func scriptReleaseTalk() async
    func scriptHandsFree(_ on: Bool) async
    func scriptStopSpeaking() async
    func scriptText(_ text: String) async
    /// Picks `choice` on the card on screen, as a tap would; false when no question waits or it has no such choice.
    func scriptAnswer(choice: String) async -> Bool
    func scriptSwitch(_ request: SwitchRequest) async
    func scriptRefreshStatus() async
    /// Evidence of what is on screen now (the Mac app draws its panel to a PNG); nothing where there is no screen.
    func scriptSnapshot(_ label: String) async
}

extension ScriptTarget {
    public func scriptSnapshot(_ label: String) async {}
}

/// One step of a session script. Steps are separated by `;` or newlines:
///
///     press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms
///
/// `say:` and `say-wait:` take `tone:<hz>:<seconds>`, `silence:<seconds>` or `file:<path>` (any audio file;
/// a `say -o` recording is the speech fixture). `wait:<key>[:<ms>]` consumes one occurrence of an event key
/// (see `ClientEvent.tallyKeys`): it returns at once if one happened since the last wait for that key.
/// `next:<key>[:<ms>]` waits for an occurrence after the step starts, whatever happened before, and consumes
/// everything up to it: for a real server, where how many replies or states come before is not known in advance.
/// `space:<name>` and `mode:<name>` ask the server to switch (wait for `switch:switched`, `switch:refused` or
/// `switch:no-answer`); `status` fetches `/v1/status` (wait for `status`).
/// `confirm:yes` and `confirm:no` pick "Do it" (`allow_once`) and "Don't" (`deny`) on the approval card on screen,
/// `choose:<id>` any choice it offers (`choose:allow_session`); wait for `approval-shown` and `approval-closed`.
/// `snapshot:<label>` asks the app for a picture of its screen.
public enum ScriptStep: Sendable, Equatable, CustomStringConvertible {
    case connect, disconnect, press, release, handsFreeOn, handsFreeOff, stopSpeaking
    case say(clip: String, wait: Bool)
    case sleep(ms: Int)
    case wait(key: String, timeoutMs: Int?)
    case next(key: String, timeoutMs: Int?)
    case text(String)
    case confirm(Bool)
    case choose(String)
    case snapshot(String)
    case mark(String)
    case switchTo(SwitchRequest)
    case status

    public var description: String {
        switch self {
        case .connect: return "connect"
        case .disconnect: return "disconnect"
        case .press: return "press"
        case .release: return "release"
        case .handsFreeOn: return "handsfree-on"
        case .handsFreeOff: return "handsfree-off"
        case .stopSpeaking: return "stop-speaking"
        case let .say(clip, wait): return (wait ? "say-wait:" : "say:") + clip
        case let .sleep(ms): return "sleep:\(ms)"
        case let .wait(key, timeout): return "wait:\(key)" + (timeout.map { ":\($0)" } ?? "")
        case let .next(key, timeout): return "next:\(key)" + (timeout.map { ":\($0)" } ?? "")
        case let .text(t): return "text:\(t)"
        case let .confirm(yes): return "confirm:\(yes ? "yes" : "no")"
        case let .choose(id): return "choose:\(id)"
        case let .snapshot(label): return "snapshot:\(label)"
        case let .mark(m): return "mark:\(m)"
        case let .switchTo(r): return "\(r.kind.rawValue):\(r.name)"
        case .status: return "status"
        }
    }

    public struct ParseError: Error, CustomStringConvertible {
        public let step: String
        public var description: String { "cannot parse script step \"\(step)\"" }
    }

    public static func parse(_ script: String) throws -> [ScriptStep] {
        try script.split(whereSeparator: { $0 == ";" || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map(parseOne)
    }

    static func parseOne(_ s: String) throws -> ScriptStep {
        let (head, rest) = s.firstIndex(of: ":").map { (String(s[..<$0]), String(s[s.index(after: $0)...])) }
            ?? (s, "")
        switch head {
        case "connect": return .connect
        case "disconnect": return .disconnect
        case "press": return .press
        case "release": return .release
        case "handsfree-on": return .handsFreeOn
        case "handsfree-off": return .handsFreeOff
        case "stop-speaking": return .stopSpeaking
        case "say": return .say(clip: rest, wait: false)
        case "say-wait": return .say(clip: rest, wait: true)
        case "sleep":
            guard let ms = Int(rest) else { throw ParseError(step: s) }
            return .sleep(ms: ms)
        case "wait", "next":
            // A key may itself contain a colon (sent:played_ms); a trailing all-digit part is the timeout.
            func make(_ key: String, _ ms: Int?) -> ScriptStep {
                head == "wait" ? .wait(key: key, timeoutMs: ms) : .next(key: key, timeoutMs: ms)
            }
            let parts = rest.split(separator: ":").map(String.init)
            if parts.count >= 2, let ms = Int(parts.last!) {
                return make(parts.dropLast().joined(separator: ":"), ms)
            }
            guard !rest.isEmpty else { throw ParseError(step: s) }
            return make(rest, nil)
        case "text": return .text(rest)
        case "confirm":
            guard rest == "yes" || rest == "no" else { throw ParseError(step: s) }
            return .confirm(rest == "yes")
        case "choose":
            guard !rest.isEmpty else { throw ParseError(step: s) }
            return .choose(rest)
        case "snapshot":
            guard !rest.isEmpty else { throw ParseError(step: s) }
            return .snapshot(rest)
        case "mark": return .mark(rest)
        case "space", "mode":
            guard !rest.isEmpty else { throw ParseError(step: s) }
            return .switchTo(SwitchRequest(head == "space" ? .space : .mode, rest))
        case "status": return .status
        default: throw ParseError(step: s)
        }
    }

    /// The samples of a clip spec, at 48 kHz.
    public static func clip(_ spec: String) throws -> [Float] {
        let parts = spec.split(separator: ":", maxSplits: 1).map(String.init)
        switch parts.first {
        case "tone":
            let args = (parts.count > 1 ? parts[1] : "").split(separator: ":").compactMap { Double($0) }
            guard args.count == 2 else { throw ParseError(step: spec) }
            return SyntheticMicrophone.tone(frequency: args[0], seconds: args[1])
        case "silence":
            guard parts.count > 1, let seconds = Double(parts[1]) else { throw ParseError(step: spec) }
            return SyntheticMicrophone.silence(seconds: seconds)
        case "file":
            guard parts.count > 1 else { throw ParseError(step: spec) }
            return try SyntheticMicrophone.load(URL(fileURLWithPath: parts[1]))
        default:
            throw ParseError(step: spec)
        }
    }
}

public struct ScriptFailure: Error, CustomStringConvertible {
    public let step: String
    public let reason: String

    public init(step: String, reason: String) {
        self.step = step
        self.reason = reason
    }
    public var description: String { "script step \"\(step)\" failed: \(reason)" }
}

/// Runs a script against a target, waiting on the tally.
public final class ScriptRunner: Sendable {
    private let target: any ScriptTarget
    private let tally: EventTally
    private let log: EventLog?
    private let consumed = Mutex<[String: Int]>([:])

    public init(target: any ScriptTarget, tally: EventTally, log: EventLog?) {
        self.target = target
        self.tally = tally
        self.log = log
        // Logged with its time: the moment the clip's last sample left the microphone, from which a test against a
        // real server finds the end of speech (the clip's own trailing silence subtracted).
        target.microphone?.onClipFinished = { [tally, log] name in
            log?.write([("event", "clip"), ("phase", "finished"), ("name", .string(name))])
            tally.note(key: "clip-finished")
        }
    }

    public func run(_ steps: [ScriptStep], waitTimeoutMs: Int = 15_000) async throws {
        for step in steps {
            log?.write([("event", "step"), ("step", .string(step.description))])
            switch step {
            case .connect: await target.scriptConnect()
            case .disconnect: await target.scriptDisconnect()
            case .press: await target.scriptPressTalk()
            case .release: await target.scriptReleaseTalk()
            case .handsFreeOn: await target.scriptHandsFree(true)
            case .handsFreeOff: await target.scriptHandsFree(false)
            case .stopSpeaking: await target.scriptStopSpeaking()
            case let .text(t): await target.scriptText(t)
            case let .switchTo(r): await target.scriptSwitch(r)
            case .status: await target.scriptRefreshStatus()
            case let .confirm(yes):
                try await answer(yes ? ApprovalChoice.allowOnce : ApprovalChoice.deny, step: step)
            case let .choose(id):
                try await answer(id, step: step)
            case let .snapshot(label):
                await target.scriptSnapshot(label)
            case let .sleep(ms):
                try await Task.sleep(for: .milliseconds(ms))
            case .mark:
                break
            case let .say(spec, wait):
                guard let mic = target.microphone else {
                    throw ScriptFailure(step: step.description, reason: "no synthetic microphone")
                }
                let samples: [Float]
                do {
                    samples = try ScriptStep.clip(spec)
                } catch {
                    throw ScriptFailure(step: step.description, reason: "\(error)")
                }
                mic.play(samples, name: spec)
                if wait {
                    try await self.wait(for: "clip-finished", timeoutMs: Int(Double(samples.count) / 48) + 5_000,
                                        step: step)
                }
            case let .wait(key, timeout):
                try await self.wait(for: key, timeoutMs: timeout ?? waitTimeoutMs, step: step)
            case let .next(key, timeout):
                let needed = tally.count(key) + 1
                try await poll(key, until: needed, timeoutMs: timeout ?? waitTimeoutMs, step: step)
                let seen = tally.count(key)
                consumed.withLock { $0[key] = max($0[key, default: 0], seen) }
            }
        }
    }

    private func answer(_ choice: String, step: ScriptStep) async throws {
        guard await target.scriptAnswer(choice: choice) else {
            throw ScriptFailure(step: step.description, reason: "no question waiting, or no \(choice) on its card")
        }
    }

    private func wait(for key: String, timeoutMs: Int, step: ScriptStep) async throws {
        let needed = consumed.withLock { c -> Int in
            c[key, default: 0] += 1
            return c[key]!
        }
        try await poll(key, until: needed, timeoutMs: timeoutMs, step: step)
    }

    private func poll(_ key: String, until needed: Int, timeoutMs: Int, step: ScriptStep) async throws {
        let deadline = MonotonicClock.now() + .ms(timeoutMs)
        while tally.count(key) < needed {
            if MonotonicClock.now() > deadline {
                throw ScriptFailure(step: step.description, reason: "no \(key) within \(timeoutMs) ms")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}
