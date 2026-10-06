import Foundation

extension ClientEvent {
    /// One flat JSON object per event, for the JSONL log the e2e tests read.
    public var json: [(String, JSONValue)] {
        switch self {
        case let .connection(state):
            var o: [(String, JSONValue)] = [("event", "connection")]
            switch state {
            case .idle: o.append(("state", "idle"))
            case let .connecting(attempt): o += [("state", "connecting"), ("attempt", .int(attempt))]
            case .handshaking: o.append(("state", "handshaking"))
            case let .ready(session): o += [("state", "ready"), ("session", .string(session))]
            case let .waiting(next, delay, after):
                o += [("state", "waiting"), ("next_attempt", .int(next)), ("delay_ms", .int(delay.milliseconds)),
                      ("code", .int(after.code)), ("reason", .string(after.reason))]
            case let .stopped(report):
                o.append(("state", "stopped"))
                if let report { o += [("code", .int(report.code)), ("reason", .string(report.reason))] }
            }
            return o
        case let .server(message):
            return [("event", "recv"), ("msg", ProtocolCodec.json(message))]
        case let .malformed(text):
            return [("event", "malformed"), ("text", .string(text))]
        case let .talk(phase):
            return [("event", "talk"), ("phase", .string(phase.rawValue))]
        case let .handsFree(on):
            return [("event", "hands_free"), ("on", .bool(on))]
        case let .playback(notice):
            switch notice {
            case let .started(reply, preroll):
                return [("event", "playback"), ("kind", "started"), ("reply", reply.map(JSONValue.string) ?? .null),
                        ("preroll_ms", .int(preroll))]
            case let .finished(reply, ms, interrupted):
                return [("event", "playback"), ("kind", interrupted ? "interrupted" : "finished"),
                        ("reply", .string(reply)), ("played_ms", .int(ms))]
            case let .underrun(reply, gap, resumed, preroll):
                return [("event", "playback"), ("kind", "underrun"), ("reply", reply.map(JSONValue.string) ?? .null),
                        ("gap_ms", .int(gap)), ("resumed", .bool(resumed)), ("preroll_ms", .int(preroll))]
            }
        case let .sent(message):
            return [("event", "sent"), ("msg", ProtocolCodec.json(message))]
        case let .utterance(u):
            return [("event", "utterance"), ("frames", .int(u.frames)), ("speech_ms", .int(u.speechMs)),
                    ("tail_ms", .int(u.tailMs)), ("peak_dbfs", .double((u.peakDBFS * 10).rounded() / 10))]
        case let .latency(l):
            return [("event", "latency"), ("reply", l.reply.map(JSONValue.string) ?? .null),
                    ("stop_to_audio_start_ms", l.stopToAudioStartMs.map(JSONValue.int) ?? .null),
                    ("stop_to_playback_ms", l.stopToPlaybackMs.map(JSONValue.int) ?? .null),
                    ("release_to_playback_ms", l.releaseToPlaybackMs.map(JSONValue.int) ?? .null)]
        case let .gate(armed):
            return [("event", "gate"), ("armed", .bool(armed))]
        case let .audioEngine(status):
            switch status {
            case .starting: return [("event", "engine"), ("status", "starting")]
            case let .running(d): return [("event", "engine"), ("status", "running"), ("detail", .string(d))]
            case .stopped: return [("event", "engine"), ("status", "stopped")]
            case let .failed(e): return [("event", "engine"), ("status", "failed"), ("detail", .string(e))]
            }
        case let .discardedAudio(bytes):
            return [("event", "discarded_audio"), ("bytes", .int(bytes))]
        case let .replyAudio(reply, bytes, chunks, maxChunk):
            return [("event", "reply_audio"), ("reply", .string(reply)), ("bytes", .int(bytes)),
                    ("chunks", .int(chunks)), ("max_chunk_bytes", .int(maxChunk))]
        case let .captureChunk(ms):
            return [("event", "capture_chunk"), ("ms", .int(ms))]
        case let .status(reason, status):
            return [("event", "status"), ("reason", .string(reason.rawValue)), ("status", status.json)]
        case let .statusFailed(reason, error):
            return [("event", "status_failed"), ("reason", .string(reason.rawValue)), ("error", .string(error))]
        case let .switchRequested(r):
            return [("event", "switch"), ("phase", "requested"), ("kind", .string(r.kind.rawValue)),
                    ("name", .string(r.name))]
        case let .switchOutcome(outcome):
            let r = outcome.request
            var o: [(String, JSONValue)] = [("event", "switch"), ("phase", .string(outcome.phase)),
                                            ("kind", .string(r.kind.rawValue)), ("name", .string(r.name))]
            switch outcome {
            case let .switched(_, info):
                o += [("space", .string(info.name)), ("mode", info.mode.map(JSONValue.string) ?? .null)]
            case let .refused(_, reason): o.append(("reason", .string(reason)))
            case .noAnswer: break
            }
            return o
        }
    }

    /// Names a script can wait for (`wait:<key>`). Each event yields its general and specific keys.
    public var tallyKeys: [String] {
        switch self {
        case let .connection(state):
            switch state {
            case .ready: return ["ready"]
            case let .stopped(r): return ["stopped"] + (r.map { ["stopped:\($0.code)"] } ?? [])
            case let .waiting(_, _, after): return ["waiting", "closed:\(after.code)"]
            case .connecting: return ["connecting"]
            case .handshaking: return ["handshaking"]
            case .idle: return []
            }
        case let .server(m):
            switch m {
            case let .state(s): return ["state", "state:\(s.wire)"]
            case let .transcript(final, _): return final ? ["transcript", "transcript:final"] : ["transcript"]
            case let .tool(e): return ["tool", "tool:\(e.phase.wire)"]
            case let .hold(h): return ["hold", "hold:\(h.phase.wire)"]
            default: return [m.type]
            }
        case .malformed: return ["malformed"]
        case let .talk(p): return ["talk:\(p.rawValue)"]
        case let .handsFree(on): return [on ? "hands-free-on" : "hands-free-off"]
        case let .playback(n):
            switch n {
            case .started: return ["playback-started"]
            case let .finished(_, _, interrupted): return [interrupted ? "playback-interrupted" : "playback-finished"]
            case .underrun: return ["underrun"]
            }
        case let .sent(m): return ["sent:\(m.type)"]
        case .utterance: return ["utterance"]
        case .latency: return ["latency"]
        case let .gate(armed): return [armed ? "gate-armed" : "gate-open"]
        case let .audioEngine(s):
            switch s {
            case .running: return ["engine-running"]
            case .failed: return ["engine-failed"]
            case .stopped: return ["engine-stopped"]
            case .starting: return ["engine-starting"]
            }
        case .discardedAudio: return ["discarded-audio"]
        case .replyAudio: return ["reply-audio"]
        case .captureChunk: return ["capture-chunk"]
        case let .status(reason, _): return ["status", "status:\(reason.rawValue)"]
        case .statusFailed: return ["status-failed"]
        case .switchRequested: return ["switch-requested"]
        case let .switchOutcome(o): return ["switch", "switch:\(o.phase)"]
        }
    }
}

extension SwitchOutcome {
    /// `switched`, `refused` or `no-answer`: the log's phase and the script's wait key (`wait:switch:refused`).
    public var phase: String {
        switch self {
        case .switched: return "switched"
        case .refused: return "refused"
        case .noAnswer: return "no-answer"
        }
    }
}

/// JSONL: one object per line with `t`, seconds since the log opened (monotonic).
public final class EventLog: @unchecked Sendable {
    // Invariant for @unchecked Sendable: the handle is only written under `lock`.
    private let lock = NSLock()
    private let handle: FileHandle?
    private let echo: Bool
    public let startedAt: Nanos

    /// `url` nil writes nothing (unless `echo`); `echo` also prints each line to stdout.
    public init(url: URL?, echo: Bool = false) throws {
        if let url {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            handle.seekToEndOfFile()
            self.handle = handle
        } else {
            self.handle = nil
        }
        self.echo = echo
        self.startedAt = MonotonicClock.now()
    }

    public func seconds(at now: Nanos = MonotonicClock.now()) -> Double {
        (Double(now &- startedAt) / 1e9 * 10_000).rounded() / 10_000
    }

    public func write(_ fields: [(String, JSONValue)], at now: Nanos = MonotonicClock.now()) {
        let line = JSONValue.object([("t", .double(seconds(at: now)))] + fields).serialized() + "\n"
        lock.lock()
        defer { lock.unlock() }
        handle?.write(Data(line.utf8))
        if echo { FileHandle.standardOutput.write(Data(line.utf8)) }
    }

    public func record(_ event: ClientEvent) { write(event.json) }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        try? handle?.close()
    }
}
