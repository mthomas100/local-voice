// lvclient: a protocol v1 client on the command line, built from the same LocalVoiceKit core as the iOS and macOS
// apps. It runs a scripted session (a synthetic microphone speaks planted tones or a `say` recording; replies play on
// a headless engine or the speaker), writes every event to JSONL, and prints a one-line JSON summary.
//
//   lvclient --url ws://127.0.0.1:18770/v1/voice --mic ptt \
//            --script "press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms" \
//            --log runs/x/client.jsonl
//
// Exit status: 0 when the script finished, 1 when a step failed or timed out, 2 for bad arguments.

import Foundation
import LocalVoiceKit
import Synchronization

struct Options {
    var url = URL(string: "ws://127.0.0.1:\(ProtocolV1.defaultPort)\(ProtocolV1.path)")!
    var device = "lvclient"
    var client = ClientKind.test
    var mic = MicMode.ptt
    var audio = "headless"
    var script = "press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms"
    var log: URL?
    var record: URL?
    var timeout = 60.0
    var prerollMs = 100
    var gateMs = 600
    var gateMode = MicGate.Mode.hard
    var chunkFrames = 960
    var echo = false
    var settleMs = 400
    var status = true
}

func usage(_ message: String? = nil) -> Never {
    if let message { FileHandle.standardError.write(Data("lvclient: \(message)\n".utf8)) }
    FileHandle.standardError.write(Data("""
        usage: lvclient [--url ws://host:port/v1/voice] [--device NAME] [--client test|mac|iphone] [--mic ptt|vad]
                        [--audio headless|device|device-muted|microphone] [--script STEPS] [--log FILE.jsonl]
                        [--record-output FILE.wav] [--timeout SECONDS] [--preroll-ms N] [--gate-ms N]
                        [--gate-mode hard|twoTier] [--chunk-frames N] [--settle-ms N] [--echo] [--no-status]
        --audio microphone uses the real microphone with voice processing (asks for permission once).
        --no-status fetches no /v1/status (by default it is fetched on connect, after each turn and after a switch).

        """.utf8))
    exit(2)
}

func parse() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    func value(_ flag: String) -> String {
        guard !args.isEmpty else { usage("\(flag) needs a value") }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--url": o.url = URL(string: value(flag)) ?? { usage("bad url") }()
        case "--device": o.device = value(flag)
        case "--client": o.client = ClientKind(rawValue: value(flag)) ?? { usage("bad --client") }()
        case "--mic": o.mic = MicMode(rawValue: value(flag)) ?? { usage("bad --mic") }()
        case "--audio": o.audio = value(flag)
        case "--script": o.script = value(flag)
        case "--log": o.log = URL(fileURLWithPath: value(flag))
        case "--record-output": o.record = URL(fileURLWithPath: value(flag))
        case "--timeout": o.timeout = Double(value(flag)) ?? { usage("bad --timeout") }()
        case "--preroll-ms": o.prerollMs = Int(value(flag)) ?? { usage("bad --preroll-ms") }()
        case "--gate-ms": o.gateMs = Int(value(flag)) ?? { usage("bad --gate-ms") }()
        case "--gate-mode": o.gateMode = MicGate.Mode(rawValue: value(flag)) ?? { usage("bad --gate-mode") }()
        case "--chunk-frames": o.chunkFrames = Int(value(flag)) ?? { usage("bad --chunk-frames") }()
        case "--settle-ms": o.settleMs = Int(value(flag)) ?? { usage("bad --settle-ms") }()
        case "--echo": o.echo = true
        case "--no-status": o.status = false
        case "-h", "--help": usage()
        default: usage("unknown option \(flag)")
        }
    }
    return o
}

/// The questions waiting, kept as the apps keep them, so the log shows the card an app would draw for each request
/// (`approval` lines) and the script answers the card on screen.
final class Approvals: Sendable {
    private let queue = Mutex(ApprovalQueue())

    func receive(_ request: ConfirmRequest) -> (ApprovalQueue.Received, PendingApproval) {
        queue.withLock { q in
            let received = q.receive(request, at: Date())
            return (received, q.waiting.first { $0.id == request.id }!)
        }
    }

    func answerCurrent(choice: String) -> ApprovalQueue.Answer? {
        queue.withLock { q in q.current.flatMap { q.answer(id: $0.id, choice: choice) } }
    }

    func cancel(_ id: ScalarID) -> PendingApproval? { queue.withLock { $0.cancel(id: id) } }

    func closeAll() -> [PendingApproval] { queue.withLock { $0.closeAll() } }
}

final class CLITarget: ScriptTarget {
    let client: VoiceClient
    let microphone: SyntheticMicrophone?
    let approvals: Approvals
    let log: EventLog
    let tally: EventTally

    init(client: VoiceClient, microphone: SyntheticMicrophone?, approvals: Approvals, log: EventLog, tally: EventTally) {
        self.client = client
        self.microphone = microphone
        self.approvals = approvals
        self.log = log
        self.tally = tally
    }

    func scriptConnect() async { client.connect() }
    func scriptDisconnect() async { await client.disconnect() }
    func scriptPressTalk() async { client.pressTalk() }
    func scriptReleaseTalk() async { client.releaseTalk() }
    func scriptHandsFree(_ on: Bool) async { on ? client.startHandsFree() : client.stopHandsFree() }
    func scriptStopSpeaking() async { client.stopSpeaking() }
    func scriptText(_ text: String) async { client.sendText(text) }
    func scriptAnswer(choice: String) async -> Bool {
        guard let answer = approvals.answerCurrent(choice: choice) else { return false }
        client.answer(answer)
        log.write(answer.approval.logFields("answered", [("choice", .string(answer.choice.id)),
                                                         ("label", .string(answer.choice.label)),
                                                         ("confirmed", .bool(answer.choice.allows))]))
        tally.note(key: "approval-answered")
        return true
    }
    func scriptSwitch(_ request: SwitchRequest) async {
        request.kind == .space ? client.switchSpace(request.name) : client.switchMode(request.name)
    }
    func scriptRefreshStatus() async { client.refreshStatus() }
}

let options = parse()
let steps: [ScriptStep]
do {
    steps = try ScriptStep.parse(options.script)
} catch {
    usage("\(error)")
}

let microphone: SyntheticMicrophone? = options.audio == "microphone" ? nil
    : SyntheticMicrophone(chunkFrames: options.chunkFrames)
let headless: HeadlessAudioIO?
let audio: any AudioIO
switch options.audio {
case "headless":
    let io = HeadlessAudioIO(microphone: microphone, recordOutputTo: options.record)
    headless = io
    audio = io
case "device":
    headless = nil
    audio = LiveAudioIO(options: .init(capture: .synthetic(microphone!)))
case "device-muted":
    headless = nil
    audio = LiveAudioIO(options: .init(capture: .synthetic(microphone!), outputVolume: 0))
case "microphone":
    headless = nil
    audio = LiveAudioIO(options: .init(capture: .microphone(voiceProcessing: true)))
default:
    usage("bad --audio \(options.audio)")
}

let settings = ClientSettings(mic: options.mic, gate: MicGate(durationMs: options.gateMs, mode: options.gateMode),
                              prerollMs: options.prerollMs, engineIdleStopSeconds: nil)
let client = VoiceClient(
    configuration: .init(url: options.url, hello: Hello(client: options.client, device: options.device, mic: options.mic),
                         settings: settings, fetchStatus: options.status),
    audio: audio)
let log: EventLog
do {
    log = try EventLog(url: options.log, echo: options.echo)
} catch {
    usage("cannot open log: \(error)")
}
let tally = EventTally()
let approvals = Approvals()

/// Times at which playback was cut (a server interrupt arriving, or the client's own barge-in), for the flush check.
let cuts = Mutex<[(at: Nanos, why: String)]>([])
let playedMs = Mutex<[String: Int]>([:])
let latencies = Mutex<[TurnLatency]>([])

let consumer = Task {
    for await event in client.events {
        log.record(event)
        tally.note(event)
        switch event {
        case .server(.interrupt): cuts.withLock { $0.append((MonotonicClock.now(), "server interrupt")) }
        case .sent(.interrupt): cuts.withLock { $0.append((MonotonicClock.now(), "client interrupt")) }
        case let .sent(.playedMs(reply, ms)): playedMs.withLock { $0[reply] = ms }
        case let .latency(l): latencies.withLock { $0.append(l) }
        case let .server(.confirmRequest(request)):
            let (received, approval) = approvals.receive(request)
            log.write(approval.logFields("shown", [("received", .string(received.rawValue))]))
            tally.note(key: "approval-shown")
        case let .server(.confirmCancel(id, why)):
            if let gone = approvals.cancel(id) {
                log.write(gone.logFields("closed", [("why", .string(why)), ("by", "server")]))
                tally.note(key: "approval-closed")
            }
        case .connection(.waiting), .connection(.stopped):
            for gone in approvals.closeAll() {
                log.write(gone.logFields("closed", [("why", "connection lost"), ("by", "client")]))
                tally.note(key: "approval-closed")
            }
        default: break
        }
    }
}

log.write([("event", "start"), ("url", .string(options.url.absoluteString)), ("device", .string(options.device)),
           ("mic", .string(options.mic.rawValue)), ("audio", .string(options.audio)),
           ("script", .string(options.script))])

let runner = ScriptRunner(target: CLITarget(client: client, microphone: microphone, approvals: approvals, log: log,
                                            tally: tally),
                          tally: tally, log: log)
var failure: String?
do {
    try await withThrowingTaskGroup(of: Void.self) { group in
        group.addTask { try await runner.run(steps) }
        group.addTask {
            try await Task.sleep(for: .seconds(options.timeout))
            throw ScriptFailure(step: "(whole script)", reason: "timed out after \(options.timeout) s")
        }
        try await group.next()
        group.cancelAll()
    }
} catch {
    failure = "\(error)"
}

// Let the last events (played_ms, end of playback) land before the summary.
try? await Task.sleep(for: .milliseconds(options.settleMs))
await client.shutdown()
_ = await consumer.value
headless?.monitor.close()

var summary: [(String, JSONValue)] = [
    ("event", "summary"), ("ok", .bool(failure == nil)),
    ("played_ms", .object(playedMs.withLock { $0 }.sorted { $0.key < $1.key }.map { ($0.key, .int($0.value)) })),
    ("latency", .array(latencies.withLock { $0 }.map { l in
        .object([("reply", l.reply.map(JSONValue.string) ?? .null),
                 ("stop_to_audio_start_ms", l.stopToAudioStartMs.map(JSONValue.int) ?? .null),
                 ("stop_to_playback_ms", l.stopToPlaybackMs.map(JSONValue.int) ?? .null),
                 ("release_to_playback_ms", l.releaseToPlaybackMs.map(JSONValue.int) ?? .null)])
    })),
]
if let failure { summary.append(("failure", .string(failure))) }
if let monitor = headless?.monitor {
    let intervals = monitor.audible
    summary.append(("audible_s", .double((monitor.audibleSeconds * 1000).rounded() / 1000)))
    // For each cut: how long the output kept sounding afterwards (PROTOCOL.md: flush locally before anything else).
    let tails = cuts.withLock { $0 }.compactMap { cut -> JSONValue? in
        guard let interval = intervals.first(where: { $0.start <= cut.at && cut.at <= $0.end + .ms(15) }) else {
            return nil
        }
        let ms = (Double(Int64(interval.end) - Int64(cut.at)) / 1e6).rounded()
        return .object([("why", .string(cut.why)), ("tail_ms", .int(Int(ms)))])
    }
    summary.append(("flush_tails", .array(tails)))
    // When the output was audible, in the log's seconds: against a real server this shows when a reply went quiet
    // relative to the user's speech (the server may stop sending before its `interrupt`, orchestrator bargein.py).
    summary.append(("audible", .array(intervals.map { i in
        .array([.double(log.seconds(at: i.start)), .double(log.seconds(at: i.end))])
    })))
}
log.write(summary)
log.close()
print(JSONValue.object(summary).serialized())
exit(failure == nil ? 0 : 1)
