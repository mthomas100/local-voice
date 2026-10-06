import Foundation
import Synchronization
import Testing
@testable import LocalVoiceKit

/// A script target that only records what it was asked to do.
final class RecordingTarget: ScriptTarget, @unchecked Sendable {
    let microphone: SyntheticMicrophone? = nil
    let calls = Mutex<[String]>([])
    private func note(_ s: String) { calls.withLock { $0.append(s) } }
    func scriptConnect() async { note("connect") }
    func scriptDisconnect() async { note("disconnect") }
    func scriptPressTalk() async { note("press") }
    func scriptReleaseTalk() async { note("release") }
    func scriptHandsFree(_ on: Bool) async { note(on ? "handsfree-on" : "handsfree-off") }
    func scriptStopSpeaking() async { note("stop-speaking") }
    func scriptText(_ text: String) async { note("text:\(text)") }
    /// Whether a question is waiting with the asked choice on its card.
    let answerable: Bool
    init(answerable: Bool = true) { self.answerable = answerable }
    func scriptAnswer(choice: String) async -> Bool {
        note("answer:\(choice)")
        return answerable
    }
    func scriptSnapshot(_ label: String) async { note("snapshot:\(label)") }
    func scriptSwitch(_ request: SwitchRequest) async { note("\(request.kind.rawValue):\(request.name)") }
    func scriptRefreshStatus() async { note("status") }
}

@Suite("Session scripts")
struct ScriptTests {
    @Test("steps parse, keys may hold colons, a trailing number is the timeout")
    func parsing() throws {
        let steps = try ScriptStep.parse("""
            press; say-wait:file:/tmp/a b.aiff; release
            wait:sent:played_ms:30000; next:end_of_turn; next:playback-finished:500
            # a comment
            text:What is it: a test?; sleep:20
            space:atlas; mode:act; status
            """)
        #expect(steps == [
            .press, .say(clip: "file:/tmp/a b.aiff", wait: true), .release,
            .wait(key: "sent:played_ms", timeoutMs: 30000), .next(key: "end_of_turn", timeoutMs: nil),
            .next(key: "playback-finished", timeoutMs: 500),
            .text("What is it: a test?"), .sleep(ms: 20),
            .switchTo(SwitchRequest(.space, "atlas")), .switchTo(SwitchRequest(.mode, "act")), .status,
        ])
        #expect(steps.map(\.description).joined(separator: "; ").contains("next:playback-finished:500"))
        #expect(throws: ScriptStep.ParseError.self) { try ScriptStep.parse("next:") }
        #expect(throws: ScriptStep.ParseError.self) { try ScriptStep.parse("jump") }
        #expect(throws: ScriptStep.ParseError.self) { try ScriptStep.parse("space:") }
    }

    @Test("wait: consumes earlier occurrences; next: waits for one after the step starts")
    func waitAndNext() async throws {
        let tally = EventTally()
        let runner = ScriptRunner(target: RecordingTarget(), tally: tally, log: nil)
        tally.note(key: "end_of_turn")
        // An occurrence before the step: wait returns at once.
        try await runner.run([.wait(key: "end_of_turn", timeoutMs: 200)])
        tally.note(key: "end_of_turn")
        // next ignores the one that already happened and times out without a new one.
        await #expect(throws: ScriptFailure.self) { try await runner.run([.next(key: "end_of_turn", timeoutMs: 100)]) }
        // A new occurrence while next waits ends the wait.
        let late = Task {
            try await Task.sleep(for: .milliseconds(50))
            tally.note(key: "end_of_turn")
        }
        try await runner.run([.next(key: "end_of_turn", timeoutMs: 2000)])
        try await late.value
        // next consumed everything up to it, so wait needs yet another occurrence.
        await #expect(throws: ScriptFailure.self) { try await runner.run([.wait(key: "end_of_turn", timeoutMs: 100)]) }
    }

    @Test("approval steps: confirm picks Do it or Don't, choose any id, and an answer that cannot be sent fails")
    func approvalSteps() async throws {
        let steps = try ScriptStep.parse("confirm:yes; confirm:no; choose:allow_session; snapshot:card")
        #expect(steps == [.confirm(true), .confirm(false), .choose("allow_session"), .snapshot("card")])
        #expect(steps.map(\.description) == ["confirm:yes", "confirm:no", "choose:allow_session", "snapshot:card"])
        #expect(throws: ScriptStep.ParseError.self) { try ScriptStep.parse("choose:") }
        #expect(throws: ScriptStep.ParseError.self) { try ScriptStep.parse("confirm:maybe") }

        let target = RecordingTarget()
        try await ScriptRunner(target: target, tally: EventTally(), log: nil).run(steps)
        #expect(target.calls.withLock { $0 } == ["answer:allow_once", "answer:deny", "answer:allow_session",
                                                 "snapshot:card"])
        let refusing = ScriptRunner(target: RecordingTarget(answerable: false), tally: EventTally(), log: nil)
        await #expect(throws: ScriptFailure.self) { try await refusing.run([.choose("allow_session")]) }
    }
}
