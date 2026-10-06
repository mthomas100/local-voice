import Foundation
import Testing
@testable import LocalVoiceKit

/// The approval card's content and the queue of waiting questions (PROTOCOL.md "Approvals", 2026-10-05).
@Suite("Approvals")
struct ApprovalTests {
    /// An old-style request (title and message only), as early servers sent it.
    static let sessionRequest = ConfirmRequest(
        id: .string("q1"), title: "May I change your knowledge base?",
        message: "kb new Issue tea-brewing-notes --title Tea brewing notes",
        timeoutMs: 20_000)

    static func request(_ id: String, action: ApprovalAction? = nil, choices: [ApprovalChoice] = [],
                        timeoutMs: Int? = 120_000, summary: String = "Do the thing.") -> ConfirmRequest {
        ConfirmRequest(id: .string(id), title: "", message: "", timeoutMs: timeoutMs, summary: summary, action: action,
                       choices: choices)
    }

    static let twoChoices = [ApprovalChoice(id: "allow_once", label: "Do it"), ApprovalChoice(id: "deny", label: "Don't")]

    // MARK: The card

    @Test("a full request: the summary as the headline, the exact command and path, where it runs, the text, the choices")
    func fullCard() {
        let c = ApprovalContent(CodecTests.approval)
        #expect(c.headline == "Create a new page in your knowledge base titled 'Tea brewing notes'.")
        #expect(!c.isLegacy && c.effect == .create && c.effectLabel == "Creates")
        #expect(c.facts == ["kb", "home space", "act mode"])
        #expect(c.exact.map(\.kind) == [.command, .path] && c.exact.map(\.label) == ["Command", "New file"])
        #expect(c.exact[0].text == "kb new Analysis tea-brewing-notes --title 'Tea brewing notes'")
        #expect(c.cwd == "/Users/owner/kb")
        #expect(c.preview?.kind == .text && c.preview?.label == "Text to be written" && c.preview?.cutMarker == nil)
        #expect(c.choices.map(\.id) == ["allow_once", "allow_session", "deny"])
        #expect(c.choices.map(\.role) == [.primary, .secondary, .deny])
        #expect(c.choices.map(\.label) == ["Do it", "Allow this for the rest of the session", "Don't"])
        #expect(c.choicesOffered && c.spokenHint == "Or say “yes”, “yes, for this session” or “no”.")
    }

    @Test("an old-style request: its title as the headline, its message exactly, and Do it or Don't")
    func legacyCard() {
        let c = ApprovalContent(Self.sessionRequest)
        #expect(c.isLegacy && c.headline == "May I change your knowledge base?")
        #expect(c.exact == [.init(kind: .details, label: "Details", text: Self.sessionRequest.message)])
        #expect(c.preview == nil && c.effect == nil && c.facts.isEmpty && c.cwd == nil)
        #expect(c.choices.map(\.id) == ["allow_once", "deny"] && c.choices.map(\.label) == ["Do it", "Don't"])
        #expect(!c.choicesOffered && c.spokenHint == "Or say “yes” or “no”.")

        let untitled = ApprovalContent(ConfirmRequest(id: .int(1), title: " ", message: "write /tmp/a.txt", timeoutMs: nil))
        #expect(untitled.headline == "Your agent asks for permission" && untitled.exact.map(\.text) == ["write /tmp/a.txt"])
        #expect(ApprovalContent(ConfirmRequest(id: .int(2), title: "", message: "", timeoutMs: nil)).exact.isEmpty)
    }

    @Test("a shell command: the command as sent, where it runs, no path")
    func shellCard() {
        let command = "ls -lt ~/Downloads | head -20\n"
        let c = ApprovalContent(Self.request("c3", action: ApprovalAction(tool: "bash", effect: .run, command: command,
                                                                          cwd: "/Users/owner", space: "home",
                                                                          mode: "act"),
                                             choices: Self.twoChoices))
        #expect(c.exact == [.init(kind: .command, label: "Command", text: command)], "exact: untrimmed")
        #expect(c.cwd == "/Users/owner" && c.effectLabel == "Runs a command" && c.preview == nil)
    }

    @Test("a path's label says what happens to the file; a working directory shows only under a command")
    func pathLabels() {
        func label(_ effect: ApprovalEffect?) -> String? {
            ApprovalContent(Self.request("p", action: ApprovalAction(effect: effect, path: "/a", cwd: "/"))).exact.first?.label
        }
        #expect(label(.create) == "New file" && label(.modify) == "File to change" && label(.delete) == "File to delete")
        #expect(label(nil) == "File" && label(.run) == "File")
        #expect(ApprovalContent(Self.request("p", action: ApprovalAction(path: "/a", cwd: "/"))).cwd == nil)
    }

    @Test("an edit's diff: labelled Changes, every line's kind, and a Markdown list is not a diff")
    func diffPreview() {
        let diff = "--- a/todo.md\n+++ b/todo.md\n@@ -1,3 +1,3 @@\n # Todo\n-buy milk\n+buy oat milk\n--- not a header\n\\ No newline at end of file"
        let c = ApprovalContent(Self.request("e", action: ApprovalAction(tool: "edit", effect: .modify, path: "/n/todo.md",
                                                                         preview: diff)))
        #expect(c.preview?.kind == .diff && c.preview?.label == "Changes")
        #expect(ApprovalContent.diffLines(diff).map(\.0) == [.header, .header, .hunk, .context, .removed, .added,
                                                             .removed, .context])
        let list = "# Plan\n- walk before breakfast\n+ call Mum\n"
        #expect(!ApprovalContent.isDiff(list))
        let write = ApprovalContent(Self.request("w", action: ApprovalAction(tool: "write", effect: .create, preview: list)))
        #expect(write.preview?.kind == .text && write.preview?.label == "Text to be written")
        let other = ApprovalContent(Self.request("n", action: ApprovalAction(tool: "web", effect: .network, preview: "GET /")))
        #expect(other.preview?.label == "Preview")
    }

    @Test("the cut marker is pinned only when the server cut the preview")
    func cutMarker() {
        let body = String(repeating: "All work and no play makes a dull boy.\n", count: 103)  // 4,017 characters
        #expect(ApprovalContent.cutMarker(in: body + "… [cut: 2,180 more characters]") == "… [cut: 2,180 more characters]")
        // The orchestrator's own marker (2026-10-05).
        #expect(ApprovalContent.cutMarker(in: body + "\n… [cut here: 1,904 more characters not shown]")
            == "… [cut here: 1,904 more characters not shown]")
        #expect(ApprovalContent.cutMarker(in: body + "\n[truncated at 4,000 characters]\n\n")
            == "[truncated at 4,000 characters]")
        #expect(ApprovalContent.cutMarker(in: body + "the last words … [cut: 12 more lines]") == "… [cut: 12 more lines]",
                "a marker at the end of the last line")
        #expect(ApprovalContent.cutMarker(in: "short text\n… [cut: 2 more characters]") == nil, "under 2,000: never cut")
        #expect(ApprovalContent.cutMarker(in: body + "[read more](https://example.com)") == nil)
        #expect(ApprovalContent.cutMarker(in: body + "- cut the onions") == nil, "a list item about cutting")
        #expect(ApprovalContent.cutMarker(in: body) == nil)
        let c = ApprovalContent(Self.request("w", action: ApprovalAction(effect: .create, path: "/p",
                                                                         preview: body + "… [cut: 9 more characters]")))
        #expect(c.preview?.cutMarker == "… [cut: 9 more characters]" && c.preview?.text.hasSuffix("characters]") == true,
                "the marker also stays in the text, where the server put it")
    }

    @Test("choices: the server's order, each id once, a Don't always there, defaults for missing labels")
    func choices() {
        func ids(_ offered: [ApprovalChoice], effect: ApprovalEffect? = nil) -> [ApprovalContent.Choice] {
            ApprovalContent.resolveChoices(offered, effect: effect)
        }
        let denyFirst = ids([ApprovalChoice(id: "deny"), ApprovalChoice(id: "allow_once")])
        #expect(denyFirst.map(\.id) == ["deny", "allow_once"] && denyFirst.map(\.role) == [.deny, .primary])
        #expect(ids([ApprovalChoice(id: "allow_once"), ApprovalChoice(id: "allow_once", label: "Again"),
                     ApprovalChoice(id: "deny")]).map(\.label) == ["Do it", "Don't"])
        let noDeny = ids([ApprovalChoice(id: "allow_once", label: "Go ahead")])
        #expect(noDeny.map(\.id) == ["allow_once", "deny"] && noDeny.map(\.label) == ["Go ahead", "Don't"])
        #expect(ids([ApprovalChoice(id: "allow_once", label: "  "), ApprovalChoice(id: "deny")]).map(\.label)
            == ["Do it", "Don't"])
        let unknown = ids([ApprovalChoice(id: "allow_once"), ApprovalChoice(id: "ask_later"), ApprovalChoice(id: "deny")])
        #expect(unknown[1].label == "Ask later" && unknown[1].role == .secondary && !unknown[1].allows)
        let delete = ids(Self.twoChoices, effect: .delete)
        #expect(delete.map(\.destructive) == [true, false], "the allow of a delete is red; Don't never is")
    }

    @Test("confirm_cancel's codes in words: answered, timeout and overtaken; anything else as written")
    func withdrawalWords() {
        #expect(ApprovalContent.withdrawal("answered") == ("answered by voice", false))
        #expect(ApprovalContent.withdrawal("timeout") == ("no answer in time", true))
        #expect(ApprovalContent.withdrawal("timed out").timedOut, "the phrase means the same")
        #expect(ApprovalContent.withdrawal("overtaken").words == "overtaken by what came next")
        #expect(ApprovalContent.withdrawal("the space changed") == ("the space changed", false))
        #expect(ApprovalContent.withdrawal("") == ("", false))
    }

    // MARK: The queue

    @Test("a second question waits behind the first; the same id again restarts its time in place")
    func queueOrder() {
        var q = ApprovalQueue()
        let t0 = Date(timeIntervalSince1970: 1_000_000)
        #expect(q.receive(Self.request("c1"), at: t0) == .new)
        #expect(q.receive(Self.request("c2"), at: t0 + 1) == .new)
        #expect(q.waiting.map(\.id) == [.string("c1"), .string("c2")] && q.current?.id == .string("c1"))
        #expect(q.receive(Self.request("c1", summary: "Asked again."), at: t0 + 30) == .repeated)
        #expect(q.waiting.map(\.id) == [.string("c1"), .string("c2")])
        #expect(q.current?.content.headline == "Asked again." && q.current?.deadline == t0 + 150)
    }

    @Test("an answer says confirmed and the choice; an old-style request gets confirmed alone; nothing for the rest")
    func answers() {
        var q = ApprovalQueue()
        let t0 = Date()
        q.receive(CodecTests.approval, at: t0)
        q.receive(Self.request("c9", choices: Self.twoChoices), at: t0)
        q.receive(Self.sessionRequest, at: t0)
        #expect(q.answer(id: .string("c8"), choice: "allow_forever") == nil, "not on the card: nothing sent")
        #expect(q.waiting.count == 3)
        let session = q.answer(id: .string("c8"), choice: "allow_session")
        #expect(session?.message == .confirmResponse(id: .string("c8"), confirmed: true, choice: "allow_session"))
        #expect(session?.choice.label == "Allow this for the rest of the session")
        #expect(q.answer(id: .string("c8"), choice: "allow_session") == nil, "answered already")
        #expect(q.answer(id: .string("c9"), choice: "deny")?.message
            == .confirmResponse(id: .string("c9"), confirmed: false, choice: "deny"))
        #expect(q.answer(id: .string("q1"), choice: "allow_once")?.message
            == .confirmResponse(id: .string("q1"), confirmed: true, choice: nil), "an older server reads only confirmed")
        #expect(q.waiting.isEmpty)
    }

    @Test("a cancel closes only a waiting question; time runs out only after the grace; a dropped connection closes all")
    func closing() {
        var q = ApprovalQueue()
        let t0 = Date(timeIntervalSince1970: 2_000_000)
        q.receive(Self.request("c1", timeoutMs: 3_000), at: t0)
        q.receive(Self.request("c2", timeoutMs: nil), at: t0)
        q.receive(Self.request("c3"), at: t0)
        #expect(q.cancel(id: .string("nope")) == nil)
        #expect(q.nextExpiry() == t0 + 3 + ApprovalQueue.grace)
        #expect(q.expire(now: t0 + 5.9).isEmpty, "a current server's cancel, with its reason, comes first")
        #expect(q.expire(now: t0 + 6).map(\.id) == [.string("c1")])
        #expect(q.cancel(id: .string("c3"))?.id == .string("c3"))
        #expect(q.nextExpiry() == nil, "a question with no time never runs out here")
        #expect(q.expire(now: t0 + 86_400).isEmpty)
        #expect(q.closeAll().map(\.id) == [.string("c2")] && q.waiting.isEmpty)
    }

    @Test("seconds left round up and stop at zero; no time given, none shown")
    func secondsLeft() {
        let t0 = Date(timeIntervalSince1970: 3_000_000)
        let a = PendingApproval(request: Self.request("c", timeoutMs: 120_000), receivedAt: t0)
        #expect(a.secondsLeft(at: t0) == 120 && a.secondsLeft(at: t0 + 5.2) == 115 && a.secondsLeft(at: t0 + 200) == 0)
        #expect(PendingApproval(request: Self.request("c", timeoutMs: 0), receivedAt: t0).secondsLeft(at: t0) == nil)
    }

    @Test("the log: a shown card carries all of it; an answer names the question")
    func logLines() throws {
        let a = PendingApproval(request: CodecTests.approval, receivedAt: Date())
        let shown = JSONValue.object(a.logFields("shown"))
        #expect(shown["event"] == .string("approval") && shown["phase"] == .string("shown"))
        #expect(shown["exact"]?.arrayValue?.count == 2 && shown["preview"]?["kind"] == .string("text"))
        #expect(shown["choices"]?.arrayValue?.compactMap { $0["id"]?.stringValue } == ["allow_once", "allow_session", "deny"])
        #expect(shown["timeout_ms"] == .int(120_000) && shown["preview"]?["cut_marker"] == .null)
        let answered = JSONValue.object(a.logFields("answered", [("choice", "deny")]))
        #expect(answered["exact"] == nil && answered["choice"] == .string("deny") && answered["id"] == .string("c8"))
    }
}
