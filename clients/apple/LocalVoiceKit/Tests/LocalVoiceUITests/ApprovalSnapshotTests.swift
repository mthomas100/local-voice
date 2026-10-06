import Foundation
import ImageIO
import LocalVoiceKit
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LocalVoiceUI
#if canImport(AppKit)
import AppKit
#endif

/// The approval card drawn by `ImageRenderer` for each kind of question the mock serves (PROTOCOL.md "Approvals"):
/// evidence of what the owner sees, with no app launched and no simulator booted. ImageRenderer cannot draw scroll
/// views, so the boxes are drawn clipped at their height (`approvalCardSnapshot`); the scrolling sizes are checked
/// with real AppKit layout below. With LV_SNAPSHOT_DIR set (`../build.sh snapshots`), the PNGs are kept there.
@MainActor
@Suite("Approval card snapshots")
struct ApprovalSnapshotTests {
    static let home = "/Users/owner"
    static let receivedAt = Date(timeIntervalSince1970: 1_791_300_000)
    static let twoChoices = [ApprovalChoice(id: "allow_once", label: "Do it"), ApprovalChoice(id: "deny", label: "Don't")]

    static func approval(_ id: String, _ summary: String, _ action: ApprovalAction,
                         choices: [ApprovalChoice] = twoChoices, timeoutMs: Int = 120_000) -> PendingApproval {
        PendingApproval(request: ConfirmRequest(id: .string(id), title: "", message: "", timeoutMs: timeoutMs,
                                                summary: summary, action: action, choices: choices),
                        receivedAt: receivedAt)
    }

    static let longText = (1...140).map { "Line \($0): the agent's notes from the session, kept word for word." }
        .joined(separator: "\n")

    /// The questions of the mock's scenarios (MockServer/mock_server.py, --approval).
    static let cases: [(String, PendingApproval)] = [
        ("write", approval("c1", "Create a new file plan.txt in your notes folder with the text below.",
                           ApprovalAction(tool: "write", effect: .create, path: "\(home)/notes/plan.txt",
                                          cwd: "\(home)/notes", space: "home", mode: "act",
                                          preview: "Plan for Saturday\n\n- walk before breakfast\n- call Mum at 11\n"))),
        ("edit", approval("c2", "Change one line in todo.md in your notes folder: 'buy milk' becomes 'buy oat milk'.",
                          ApprovalAction(tool: "edit", effect: .modify, path: "\(home)/notes/todo.md",
                                         cwd: "\(home)/notes", space: "home", mode: "act",
                                         preview: "--- a/todo.md\n+++ b/todo.md\n@@ -1,4 +1,4 @@\n # This week\n-- buy milk\n+- buy oat milk\n - post the parcel\n - book the dentist"))),
        ("bash", approval("c3", "Run a command that lists the 20 newest files in your Downloads folder.",
                          ApprovalAction(tool: "bash", effect: .run, command: "ls -lt ~/Downloads | head -20",
                                         cwd: home, space: "home", mode: "act"))),
        ("session", approval("c4", "Create a new page in your knowledge base titled 'Tea brewing notes'.",
                             ApprovalAction(tool: "kb", effect: .create,
                                            command: "kb new Analysis tea-brewing-notes --title 'Tea brewing notes'",
                                            cwd: "\(home)/kb", space: "home", mode: "act",
                                            preview: "# Tea brewing notes\n\nGreen tea: 80 °C for two minutes. Black tea: just off the boil, four minutes. Oolong: 90 °C, three short steeps.\n"),
                             choices: [ApprovalChoice(id: "allow_once", label: "Do it"),
                                       ApprovalChoice(id: "allow_session", label: "Allow this for the rest of the session"),
                                       ApprovalChoice(id: "deny", label: "Don't")])),
        ("long", approval("c5", "Create a new file session-notes.md in your notes folder with the text below.",
                          ApprovalAction(tool: "write", effect: .create, path: "\(home)/notes/session-notes.md",
                                         space: "home", mode: "act",
                                         preview: String(longText.prefix(4_000)) + "\n… [cut: 5,563 more characters]"))),
        ("delete", approval("c6", "Delete the file old-plan.txt from your notes folder.",
                            ApprovalAction(tool: "bash", effect: .delete, command: "rm ~/notes/old-plan.txt",
                                           path: "\(home)/notes/old-plan.txt", cwd: home, space: "home", mode: "act"),
                            timeoutMs: 14_000)),
        // An old-style request (title and message only), as early servers sent it.
        ("legacy", PendingApproval(request: ConfirmRequest(
            id: .string("q1"), title: "May I change your knowledge base?",
            message: "kb new Issue tea-brewing-notes --title Tea brewing notes",
            timeoutMs: 20_000), receivedAt: receivedAt)),
    ]

    @Test("every kind of question renders, in light and dark")
    func renders() throws {
        for (name, approval) in Self.cases {
            for scheme in [ColorScheme.light, .dark] {
                let card = ApprovalCard(approval: approval, waitingBehind: name == "bash" ? 1 : 0,
                                        now: Self.receivedAt.addingTimeInterval(6)) { _ in }
                let image = try #require(render(card, scheme: scheme), "\(name) did not render")
                #expect(image.width >= 780 && image.height > 300, "\(name): \(image.width)x\(image.height)")
                save(image, as: "approval-\(name)-\(scheme == .dark ? "dark" : "light").png")
            }
        }
    }

    @Test("the card says what each question shows: the cut marker pinned, the right buttons in order")
    func content() throws {
        let cases = Dictionary(uniqueKeysWithValues: Self.cases)
        #expect(cases["long"]?.content.preview?.cutMarker == "… [cut: 5,563 more characters]")
        #expect(cases["session"]?.content.choices.map(\.label)
            == ["Do it", "Allow this for the rest of the session", "Don't"])
        #expect(cases["delete"]?.content.choices.map(\.destructive) == [true, false])
        #expect(cases["legacy"]?.content.exact.first?.label == "Details")
        #expect(cases["edit"]?.content.preview?.kind == .diff)
        let diff = ApprovalCard.diff(try #require(cases["edit"]?.content.preview?.text))
        #expect(String(diff.characters) == cases["edit"]?.content.preview?.text, "colouring keeps every character")
    }

    #if canImport(AppKit)
    @Test("a box is as tall as its text up to its limit, then scrolls; the cards fit a screen (AppKit layout)")
    func layout() throws {
        func height(_ view: some View) -> CGFloat { NSHostingView(rootView: view.frame(width: 392)).fittingSize.height }
        let one = height(CodeBox(text: AttributedString("ls -lt ~/Downloads | head -20"), maxHeight: 120))
        let many = height(CodeBox(text: AttributedString(Self.longText), maxHeight: 120))
        #expect(one > 20 && one < 60, "one line: \(one)")
        #expect(abs(many - 120) < 1, "140 lines, capped: \(many)")
        for (name, approval) in Self.cases {
            let card = height(ApprovalCard(approval: approval, previewMaxHeight: 180) { _ in })
            #expect(card < 820, "\(name): \(card) pt high in the Mac panel")
        }
    }
    #endif

    private func render(_ view: some View, scheme: ColorScheme) -> CGImage? {
        let content = view
            .environment(\.approvalCardSnapshot, true)
            .padding(14)
            .frame(width: 420, alignment: .leading)
            .background(scheme == .dark ? Color.black : Color.white)
            .environment(\.colorScheme, scheme)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        return renderer.cgImage
    }

    private func save(_ image: CGImage, as name: String) {
        guard let dir = ProcessInfo.processInfo.environment["LV_SNAPSHOT_DIR"], !dir.isEmpty else { return }
        let folder = URL(fileURLWithPath: dir)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        guard let out = CGImageDestinationCreateWithURL(folder.appendingPathComponent(name) as CFURL,
                                                        UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(out, image, nil)
        CGImageDestinationFinalize(out)
    }
}
