import Foundation
import Testing
@testable import LocalVoiceUI

@Suite("Agent captions")
struct CaptionTests {
    /// The reply the real orchestrator streamed on 2026-10-05 (MockServer/runs/20261005-143434, push-to-talk barge-in).
    @Test("a rule line and the blank lines around it go; a bold title is drawn bold, without its asterisks")
    func realReply() {
        let text = "Here you go.\n\n---\n\n**The Weight of Light**"
        #expect(Caption.tidy(text) == "Here you go.\n\n**The Weight of Light**")
        let shown = Caption.attributed(text)
        #expect(String(shown.characters) == "Here you go.\n\nThe Weight of Light")
        let bold = shown.runs.first { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true }
        #expect(bold.map { String(shown[$0.range].characters) } == "The Weight of Light")
    }

    @Test("emphasis, code and headings lose their marks; plain text and unparsed marks stay as written")
    func marks() {
        #expect(Caption.plain("A cow, famously in *Hitchhiker's Guide*.") == "A cow, famously in Hitchhiker's Guide.")
        #expect(Caption.plain("Run `ls` first.") == "Run ls first.")
        #expect(Caption.plain("## Tomorrow\nRain.") == "Tomorrow\nRain.")
        #expect(Caption.plain("Paris.") == "Paris.")
        #expect(Caption.plain("2 * 3 is 6, and - is minus") == "2 * 3 is 6, and - is minus")
        #expect(Caption.plain("**half a title") == "**half a title", "a delta that has not closed its mark yet")
    }
}
