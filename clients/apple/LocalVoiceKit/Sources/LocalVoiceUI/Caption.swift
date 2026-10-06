import Foundation

/// The agent's words as the apps show them. Replies arrive as the LLM wrote them, Markdown and all: against the real
/// orchestrator (2026-10-05) they carried `**…**` titles, `*…*` emphasis and `---` rule lines, which `Text(String)`
/// draws verbatim. Inline emphasis, code and links are drawn; rule lines and heading marks are dropped; text that does
/// not parse is shown as written.
public enum Caption {
    public static func attributed(_ text: String) -> AttributedString {
        let tidied = tidy(text)
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: tidied, options: options)) ?? AttributedString(tidied)
    }

    /// The same words without Markdown, for places that take a plain string (the Live Activity).
    public static func plain(_ text: String) -> String {
        String(attributed(text).characters)
    }

    /// Drops rule lines (`---`, `***`, `___`), heading marks, and the blank lines a dropped rule leaves behind.
    static func tidy(_ text: String) -> String {
        var lines: [Substring] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.count >= 3, let mark = trimmed.first, "-*_".contains(mark), trimmed.allSatisfy({ $0 == mark }) {
                continue
            }
            if trimmed.hasPrefix("#") {
                lines.append(Substring(trimmed.drop(while: { $0 == "#" }).drop(while: { $0 == " " })))
                continue
            }
            if trimmed.isEmpty, lines.last?.trimmingCharacters(in: .whitespaces).isEmpty ?? true { continue }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }
}
