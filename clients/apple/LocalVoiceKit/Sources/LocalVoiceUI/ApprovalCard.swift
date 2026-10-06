import Foundation
import LocalVoiceKit
import SwiftUI

extension EnvironmentValues {
    /// Snapshots draw a long box clipped at its height instead of scrolling (ImageRenderer cannot draw scroll views).
    @Entry public var approvalCardSnapshot = false
}

/// The agent asks permission (PROTOCOL.md "Approvals"): what will happen, the exact command or path, what will be
/// written, and one button per answer the server offers, in its order, like a coding agent's permission prompt.
///
/// Only a tap answers it. No key is bound to any button, so a key press meant for something else (Return in the menu's
/// text field, Escape to close the menu) can never answer; "Don't" is never a default action, and neither is anything
/// else. When the time runs out nothing is chosen: silence is no, and the server withdraws the question.
public struct ApprovalCard: View {
    let approval: PendingApproval
    let waitingBehind: Int
    let previewMaxHeight: CGFloat
    let framed: Bool
    let now: Date?
    let answer: (String) -> Void

    /// `now` fixes the clock (snapshots); otherwise the time left counts down by itself. `framed` draws the card's own
    /// background (the Mac's panel and menu); a sheet is its own frame.
    public init(approval: PendingApproval, waitingBehind: Int = 0, previewMaxHeight: CGFloat = 200, framed: Bool = true,
                now: Date? = nil, answer: @escaping (String) -> Void) {
        self.approval = approval
        self.waitingBehind = waitingBehind
        self.previewMaxHeight = previewMaxHeight
        self.framed = framed
        self.now = now
        self.answer = answer
    }

    public var body: some View {
        let content = approval.content
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("Permission needed", systemImage: "hand.raised.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.orange)
                Spacer(minLength: 8)
                if approval.deadline != nil { TimeLeft(approval: approval, now: now) }
            }
            Text(content.headline)
                .font(.headline)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
            if content.effectLabel != nil || !content.facts.isEmpty {
                FactsRow(effect: content.effect, effectLabel: content.effectLabel, facts: content.facts)
            }
            ForEach(content.exact) { block in
                VStack(alignment: .leading, spacing: 4) {
                    BoxLabel(text: block.label)
                    CodeBox(text: AttributedString(block.text), maxHeight: 120)
                    if block.kind == .command, let cwd = content.cwd {
                        Label("Runs in \(cwd)", systemImage: "folder")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if let preview = content.preview {
                VStack(alignment: .leading, spacing: 4) {
                    BoxLabel(text: preview.label)
                    CodeBox(text: preview.kind == .diff ? Self.diff(preview.text) : AttributedString(preview.text),
                            maxHeight: previewMaxHeight, small: true)
                    if let marker = preview.cutMarker {
                        // Pinned under the box: the server's own words for where it cut the preview, seen without
                        // scrolling to the end.
                        Label(marker, systemImage: "scissors")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .accessibilityLabel("The preview is shortened: \(marker)")
                    }
                }
            }
            VStack(spacing: 8) {
                ForEach(content.choices) { choice in
                    Button(choice.label) { answer(choice.id) }
                        .buttonStyle(ApprovalButtonStyle(role: choice.role, destructive: choice.destructive))
                }
            }
            .padding(.top, 2)
            Text(content.spokenHint)
                .font(.caption)
                .foregroundStyle(.secondary)
            if waitingBehind > 0 {
                Text(waitingBehind == 1 ? "1 more question waiting" : "\(waitingBehind) more questions waiting")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(framed ? 14 : 0)
        .background {
            if framed {
                RoundedRectangle(cornerRadius: 14).fill(.thinMaterial)
                RoundedRectangle(cornerRadius: 14).strokeBorder(.orange.opacity(0.45), lineWidth: 1)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Permission needed")
    }

    /// A unified diff with its added lines green, removed lines red, hunk headers blue and file headers dimmed.
    static func diff(_ text: String) -> AttributedString {
        let lines = ApprovalContent.diffLines(text)
        var out = AttributedString()
        for (i, (kind, line)) in lines.enumerated() {
            var run = AttributedString(String(line) + (i < lines.count - 1 ? "\n" : ""))
            switch kind {
            case .added: run.foregroundColor = .green
            case .removed: run.foregroundColor = .red
            case .hunk: run.foregroundColor = .blue
            case .header: run.foregroundColor = .secondary
            case .context: break
            }
            out += run
        }
        return out
    }
}

/// "1:54 left", counting down; orange in the last 15 seconds.
struct TimeLeft: View {
    let approval: PendingApproval
    let now: Date?
    @Environment(\.approvalCardSnapshot) private var snapshot

    var body: some View {
        if let now {
            label(at: now)
        } else if snapshot {
            label(at: Date())
        } else {
            TimelineView(.periodic(from: .now, by: 1)) { context in label(at: context.date) }
        }
    }

    private func label(at date: Date) -> some View {
        let left = approval.secondsLeft(at: date) ?? 0
        return Text(left > 0 ? "\(Duration.seconds(left).formatted(.time(pattern: .minuteSecond))) left" : "Time's up")
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(left <= 15 ? AnyShapeStyle(.orange) : AnyShapeStyle(.secondary))
            .accessibilityLabel(left > 0
                ? "\(Duration.seconds(left).formatted(.units(allowed: [.minutes, .seconds], width: .wide))) left to answer"
                : "Time's up")
    }
}

/// The effect as a small badge, then the tool, the space and the mode.
struct FactsRow: View {
    let effect: ApprovalEffect?
    let effectLabel: String?
    let facts: [String]

    var body: some View {
        HStack(spacing: 8) {
            if let effectLabel {
                Label(effectLabel, systemImage: symbol)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .foregroundStyle(tint)
                    .background(tint.opacity(0.14), in: .capsule)
            }
            if !facts.isEmpty {
                Text(facts.joined(separator: " · "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var symbol: String {
        switch effect {
        case .create?: return "doc.badge.plus"
        case .modify?: return "pencil"
        case .delete?: return "trash"
        case .run?: return "terminal"
        case .network?: return "network"
        default: return "questionmark.circle"
        }
    }

    private var tint: Color {
        switch effect {
        case .delete?: return .red
        case .network?: return .orange
        default: return .blue
        }
    }
}

struct BoxLabel: View {
    let text: String

    var body: some View {
        Text(text.uppercased())
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Monospaced, selectable, as tall as its text up to `maxHeight`, then scrolling.
struct CodeBox: View {
    let text: AttributedString
    let maxHeight: CGFloat
    var small = false
    @Environment(\.approvalCardSnapshot) private var snapshot

    var body: some View {
        Group {
            if snapshot {
                code.frame(maxHeight: maxHeight, alignment: .top)
                    .fixedSize(horizontal: false, vertical: true)
                    .clipped()
            } else {
                ScrollView { code.textSelection(.enabled) }
                    .frame(maxHeight: maxHeight)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 8))
    }

    private var code: some View {
        Text(text)
            .font(.system(small ? .caption : .callout, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(8)
    }
}

/// Full-width buttons drawn by SwiftUI itself (the same on the Mac and the iPhone, and in snapshots): the first allow
/// filled, the others and "Don't" outlined alike, so refusing looks as easy as agreeing.
struct ApprovalButtonStyle: ButtonStyle {
    let role: ApprovalContent.Choice.Role
    let destructive: Bool
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        let primary = role == .primary
        let tint: Color = destructive ? .red : .accentColor
        return configuration.label
            .font(.body.weight(primary ? .semibold : .regular))
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity, minHeight: 22)
            .padding(.vertical, 9)
            .padding(.horizontal, 12)
            .foregroundStyle(primary ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
            .background {
                RoundedRectangle(cornerRadius: 10)
                    .fill(primary ? AnyShapeStyle(tint) : AnyShapeStyle(.fill.tertiary))
                if !primary { RoundedRectangle(cornerRadius: 10).strokeBorder(.separator, lineWidth: 1) }
            }
            .opacity(configuration.isPressed ? 0.75 : isEnabled ? 1 : 0.5)
            .contentShape(.rect(cornerRadius: 10))
    }
}
