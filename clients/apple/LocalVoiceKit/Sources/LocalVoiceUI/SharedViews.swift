import LocalVoiceKit
import SwiftUI

/// The agent's state in one glance: symbol, line, and a colour per state.
public struct StatusPill: View {
    let model: VoiceSessionModel

    public init(model: VoiceSessionModel) {
        self.model = model
    }

    public var body: some View {
        Label(model.statusLine, systemImage: model.symbolName)
            .font(.callout.weight(.medium))
            .lineLimit(1)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(tint.opacity(0.18), in: .capsule)
            .foregroundStyle(tint)
            .contentTransition(.symbolEffect(.replace))
            .accessibilityLabel(Text("Agent: \(model.statusLine)"))
    }

    private var tint: Color {
        if !model.isReady { return .secondary }
        if model.hold.phase == .held { return .orange }
        if model.talk == .talking { return .red }
        if model.pendingApproval != nil { return .orange }
        if model.toolLabel != nil { return .purple }
        switch model.agentState {
        case .thinking: return .indigo
        case .speaking: return .blue
        default: return .green
        }
    }
}

/// The conversation so far: what the user said, what the agent said (cut short where it was interrupted).
public struct TranscriptList: View {
    let entries: [VoiceSessionModel.Entry]
    let partial: String?

    public init(entries: [VoiceSessionModel.Entry], partial: String?) {
        self.entries = entries
        self.partial = partial
    }

    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(entries) { entry in
                        EntryRow(entry: entry)
                            .id(entry.id)
                    }
                    if let partial, !partial.isEmpty {
                        Text(partial)
                            .italic()
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                            .id("partial")
                    }
                }
                .padding(.vertical, 8)
            }
            .scrollIndicators(.hidden)
            .onChange(of: entries.last?.text) {
                if let last = entries.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
            .onChange(of: entries.count) {
                if let last = entries.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }
}

struct EntryRow: View {
    let entry: VoiceSessionModel.Entry

    var body: some View {
        switch entry.role {
        case .user:
            Text(entry.text)
                .padding(10)
                .background(Color.accentColor.opacity(0.15), in: .rect(cornerRadius: 12))
                .frame(maxWidth: .infinity, alignment: .trailing)
                .textSelection(.enabled)
        case .agent:
            VStack(alignment: .leading, spacing: 2) {
                Text(Caption.attributed(entry.text))
                    .textSelection(.enabled)
                if entry.interrupted {
                    Text("interrupted")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 12))
            .frame(maxWidth: .infinity, alignment: .leading)
        case .notice:
            Text(entry.text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

/// What the agent is doing right now ("reading your journal").
public struct ToolActivityRow: View {
    let label: String

    public init(label: String) {
        self.label = label
    }

    public var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text(label.prefix(1).uppercased() + label.dropFirst())
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}

/// The Mac's GPU is held by a render: the agent cannot think or speak with its models until it ends.
public struct HoldBanner: View {
    let hold: HoldStatus

    public init(hold: HoldStatus) {
        self.hold = hold
    }

    public var body: some View {
        Label(hold.why.isEmpty ? "The Mac is busy; I'll answer when it's free."
                  : "The Mac is busy (\(hold.why)); I'll answer when it's free.",
              systemImage: "hourglass")
            .font(.callout)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.15), in: .rect(cornerRadius: 10))
    }
}
