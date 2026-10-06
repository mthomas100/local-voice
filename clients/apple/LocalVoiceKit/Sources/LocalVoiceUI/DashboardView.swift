import LocalVoiceKit
import SwiftUI

/// The dashboard: where the agent is working ("your atlas journal"), in which mode, how far it may act, with
/// which model, what it is doing, whether the GPU is free, and how fast the last turn was. Read-only and made of plain
/// views (no scroll views, pickers or text fields), so `ImageRenderer` can draw it for unattended evidence.
public struct DashboardCard: View {
    let dashboard: Dashboard
    let connected: Bool

    public init(dashboard: Dashboard, connected: Bool) {
        self.dashboard = dashboard
        self.connected = connected
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "square.stack.3d.up.fill")
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(dashboard.spaceTitle.map(Self.capitalized) ?? "Not connected yet")
                        .font(.headline)
                    if let space = dashboard.space, dashboard.spaceDescription != nil {
                        Text("space “\(space)”")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 8)
                if let mode = dashboard.mode { ModeBadge(mode: mode) }
            }
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 10, verticalSpacing: 5) {
                if let tier = dashboard.tier {
                    row("Tier", Self.tierLine(tier))
                }
                if let model = dashboard.model {
                    row("Model", model)
                }
                row("Agent", agentLine)
                row("GPU", dashboard.holdLine, tint: dashboard.hold.phase == .open ? nil : .orange)
                if let latency = dashboard.latencyLine {
                    row("Last turn", latency)
                }
            }
            .font(.callout)
            footer
        }
        .accessibilityElement(children: .combine)
    }

    private var agentLine: String {
        if !connected { return "not connected" }
        if let tool = dashboard.tool { return Self.capitalized(tool.label ?? tool.name) }
        switch dashboard.state {
        case .thinking?: return "thinking"
        case .speaking?: return "speaking"
        case .listening?: return "listening"
        case .held?: return "waiting for the GPU"
        case .error?: return "something went wrong"
        default: return "idle"
        }
    }

    @ViewBuilder
    private var footer: some View {
        if let line = dashboard.switchLine {
            Label(line, systemImage: dashboard.pendingSwitch != nil ? "arrow.triangle.2.circlepath" : "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        if let problem = dashboard.statusProblem {
            Label("Status unavailable: \(problem)", systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
        } else if let at = dashboard.statusAt {
            Text("Status from \(at.formatted(date: .omitted, time: .standard))")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private func row(_ label: String, _ value: String, tint: Color? = nil) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(.secondary)
                .gridColumnAlignment(.trailing)
            Text(value)
                .foregroundStyle(tint ?? .primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    static func capitalized(_ s: String) -> String { s.prefix(1).uppercased() + s.dropFirst() }

    /// SPACES.md's tiers, in words: `ask` makes every acting tool need a spoken yes.
    static func tierLine(_ tier: String) -> String {
        switch tier {
        case "readonly": return "read only"
        case "ask": return "asks before it changes anything"
        case "trusted": return "trusted to act"
        default: return tier
        }
    }
}

/// `conversation` or `act`, as a small capsule.
public struct ModeBadge: View {
    let mode: String

    public init(mode: String) {
        self.mode = mode
    }

    public var body: some View {
        Text(DashboardCard.capitalized(mode))
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background((mode == "act" ? Color.orange : Color.accentColor).opacity(0.18), in: .capsule)
            .foregroundStyle(mode == "act" ? Color.orange : Color.accentColor)
            .accessibilityLabel(Text("\(mode) mode"))
    }
}

/// The dashboard in one line, for the floating panel and the top of the talk screen: "atlas · act · trusted ·
/// local/qwen38 · last 844 ms".
public struct DashboardStrip: View {
    let dashboard: Dashboard

    public init(dashboard: Dashboard) {
        self.dashboard = dashboard
    }

    public var body: some View {
        HStack(spacing: 6) {
            if dashboard.hold.phase != .open {
                Image(systemName: "hourglass")
                    .foregroundStyle(.orange)
            }
            Text(line)
                .lineLimit(1)
                .truncationMode(.middle)
            if dashboard.pendingSwitch != nil {
                ProgressView()
                    .controlSize(.mini)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
    }

    var line: String {
        var parts = [dashboard.space, dashboard.mode, dashboard.tier, dashboard.model].compactMap { $0 }
        if let ms = dashboard.lastTurn?.eosToFirstAudioMs, ms > 0 { parts.append("last \(Int(ms.rounded())) ms") }
        return parts.isEmpty ? "No status yet" : parts.joined(separator: " · ")
    }
}

/// Switching space and mode. The pickers show what the server says, not what was picked: a refused or unanswered
/// switch snaps back, and the line below says why.
public struct SpaceModeControls: View {
    let model: VoiceSessionModel

    public init(model: VoiceSessionModel) {
        self.model = model
    }

    public var body: some View {
        let d = model.dashboard
        VStack(alignment: .leading, spacing: 8) {
            Picker("Space", selection: Binding(get: { d.space ?? "" }, set: { model.switchSpace($0) })) {
                ForEach(spaceChoices(d), id: \.name) { space in
                    Text(space.description.map { "\(DashboardCard.capitalized($0)) (\(space.name))" } ?? space.name)
                        .tag(space.name)
                }
            }
            .pickerStyle(.menu)
            Picker("Mode", selection: Binding(get: { d.mode ?? "conversation" }, set: { model.switchMode($0) })) {
                ForEach(Dashboard.modes, id: \.self) { mode in
                    Text(DashboardCard.capitalized(mode)).tag(mode)
                }
            }
            .pickerStyle(.segmented)
        }
        .disabled(!model.isReady || d.pendingSwitch != nil)
    }

    /// The server's spaces; the current one even before a status lists them.
    private func spaceChoices(_ d: Dashboard) -> [ServerStatus.Space] {
        var choices = d.spaces
        if let current = d.space, !choices.contains(where: { $0.name == current }) {
            choices.insert(ServerStatus.Space(name: current, description: d.spaceDescription), at: 0)
        }
        return choices
    }
}
