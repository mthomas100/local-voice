import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

/// The talk screen: the agent's state, the conversation, what it is doing, and the controls (hold to talk,
/// hands-free, stop, type).
struct TalkView: View {
    let hub: SessionHub
    @State private var typed = ""
    // -LVShowStatus YES opens the status sheet at launch (unattended screenshots); it is only read, never saved.
    @State private var sheet: TalkSheet? = UserDefaults.standard.bool(forKey: "LVShowStatus") ? .status : nil

    var body: some View {
        let model = hub.model
        NavigationStack {
            VStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    StatusPill(model: model)
                    // Where the agent is and how it works: tap for the dashboard and the space and mode switches.
                    Button { sheet = .status } label: {
                        HStack(spacing: 4) {
                            DashboardStrip(dashboard: model.dashboard)
                            Image(systemName: "chevron.right")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityHint("Shows the agent's space, mode and status, and switches them")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                if model.hold.phase == .held { HoldBanner(hold: model.hold) }
                if let problem = model.problem {
                    Text(problem)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                TranscriptList(entries: model.entries, partial: model.partialTranscript)
                if let label = model.toolLabel {
                    ToolActivityRow(label: label)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                ControlsRow(hub: hub)
                HStack {
                    TextField("Type instead", text: $typed)
                        .textFieldStyle(.roundedBorder)
                        .submitLabel(.send)
                        .onSubmit(sendTyped)
                    Button("Send", systemImage: "arrow.up.circle.fill", action: sendTyped)
                        .labelStyle(.iconOnly)
                        .font(.title2)
                        .disabled(typed.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
            .navigationTitle("Local Voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Status", systemImage: "gauge.with.dots.needle.33percent") { sheet = .status }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Settings", systemImage: "gearshape") { sheet = .settings }
                }
            }
            // One sheet at a time (SwiftUI shows no second one on top): a question replaces whatever is open, and
            // the sheet goes when the question is answered or withdrawn.
            .sheet(item: $sheet) { which in
                switch which {
                case .settings: SettingsView(hub: hub)
                case .status: DashboardSheet(model: model).presentationDetents([.medium, .large])
                case .approval: ApprovalSheet(model: model)
                }
            }
            .onChange(of: model.pendingApproval?.id, initial: true) { _, waiting in
                if waiting != nil {
                    sheet = .approval
                } else if sheet == .approval {
                    sheet = nil
                }
            }
        }
    }

    private func sendTyped() {
        hub.model.send(text: typed)
        typed = ""
    }
}

enum TalkSheet: String, Identifiable {
    case settings, status, approval
    var id: Self { self }
}

/// The agent's question on its own sheet (PROTOCOL.md "Approvals"): it stays until answered or withdrawn, so it cannot
/// be swiped away, though it can be pulled down to half height to see the conversation.
struct ApprovalSheet: View {
    let model: VoiceSessionModel
    @State private var detent = PresentationDetent.large

    var body: some View {
        ScrollView {
            if let approval = model.pendingApproval {
                ApprovalCard(approval: approval, waitingBehind: model.approvals.waiting.count - 1,
                             previewMaxHeight: 280, framed: false) { model.answer($0, to: approval.id) }
                    .padding(20)
            }
        }
        .presentationDetents([.medium, .large], selection: $detent)
        .presentationDragIndicator(.visible)
        .interactiveDismissDisabled()
    }
}

struct ControlsRow: View {
    let hub: SessionHub

    var body: some View {
        let model = hub.model
        HStack(alignment: .center, spacing: 28) {
            Button {
                Task { await hub.toggleHandsFree() }
            } label: {
                Label(model.handsFree || hub.inCall ? "End" : "Hands-free",
                      systemImage: model.handsFree || hub.inCall ? "phone.down.fill" : "waveform")
                    .labelStyle(.verticalLabel)
            }
            .tint(model.handsFree || hub.inCall ? .red : .accentColor)
            .accessibilityHint("Starts or ends an open-microphone conversation")

            TalkButton(model: model)

            Button {
                model.stopSpeaking()
            } label: {
                Label("Stop", systemImage: "stop.fill")
                    .labelStyle(.verticalLabel)
            }
            .disabled(!model.isPlaying)
            .accessibilityHint("Stops the agent talking")
        }
        .padding(.vertical, 6)
    }
}

/// Hold to talk: press starts a push-to-talk turn (and barges in if the agent is speaking), release ends it.
struct TalkButton: View {
    let model: VoiceSessionModel
    @State private var pressed = false

    var body: some View {
        ZStack {
            Circle()
                .fill(pressed ? Color.red : Color.accentColor)
                .frame(width: 92, height: 92)
                .shadow(radius: pressed ? 10 : 3)
            Image(systemName: pressed ? "mic.fill" : "mic")
                .font(.system(size: 36, weight: .semibold))
                .foregroundStyle(.white)
        }
        .scaleEffect(pressed ? 1.08 : 1)
        .animation(.spring(duration: 0.2), value: pressed)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !pressed else { return }
                    pressed = true
                    model.pressTalk()
                }
                .onEnded { _ in
                    pressed = false
                    model.releaseTalk()
                }
        )
        .accessibilityElement()
        .accessibilityLabel(pressed ? "Talking" : "Hold to talk")
        .accessibilityAddTraits(.isButton)
        .accessibilityAction(named: pressed ? "Stop talking" : "Start talking") {
            if pressed {
                pressed = false
                model.releaseTalk()
            } else {
                pressed = true
                model.pressTalk()
            }
        }
    }
}

struct VerticalLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(spacing: 4) {
            configuration.icon.font(.title2)
            configuration.title.font(.caption)
        }
        .frame(width: 76)
    }
}

extension LabelStyle where Self == VerticalLabelStyle {
    static var verticalLabel: VerticalLabelStyle { VerticalLabelStyle() }
}
