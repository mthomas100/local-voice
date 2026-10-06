import AppKit
import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

/// A small floating panel that appears while you talk and while the agent answers: it never takes focus from the app
/// you are in (non-activating), follows you across Spaces and full-screen apps, and fades out a few seconds after the
/// turn ends.
@MainActor
final class FloatingPanelController {
    private let panel: NSPanel
    private var hideTask: Task<Void, Never>?
    var pinned = false
    /// Where the panel's top edge was put (or dragged): AppKit keeps a window's bottom-left corner when its content
    /// grows, which would push a tall approval card's headline off the top of the screen.
    private var top: CGFloat?
    private var observers: [any NSObjectProtocol] = []

    init(model: VoiceSessionModel) {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 200),
                        styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView, .utilityWindow, .hudWindow],
                        backing: .buffered, defer: true)
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isMovableByWindowBackground = true
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        let host = NSHostingView(rootView: PanelView(model: model))
        host.sizingOptions = [.preferredContentSize]
        panel.contentView = host
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: NSWindow.didResizeNotification, object: panel, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.keepTop() }
            },
            center.addObserver(forName: NSWindow.didMoveNotification, object: panel, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { if let self { self.top = self.panel.frame.maxY } }
            },
        ]
    }

    /// The panel grows and shrinks downward from its top edge, and never below the screen's bottom.
    private func keepTop() {
        guard let top, abs(panel.frame.maxY - top) > 0.5 else { return }
        var origin = NSPoint(x: panel.frame.minX, y: top - panel.frame.height)
        if let bottom = panel.screen?.visibleFrame.minY { origin.y = max(origin.y, bottom) }
        panel.setFrameOrigin(origin)
    }

    var isVisible: Bool { panel.isVisible }

    /// Pinned, the panel stays up as the dashboard's widget; unpinned, it follows the conversation again.
    func setPinned(_ on: Bool, model: VoiceSessionModel) {
        pinned = on
        if on { show() } else { update(for: model) }
    }

    func show() {
        hideTask?.cancel()
        guard !panel.isVisible else { return }
        position()
        panel.orderFrontRegardless()
    }

    func hide(after seconds: Double) {
        guard !pinned else { return }
        hideTask?.cancel()
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.panel.orderOut(nil)
        }
    }

    /// Follow the model: visible while talking, thinking, speaking, waiting on a tool or a confirmation.
    func update(for model: VoiceSessionModel) {
        // A question stays on screen until it is answered or withdrawn.
        let busy = model.talk != .idle || model.isPlaying || model.toolLabel != nil || model.pendingApproval != nil
            || model.agentState == .thinking || model.agentState == .speaking
        if busy { show() } else if panel.isVisible { hide(after: 4) }
    }

    private func position() {
        let screen = NSScreen.screens.first { $0.frame.contains(NSEvent.mouseLocation) } ?? NSScreen.main
        guard let frame = screen?.visibleFrame else { return }
        let size = panel.frame.size
        panel.setFrameOrigin(NSPoint(x: frame.midX - size.width / 2, y: frame.maxY - size.height - 12))
        top = panel.frame.maxY
    }
}

struct PanelView: View {
    let model: VoiceSessionModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                StatusPill(model: model)
                DashboardStrip(dashboard: model.dashboard)
            }
            if model.hold.phase == .held { HoldBanner(hold: model.hold) }
            if let approval = model.pendingApproval {
                // The card in place of the caption lines: the spoken question says what the card's headline says.
                ApprovalCard(approval: approval, waitingBehind: model.approvals.waiting.count - 1,
                             previewMaxHeight: 180) { model.answer($0, to: approval.id) }
            } else {
                if let partial = model.partialTranscript, !partial.isEmpty {
                    Text(partial).italic().foregroundStyle(.secondary).lineLimit(2)
                } else if let said = model.entries.last(where: { $0.role == .user }) {
                    Text(said.text).foregroundStyle(.secondary).lineLimit(2)
                }
                if let reply = model.entries.last, reply.role == .agent {
                    Text(Caption.attributed(reply.text)).lineLimit(5)
                }
                if let label = model.toolLabel { ToolActivityRow(label: label) }
            }
        }
        .padding(14)
        .frame(width: 420, alignment: .leading)
    }
}
