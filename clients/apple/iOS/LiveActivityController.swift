import ActivityKit
import Foundation
import LocalVoiceUI

/// Drives the Live Activity from the session model: started with a conversation, updated when the one-line status
/// changes (at most twice a second; ActivityKit throttles updates anyway), ended with the conversation.
@MainActor
final class LiveActivityController {
    private var activity: Activity<VoiceActivityAttributes>?
    private var last: VoiceActivityAttributes.ContentState?
    private var lastUpdate = Date.distantPast
    private var pending: Task<Void, Never>?
    private(set) var updates = 0
    var enabled = true
    var report: (String, [String: String]) -> Void = { _, _ in }

    var isActive: Bool { activity != nil }

    func start(for model: VoiceSessionModel) {
        guard enabled, activity == nil else { return }
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            report("live-activity", ["result": "disabled in Settings"])
            return
        }
        let state = Self.state(for: model)
        do {
            activity = try Activity.request(attributes: VoiceActivityAttributes(device: model.settings.device),
                                            content: .init(state: state, staleDate: nil), pushType: nil)
            last = state
            report("live-activity", ["result": "started", "id": activity?.id ?? ""])
        } catch {
            report("live-activity", ["result": "failed", "error": "\(error)"])
        }
    }

    func update(for model: VoiceSessionModel) {
        guard let activity else { return }
        let state = Self.state(for: model)
        guard state != last else { return }
        last = state
        pending?.cancel()
        let wait = max(0, 0.5 - Date().timeIntervalSince(lastUpdate))
        let handle = ActivityHandle(activity: activity)
        pending = Task { [weak self] in
            if wait > 0 { try? await Task.sleep(for: .seconds(wait)) }
            guard !Task.isCancelled, let self, let latest = self.last else { return }
            await handle.update(latest)
            self.lastUpdate = Date()
            self.updates += 1
            self.report("live-activity", ["result": "updated", "status": latest.status,
                                          "question": latest.question ?? ""])
        }
    }

    func end() {
        guard let activity else { return }
        self.activity = nil
        pending?.cancel()
        let handle = ActivityHandle(activity: activity)
        Task { await handle.end() }
        report("live-activity", ["result": "ended", "updates": "\(updates)"])
    }

    static func state(for model: VoiceSessionModel) -> VoiceActivityAttributes.ContentState {
        let reply = model.entries.last(where: { $0.role == .agent }).map { Caption.plain($0.text) }
        let approval = model.pendingApproval
        return .init(status: model.statusLine, symbol: model.symbolName, lastReply: reply.map { short($0, 120) },
                     handsFree: model.handsFree, whereLine: model.dashboard.whereLine,
                     question: approval.map { short($0.content.headline, 110) }, questionDeadline: approval?.deadline)
    }

    private static func short(_ s: String, _ limit: Int) -> String {
        s.count > limit ? String(s.prefix(limit - 3)) + "…" : s
    }
}

/// `Activity` is not marked Sendable in the iOS 27 SDK, but ActivityKit documents `update` and `end` as callable from
/// any context; this box carries it into those nonisolated async calls (invariant: only `update` and `end` are used).
struct ActivityHandle: @unchecked Sendable {
    let activity: Activity<VoiceActivityAttributes>

    nonisolated func update(_ state: VoiceActivityAttributes.ContentState) async {
        await activity.update(.init(state: state, staleDate: nil))
    }

    nonisolated func end() async {
        await activity.end(nil, dismissalPolicy: .immediate)
    }
}
