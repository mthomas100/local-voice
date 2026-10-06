// Push to Talk: the system's own talk button on the Lock Screen and anywhere, and agent-initiated speech that wakes
// the app in the background. Built only with LV_PUSH_TO_TALK=YES (project.yml), because it needs what you have
// to create first: an explicit App ID with the Push to Talk and Push Notifications capabilities, and an APNs key (.p8)
// the voice server would use to send PTT pushes. Not exercised yet: the framework does not
// run in the simulator, so this compiles and waits for a device.

#if PUSH_TO_TALK
import AVFoundation
import Foundation
import LocalVoiceUI
import PushToTalk

@MainActor
@Observable
final class PushToTalkController: NSObject {
    /// One fixed channel: "the agent on my Mac".
    static let channelID = UUID(uuidString: "6C6F6361-6C76-6F69-6365-000000000001")!
    @ObservationIgnored private var manager: PTChannelManager?
    @ObservationIgnored private weak var hub: SessionHub?
    private(set) var ephemeralToken: Data?
    private(set) var joined = false

    init(hub: SessionHub) {
        self.hub = hub
    }

    func setUp() async {
        do {
            manager = try await PTChannelManager.channelManager(delegate: self, restorationDelegate: self)
            hub?.model.log?.write([("event", "ptt"), ("result", "manager ready")])
        } catch {
            hub?.model.log?.write([("event", "ptt"), ("result", "unavailable"), ("error", .string("\(error)"))])
        }
    }

    /// Joining needs the app in the foreground and a user action (a button in settings).
    func join() {
        // The framework owns the audio session while joined.
        Task { await hub?.model.setSessionPolicy(.external) }
        manager?.requestJoinChannel(channelUUID: Self.channelID,
                                    descriptor: PTChannelDescriptor(name: "Local Voice", image: nil))
    }

    func leave() {
        manager?.leaveChannel(channelUUID: Self.channelID)
        Task { await hub?.model.setSessionPolicy(.managed) }
    }
}

extension PushToTalkController: PTChannelManagerDelegate {
    nonisolated func channelManager(_ channelManager: PTChannelManager, didJoinChannel channelUUID: UUID,
                                    reason: PTChannelJoinReason) {
        Task { @MainActor in self.joined = true }
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, didLeaveChannel channelUUID: UUID,
                                    reason: PTChannelLeaveReason) {
        Task { @MainActor in self.joined = false }
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, channelUUID: UUID,
                                    didBeginTransmittingFrom source: PTChannelTransmitRequestSource) {
        Task { @MainActor in self.hub?.model.pressTalk() }
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, channelUUID: UUID,
                                    didEndTransmittingFrom source: PTChannelTransmitRequestSource) {
        Task { @MainActor in self.hub?.model.releaseTalk() }
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, receivedEphemeralPushToken pushToken: Data) {
        // The server needs this token to wake the phone for agent-initiated speech; protocol v1 has no message for
        // it yet (an additive `ptt_token` message is the proposal).
        Task { @MainActor in
            self.ephemeralToken = pushToken
            self.hub?.model.log?.write([("event", "ptt"), ("token_bytes", .int(pushToken.count))])
        }
    }

    nonisolated func incomingPushResult(channelManager: PTChannelManager, channelUUID: UUID,
                                        pushPayload: [String: Any]) -> PTPushResult {
        .activeRemoteParticipant(PTParticipant(name: "Agent", image: nil))
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, didActivate audioSession: AVAudioSession) {
        // The framework activated the session (never activate it yourself in this mode).
        Task { @MainActor in self.hub?.model.warmUp() }
    }

    nonisolated func channelManager(_ channelManager: PTChannelManager, didDeactivate audioSession: AVAudioSession) {}
}

extension PushToTalkController: PTChannelRestorationDelegate {
    nonisolated func channelDescriptor(restoredChannelUUID channelUUID: UUID) -> PTChannelDescriptor {
        PTChannelDescriptor(name: "Local Voice", image: nil)
    }
}
#endif
