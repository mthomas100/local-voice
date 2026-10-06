import AVFoundation
import CallKit
import Foundation
import LocalVoiceUI

/// Hands-free sessions as an outgoing CallKit call (research notes): Lock Screen and AirPods controls, call-priority
/// audio, and the `voip` + `audio` background modes keep the microphone and the socket alive with the screen locked.
///
/// CallKit owns the audio session in this mode: the category is set before the call starts, CallKit activates the
/// session and calls back (`didActivate`), and only then does the engine start (with `SessionPolicy.external`, so it
/// does not activate the session a second time; doing that leaves silent audio or a route stuck off the speaker).
@MainActor
final class CallController: NSObject {
    private let provider: CXProvider
    private let controller = CXCallController()
    private(set) var callID: UUID?
    private weak var hub: SessionHub?

    init(hub: SessionHub) {
        let configuration = CXProviderConfiguration()
        configuration.supportsVideo = false
        configuration.maximumCallGroups = 1
        configuration.maximumCallsPerCallGroup = 1
        configuration.supportedHandleTypes = [.generic]
        configuration.includesCallsInRecents = false
        provider = CXProvider(configuration: configuration)
        self.hub = hub
        super.init()
        provider.setDelegate(self, queue: .main)
    }

    var isInCall: Bool { callID != nil }

    func startCall() async throws {
        guard callID == nil else { return }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
        let id = UUID()
        let action = CXStartCallAction(call: id, handle: CXHandle(type: .generic, value: "Local Voice"))
        action.isVideo = false
        callID = id
        do {
            try await controller.request(CXTransaction(action: action))
        } catch {
            callID = nil
            throw error
        }
    }

    func endCall() async {
        guard let id = callID else { return }
        do {
            try await controller.request(CXTransaction(action: CXEndCallAction(call: id)))
        } catch {
            // The call is already gone (ended from the Lock Screen, or CallKit reset): clean up locally.
            callID = nil
            await hub?.callEnded()
        }
    }

    private func note(_ what: String) {
        hub?.model.log?.write([("event", "callkit"), ("result", .string(what))])
    }

    /// Mark the call connected once the voice server has said welcome.
    func reportConnected() {
        guard let id = callID else { return }
        provider.reportOutgoingCall(with: id, connectedAt: Date())
    }
}

// CallKit calls these on the main queue (`setDelegate(_:queue: .main)`). The actions and the provider are not
// Sendable, so they are answered right here (both are safe to call from the delegate queue) and only the app's own
// state is touched inside `MainActor.assumeIsolated`.
extension CallController: CXProviderDelegate {
    nonisolated func providerDidReset(_ provider: CXProvider) {
        MainActor.assumeIsolated {
            note("provider reset")
            callID = nil
            _ = Task { await hub?.callEnded() }
        }
    }

    nonisolated func provider(_ provider: CXProvider, perform action: CXStartCallAction) {
        provider.reportOutgoingCall(with: action.callUUID, startedConnectingAt: Date())
        action.fulfill()
        MainActor.assumeIsolated { note("start call action fulfilled") }
    }

    nonisolated func provider(_ provider: CXProvider, perform action: CXEndCallAction) {
        action.fulfill()
        MainActor.assumeIsolated {
            note("end call action")
            callID = nil
            _ = Task { await hub?.callEnded() }
        }
    }

    nonisolated func provider(_ provider: CXProvider, perform action: CXSetMutedCallAction) {
        let muted = action.isMuted
        action.fulfill()
        MainActor.assumeIsolated {
            hub?.model.setMuted(muted)
        }
    }

    nonisolated func provider(_ provider: CXProvider, didActivate audioSession: AVAudioSession) {
        MainActor.assumeIsolated {
            note("audio session activated")
            _ = Task { await hub?.callAudioActivated() }
        }
    }

    nonisolated func provider(_ provider: CXProvider, didDeactivate audioSession: AVAudioSession) {
        MainActor.assumeIsolated {
            note("audio session deactivated")
            _ = Task { await hub?.callAudioDeactivated() }
        }
    }
}
