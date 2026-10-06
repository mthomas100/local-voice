import AVFoundation
import Foundation
import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

/// Everything the iPhone app runs around one session model: hands-free sessions (plain, or as a CallKit call), the
/// Live Activity, the Action-button intent, Push to Talk when built with it, and the unattended test hooks.
@MainActor
@Observable
final class SessionHub {
    let model: VoiceSessionModel
    var callMode: Bool {
        didSet { UserDefaults.standard.set(callMode, forKey: "LVCallMode") }
    }
    var liveActivitiesOn: Bool {
        didSet {
            UserDefaults.standard.set(liveActivitiesOn, forKey: "LVLiveActivity")
            liveActivity.enabled = liveActivitiesOn
            if !liveActivitiesOn { liveActivity.end() }
        }
    }
    private(set) var inCall = false

    @ObservationIgnored let liveActivity = LiveActivityController()
    @ObservationIgnored private var calls: CallController!
    @ObservationIgnored private var reportedConnected = false
    @ObservationIgnored private var launchedOnce = false
    #if PUSH_TO_TALK
    @ObservationIgnored private(set) var pushToTalk: PushToTalkController?
    #endif

    static weak var shared: SessionHub?
    /// The Action-button intent can run before the hub exists on a cold launch; it leaves a note instead.
    static var pendingStart = false

    init() {
        let defaults = UserDefaults.standard
        let settings = AppSettings.load(client: .iphone)
        model = VoiceSessionModel(settings: settings, client: .iphone)
        callMode = defaults.bool(forKey: "LVCallMode")
        liveActivitiesOn = defaults.object(forKey: "LVLiveActivity") == nil || defaults.bool(forKey: "LVLiveActivity")
        calls = CallController(hub: self)
        liveActivity.enabled = liveActivitiesOn
        liveActivity.report = { [weak model] name, fields in
            model?.log?.write([("event", .string(name))] + fields.sorted { $0.key < $1.key }.map { ($0.key, .string($0.value)) })
        }
        model.onChange = { [weak self] m in self?.modelChanged(m) }
        SessionControl.endSession = { [weak self] in await self?.endSession() }
        Self.shared = self
    }

    // MARK: Lifecycle

    func launched() {
        guard !launchedOnce else { return }
        launchedOnce = true
        model.connect()
        #if PUSH_TO_TALK
        let ptt = PushToTalkController(hub: self)
        pushToTalk = ptt
        Task { await ptt.setUp() }
        #endif
        if Self.pendingStart {
            Self.pendingStart = false
            Task { await startHandsFree() }
        }
        runTestHooks()
    }

    func scenePhaseChanged(_ phase: ScenePhase) {
        switch phase {
        case .active:
            model.connect()
        case .background:
            // Without an open session iOS suspends the app and its socket dies anyway; close it cleanly. A hands-free
            // session or a call keeps running on the `audio` (and `voip`) background modes.
            if !model.handsFree && !inCall && model.talk == .idle {
                Task { await model.disconnect() }
            }
        default:
            break
        }
    }

    // MARK: Hands-free sessions

    func startHandsFree() async {
        if callMode {
            await model.setSessionPolicy(.external)
            do {
                try await calls.startCall()
                inCall = true
                model.log?.write([("event", "callkit"), ("result", "requested")])
            } catch {
                model.log?.write([("event", "callkit"), ("result", "failed"), ("error", .string("\(error)"))])
                await model.setSessionPolicy(.managed)
                await model.setHandsFree(true)
            }
        } else {
            await model.setSessionPolicy(.managed)
            await model.setHandsFree(true)
        }
        if liveActivitiesOn { liveActivity.start(for: model) }
    }

    func endSession() async {
        if inCall {
            await calls.endCall()
        } else {
            await model.setHandsFree(false)
        }
        liveActivity.end()
    }

    func toggleHandsFree() async {
        if model.handsFree || inCall {
            await endSession()
        } else {
            await startHandsFree()
        }
    }

    // MARK: CallKit callbacks

    func callAudioActivated() async {
        model.log?.write([("event", "callkit"), ("result", "audio activated")])
        await model.setHandsFree(true)
    }

    func callAudioDeactivated() async {
        await model.setHandsFree(false)
        await model.stopAudio()
    }

    func callEnded() async {
        inCall = false
        reportedConnected = false
        await model.setHandsFree(false)
        await model.stopAudio()
        liveActivity.end()
        await model.setSessionPolicy(.managed)
    }

    private func modelChanged(_ m: VoiceSessionModel) {
        liveActivity.update(for: m)
        if inCall && m.isReady && !reportedConnected {
            reportedConnected = true
            calls.reportConnected()
        }
    }

    // MARK: Unattended test hooks (launch arguments; never set in normal use)

    private func runTestHooks() {
        let defaults = UserDefaults.standard
        if defaults.bool(forKey: "LVTestLiveActivity") { liveActivity.start(for: model) }
        if defaults.bool(forKey: "LVTestCallKit") {
            Task {
                do {
                    try await calls.startCall()
                    model.log?.write([("event", "callkit"), ("result", "requested")])
                    try? await Task.sleep(for: .seconds(2))
                    await calls.endCall()
                } catch {
                    model.log?.write([("event", "callkit"), ("result", "failed"), ("error", .string("\(error)"))])
                }
            }
        }
        if defaults.bool(forKey: "LVTestIntent") {
            Task {
                // What the Action button does: run the intent's perform() in the app process.
                _ = try? await StartTalkingIntent().perform()
                model.log?.write([("event", "intent"), ("result", "performed"), ("hands_free", .bool(model.handsFree))])
            }
        }
        model.runScriptIfConfigured { [weak self] ok in
            guard let self else { return }
            self.liveActivity.end()
            guard self.model.settings.quitAfterScript else { return }
            await self.model.shutdown()
            exit(ok ? 0 : 1)
        }
    }
}
