import AVFoundation
import Foundation
import LocalVoiceKit

/// Where capture and playback come from.
public enum AudioMode: String, Sendable, CaseIterable {
    /// The real microphone with Apple's voice processing, and the speaker.
    case microphone
    /// A synthetic microphone (scripted clips) and the real output device, muted by default: unattended app runs.
    case synthetic
    /// No device at all (manual rendering at real-time pace).
    case headless
}

/// The app's settings: UserDefaults, which launch arguments override (`-LVServerURL ws://127.0.0.1:18770/v1/voice`),
/// with defaults from the Info.plist keys the project spec sets (`LVDefaultServerHost`, `LVDefaultServerPort`).
public struct AppSettings: Sendable, Equatable {
    public var serverURL: URL
    public var device: String
    public var mic: MicMode
    public var gateMs: Int
    public var gateMode: MicGate.Mode
    public var prerollMs: Int
    public var audio: AudioMode
    public var outputVolume: Float
    /// Unattended test mode: a session script (see `ScriptStep`) run at launch.
    public var script: String?
    /// JSONL event log; a relative path lands in the app's Documents folder.
    public var eventLogPath: String?
    public var quitAfterScript: Bool
    /// Keep the engine (and the microphone) warm this long after the last use; see `ClientSettings`.
    public var engineIdleSeconds: Int
    public var pressChirp: Bool

    public enum Key {
        public static let serverURL = "LVServerURL"
        public static let device = "LVDevice"
        public static let mic = "LVMicMode"
        public static let gateMs = "LVGateMs"
        public static let gateMode = "LVGateMode"
        public static let prerollMs = "LVPrerollMs"
        public static let audio = "LVAudio"
        public static let outputVolume = "LVOutputVolume"
        public static let script = "LVScript"
        public static let eventLog = "LVEventLog"
        public static let quitAfterScript = "LVQuitAfterScript"
        public static let engineIdleSeconds = "LVEngineIdleSeconds"
        public static let pressChirp = "LVPressChirp"
    }

    public static func defaultServerURL(bundle: Bundle = .main) -> URL {
        let host = bundle.object(forInfoDictionaryKey: "LVDefaultServerHost") as? String ?? "127.0.0.1"
        let port = (bundle.object(forInfoDictionaryKey: "LVDefaultServerPort") as? String).flatMap(Int.init)
            ?? ProtocolV1.defaultPort
        return URL(string: "ws://\(host):\(port)\(ProtocolV1.path)")!
    }

    public static func load(client: ClientKind, defaults: UserDefaults = .standard, bundle: Bundle = .main) -> AppSettings {
        let d = defaults
        let audio = d.string(forKey: Key.audio).flatMap(AudioMode.init) ?? .microphone
        // The unattended test hook (a session script from UserDefaults) exists in debug builds only.
        #if DEBUG
        let script = d.string(forKey: Key.script)
        #else
        let script: String? = nil
        #endif
        return AppSettings(
            serverURL: d.string(forKey: Key.serverURL).flatMap(URL.init(string:)) ?? defaultServerURL(bundle: bundle),
            device: d.string(forKey: Key.device) ?? (client == .mac ? "mac" : "iphone"),
            mic: d.string(forKey: Key.mic).flatMap(MicMode.init) ?? .ptt,
            gateMs: d.object(forKey: Key.gateMs) as? Int ?? Int(d.string(forKey: Key.gateMs) ?? "") ?? 600,
            gateMode: d.string(forKey: Key.gateMode).flatMap(MicGate.Mode.init) ?? .hard,
            // The Mac's server is on loopback (no jitter to absorb); the phone is on Wi-Fi (7-227 ms RTT spread
            // measured, research notes). The client adds about this much before the first word (81 ms measured at 100).
            prerollMs: d.object(forKey: Key.prerollMs) as? Int ?? Int(d.string(forKey: Key.prerollMs) ?? "")
                ?? (client == .mac ? 60 : 100),
            audio: audio,
            // Unattended runs are silent unless asked otherwise: someone may be asleep next to the speaker.
            outputVolume: Float(d.string(forKey: Key.outputVolume) ?? "") ?? (audio == .microphone ? 1 : 0),
            script: script,
            eventLogPath: d.string(forKey: Key.eventLog),
            quitAfterScript: d.bool(forKey: Key.quitAfterScript),
            // The Mac is talked to in bursts from the hotkey; the phone more often hands-free (which never idles out).
            engineIdleSeconds: Int(d.string(forKey: Key.engineIdleSeconds) ?? "") ?? (client == .mac ? 60 : 30),
            pressChirp: d.object(forKey: Key.pressChirp) == nil ? audio == .microphone : d.bool(forKey: Key.pressChirp))
    }

    /// Persists what the settings screen edits (launch arguments still win while present).
    public func save(to defaults: UserDefaults = .standard) {
        defaults.set(serverURL.absoluteString, forKey: Key.serverURL)
        defaults.set(device, forKey: Key.device)
        defaults.set(mic.rawValue, forKey: Key.mic)
        defaults.set(gateMs, forKey: Key.gateMs)
        defaults.set(gateMode.rawValue, forKey: Key.gateMode)
        defaults.set(prerollMs, forKey: Key.prerollMs)
    }

    public var clientSettings: ClientSettings {
        ClientSettings(mic: mic, gate: MicGate(durationMs: gateMs, mode: gateMode), prerollMs: prerollMs,
                       engineIdleStopSeconds: engineIdleSeconds > 0 ? engineIdleSeconds : nil, pressChirp: pressChirp)
    }

    public var eventLogURL: URL? {
        guard let path = eventLogPath, !path.isEmpty else { return nil }
        if path.hasPrefix("/") { return URL(fileURLWithPath: path) }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent(path)
    }
}

public enum MicrophonePermission {
    /// Reading the status never prompts.
    public static var isGranted: Bool {
        #if os(iOS)
        return AVAudioApplication.shared.recordPermission == .granted
        #else
        return AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        #endif
    }

    /// Asks once (the system shows its prompt the first time); never called in synthetic or headless mode.
    public static func request() async -> Bool {
        #if os(iOS)
        return await AVAudioApplication.requestRecordPermission()
        #else
        return await AVCaptureDevice.requestAccess(for: .audio)
        #endif
    }
}
