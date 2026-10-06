import AVFoundation
import Foundation
import Synchronization

/// The device engine: Apple's voice-processing microphone (echo cancellation, noise suppression, AGC), converted to
/// 16 kHz, and reply speech at 24 kHz through the main mixer to the speaker.
///
/// Each `start` builds a fresh `AVAudioEngine` through `VoiceGraph.bringUp`, so voice processing is enabled in the
/// documented order every time, including after a route change or an interruption stopped the old engine.
///
/// Capture can also come from a `SyntheticMicrophone` (no microphone permission is ever requested; for unattended
/// runs), and `outputVolume` 0 keeps such runs silent.
public final class LiveAudioIO: AudioIO, @unchecked Sendable {
    // Invariant for @unchecked Sendable: `running` and `onStopped` are under `lock`; `start`/`stop` are called from one
    // serial queue (VoiceClient's setup queue).
    public enum Capture: Sendable {
        case microphone(voiceProcessing: Bool)
        case synthetic(SyntheticMicrophone)
        case none
    }

    /// Who activates the iOS audio session: this engine, or CallKit / the Push to Talk framework (which activate it
    /// themselves and call back; activating it here too leaves silent audio or a route stuck off the speaker).
    public enum SessionPolicy: Sendable {
        case managed
        case external
    }

    public struct Options: Sendable {
        public var capture: Capture
        public var outputVolume: Float
        /// Requested tap size; AVAudioNode documents 100-400 ms as the supported range, so the system may deliver more.
        public var tapBufferFrames: AVAudioFrameCount
        public var sessionPolicy: SessionPolicy
        /// macOS 14+/iOS 17+: duck other apps only while voice activity is detected, and only a little.
        public var gentleDucking: Bool

        public init(capture: Capture, outputVolume: Float = 1, tapBufferFrames: AVAudioFrameCount = 1024,
                    sessionPolicy: SessionPolicy = .managed, gentleDucking: Bool = true) {
            self.capture = capture
            self.outputVolume = outputVolume
            self.tapBufferFrames = tapBufferFrames
            self.sessionPolicy = sessionPolicy
            self.gentleDucking = gentleDucking
        }
    }

    public let options: Options
    private let driver = AVPlayerDriver()
    private let toneCue = ToneCue()
    public var player: any PlayerDriving { driver }
    public var cue: (any CueDriving)? { toneCue }

    private let lock = NSLock()
    private var running: Running?
    /// Bumped by every start, so a deferred session deactivation never lands on a session a newer start is using.
    private var starts = 0
    private var stoppedHandler: (@Sendable (String) -> Void)?

    public var onStopped: (@Sendable (String) -> Void)? {
        get { lock.withLock { stoppedHandler } }
        set { lock.withLock { stoppedHandler = newValue } }
    }

    private final class Running {
        let steps: EngineSteps
        let description: String
        var observers: [any NSObjectProtocol] = []

        init(steps: EngineSteps, description: String) {
            self.steps = steps
            self.description = description
        }
    }

    public init(options: Options) {
        self.options = options
    }

    private var usesMicrophone: Bool {
        if case .microphone = options.capture { return true }
        return false
    }

    private var voiceProcessing: Bool {
        if case let .microphone(vp) = options.capture { return vp }
        return false
    }

    public func start(capture: @escaping CaptureHandler) throws -> String {
        if let r = lock.withLock({ running }) {
            if r.steps.engine.isRunning { return r.description }
            tearDown(r)  // stopped by a route change or interruption: rebuild in order
        }
        let steps = EngineSteps(options: options, driver: driver, cue: toneCue, capture: capture)
        do {
            try VoiceGraph.bringUp(steps, voiceProcessing: voiceProcessing, microphone: usesMicrophone)
        } catch {
            VoiceGraph.tearDown(steps, voiceProcessing: voiceProcessing, microphone: usesMicrophone)
            throw error
        }
        steps.startPlayers()
        if case let .synthetic(mic) = options.capture { mic.start(capture) }
        let r = Running(steps: steps, description: steps.describe())
        r.observers = observe(steps.engine)
        lock.withLock {
            running = r
            starts += 1
        }
        return r.description
    }

    public func stop() {
        guard let r = lock.withLock({ () -> Running? in
            defer { running = nil }
            return running
        }) else { return }
        tearDown(r)
    }

    private func tearDown(_ r: Running) {
        r.observers.forEach { NotificationCenter.default.removeObserver($0) }
        if case let .synthetic(mic) = options.capture { mic.stop() }
        driver.detach()
        toneCue.detach()
        VoiceGraph.tearDown(r.steps, voiceProcessing: voiceProcessing, microphone: usesMicrophone)
        #if os(iOS)
        if options.sessionPolicy == .managed {
            // Deactivating at once can fail while system components still hold the session; production apps wait
            // about 500 ms (field report, research notes). A start in the meantime keeps the session.
            let startsAtStop = lock.withLock { starts }
            DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(500)) { [weak self] in
                guard let self, self.lock.withLock({ self.starts == startsAtStop && self.running == nil }) else { return }
                try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            }
        }
        #endif
    }

    private func observe(_ engine: AVAudioEngine) -> [any NSObjectProtocol] {
        var observers: [any NSObjectProtocol] = []
        let report: @Sendable (String) -> Void = { [weak self] reason in self?.onStopped?(reason) }
        observers.append(NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { _ in
                report("the audio route or device changed")
            })
        #if os(iOS)
        observers.append(NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: nil) { note in
                let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
                if raw == AVAudioSession.InterruptionType.began.rawValue { report("audio session interrupted") }
            })
        observers.append(NotificationCenter.default.addObserver(
            forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: nil) { _ in
                report("media services were reset")
            })
        #endif
        return observers
    }
}

/// `VoiceGraphSteps` on a fresh `AVAudioEngine`.
final class EngineSteps: VoiceGraphSteps {
    let engine = AVAudioEngine()
    private let options: LiveAudioIO.Options
    private let driver: AVPlayerDriver
    private let cue: ToneCue
    private let capture: CaptureHandler
    private let speech = AVAudioPlayerNode()
    private let cueNode = AVAudioPlayerNode()
    private var tapFormat: AVAudioFormat?
    private var tapInstalled = false
    private(set) var observedTapFrames = 0

    init(options: LiveAudioIO.Options, driver: AVPlayerDriver, cue: ToneCue, capture: @escaping CaptureHandler) {
        self.options = options
        self.driver = driver
        self.cue = cue
        self.capture = capture
    }

    func configureSession() throws {
        #if os(iOS)
        guard options.sessionPolicy == .managed else { return }
        let session = AVAudioSession.sharedInstance()
        if case .microphone = options.capture {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
        } else {
            // No microphone: no record permission is touched (unattended runs in the simulator).
            try session.setCategory(.playback, mode: .spokenAudio)
        }
        try? session.setPreferredIOBufferDuration(0.01)
        try session.setActive(true)
        #endif
    }

    func attachPlaybackGraph() throws {
        engine.attach(speech)
        engine.attach(cueNode)
        try engine.link(speech, to: engine.mainMixerNode, format: driver.format)
        try engine.link(cueNode, to: engine.mainMixerNode, format: cue.format)
        _ = engine.outputNode  // main mixer → output, the echo canceller's reference
        engine.mainMixerNode.outputVolume = options.outputVolume
    }

    func enableVoiceProcessing() throws {
        let input = engine.inputNode
        try input.setVoiceProcessingEnabled(true)
        if options.gentleDucking {
            input.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: true, duckingLevel: .min)
        }
    }

    func installCaptureTap() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)  // read after voice processing changed it
        let converter = try CaptureConverter(inputFormat: format)
        tapFormat = format
        let capture = self.capture
        input.installTap(onBus: 0, bufferSize: options.tapBufferFrames, format: format) { buffer, _ in
            let samples = converter.convert(buffer)
            if !samples.isEmpty { capture(samples, MonotonicClock.now()) }
        }
        tapInstalled = true
    }

    func prepareAndStart() throws {
        engine.prepare()
        try engine.start()
    }

    func startPlayers() {
        speech.play()
        driver.attach(speech, engine: engine, callbackType: .dataPlayedBack)
        cue.attach(cueNode, engine: engine)
    }

    func removeCaptureTap() {
        if tapInstalled { engine.inputNode.removeTap(onBus: 0) }
        tapInstalled = false
    }

    func disableVoiceProcessing() {
        try? engine.inputNode.setVoiceProcessingEnabled(false)
    }

    func stopEngine() {
        engine.stop()
    }

    func describe() -> String {
        let out = engine.outputNode.outputFormat(forBus: 0)
        var parts: [String] = []
        switch options.capture {
        case let .microphone(vp):
            let f = tapFormat
            parts.append("microphone \(Int(f?.sampleRate ?? 0)) Hz x\(f?.channelCount ?? 0)"
                         + (vp ? " with voice processing" : "") + ", tap \(options.tapBufferFrames) -> 16 kHz")
        case let .synthetic(mic):
            parts.append("synthetic microphone \(Int(mic.sampleRate)) Hz, \(mic.chunkFrames)-frame chunks -> 16 kHz")
        case .none:
            parts.append("no capture")
        }
        parts.append("speech 24 kHz -> output \(Int(out.sampleRate)) Hz x\(out.channelCount)"
                     + (options.outputVolume == 0 ? " (muted)" : ""))
        return parts.joined(separator: "; ")
    }
}
