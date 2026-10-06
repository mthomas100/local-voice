import AVFoundation
import Foundation
import Synchronization

/// What the "speaker" of a headless engine played: when it was audible, and optionally a WAV of it.
///
/// The flush check of the e2e tests reads it: after `interrupt`, the output must go silent within one render quantum,
/// the local flush happening before any message goes back (PROTOCOL.md "Barge-in").
public final class OutputMonitor: @unchecked Sendable {
    // Invariant for @unchecked Sendable: all state under `lock`; `observe` is called from the render queue only.
    public struct Interval: Sendable, Equatable {
        public var start: Nanos
        public var end: Nanos
    }

    private let lock = NSLock()
    private var intervals: [Interval] = []
    private var current: Interval?
    private var file: AVAudioFile?
    private var renderedFrames = 0
    private var audibleFrames = 0
    public let thresholdDBFS: Double

    public init(recordTo url: URL? = nil, format: AVAudioFormat? = nil, thresholdDBFS: Double = -55) {
        self.thresholdDBFS = thresholdDBFS
        if let url, let format {
            file = try? AVAudioFile(forWriting: url, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: format.sampleRate,
                AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
            ])
        }
    }

    func observe(_ buffer: AVAudioPCMBuffer, at now: Nanos) {
        let n = Int(buffer.frameLength)
        guard n > 0, let p = buffer.floatChannelData?[0] else { return }
        var sum: Float = 0
        for i in 0..<n { sum += p[i] * p[i] }
        let rms = (sum / Float(n)).squareRoot()
        let db = rms > 0 ? 20 * log10(Double(rms)) : -120
        lock.lock()
        defer { lock.unlock() }
        renderedFrames += n
        if db > thresholdDBFS {
            audibleFrames += n
            if current == nil { current = Interval(start: now, end: now) }
            current?.end = now
        } else if let c = current {
            intervals.append(c)
            current = nil
        }
        if let file, let mono = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate:
            buffer.format.sampleRate, channels: 1)!, frameCapacity: buffer.frameLength) {
            mono.frameLength = buffer.frameLength
            mono.floatChannelData![0].update(from: p, count: n)
            try? file.write(from: mono)
        }
    }

    /// Spans of audible output so far (the open one included).
    public var audible: [Interval] {
        lock.withLock { intervals + (current.map { [$0] } ?? []) }
    }

    public var audibleSeconds: Double {
        lock.withLock { Double(audibleFrames) / 48_000 }
    }

    /// The last moment anything audible was rendered.
    public var lastAudibleAt: Nanos? { audible.last?.end }

    public func close() {
        lock.withLock { file = nil }
    }
}

/// No device at all: a synthetic microphone, and an engine in offline manual rendering mode pulled at real-time pace
/// by a timer, so the real `AVAudioPlayerNode` scheduling, the mixer's 24 to 48 kHz conversion and the played-time
/// accounting all run, silently, in tests and on a headless Mac. Voice processing does not exist here (it needs a
/// device), which is why the voice-processing order is tested through `VoiceGraph` instead.
public final class HeadlessAudioIO: AudioIO, @unchecked Sendable {
    // Invariant for @unchecked Sendable: engine state is touched on `renderQueue` (rendering) and from the caller's
    // setup queue (start/stop) with the render timer cancelled synchronously in between.
    public let microphone: SyntheticMicrophone?
    public let monitor: OutputMonitor
    public let outputRate: Double = 48_000
    private let driver = AVPlayerDriver()
    private let toneCue = ToneCue()
    public var player: any PlayerDriving { driver }
    public var cue: (any CueDriving)? { toneCue }
    public var onStopped: (@Sendable (String) -> Void)?

    private let renderQueue = DispatchQueue(label: "lv.headless.render", qos: .userInteractive)
    private var engine: AVAudioEngine?
    private var timer: DispatchSourceTimer?
    private var renderBuffer: AVAudioPCMBuffer?
    private var startTime: Nanos = 0
    private var renderedFrames = 0

    public init(microphone: SyntheticMicrophone?, recordOutputTo url: URL? = nil) {
        self.microphone = microphone
        self.monitor = OutputMonitor(recordTo: url,
                                     format: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1))
    }

    public func start(capture: @escaping CaptureHandler) throws -> String {
        if engine != nil { return describe() }
        let engine = AVAudioEngine()
        let speech = AVAudioPlayerNode()
        let cueNode = AVAudioPlayerNode()
        engine.attach(speech)
        engine.attach(cueNode)
        let outFormat = AVAudioFormat(standardFormatWithSampleRate: outputRate, channels: 2)!
        try engine.enableManualRenderingMode(.offline, format: outFormat, maximumFrameCount: 4096)
        try engine.link(speech, to: engine.mainMixerNode, format: driver.format)
        try engine.link(cueNode, to: engine.mainMixerNode, format: toneCue.format)
        engine.prepare()
        try engine.start()
        speech.play()
        driver.attach(speech, engine: engine, callbackType: .dataRendered)
        toneCue.attach(cueNode, engine: engine)
        renderBuffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4096)
        self.engine = engine
        startTime = MonotonicClock.now()
        renderedFrames = 0
        let t = DispatchSource.makeTimerSource(flags: .strict, queue: renderQueue)
        t.schedule(deadline: .now(), repeating: .milliseconds(5), leeway: .milliseconds(1))
        t.setEventHandler { [weak self] in self?.renderDue() }
        timer = t
        t.resume()
        microphone?.start(capture)
        return describe()
    }

    public func stop() {
        microphone?.stop()
        renderQueue.sync {
            timer?.cancel()
            timer = nil
        }
        driver.detach()
        toneCue.detach()
        engine?.stop()
        engine = nil
    }

    /// Renders whatever real time says is due, in 10 ms quanta.
    private func renderDue() {
        guard let engine, let buffer = renderBuffer else { return }
        let now = MonotonicClock.now()
        var due = Int(Double(now - startTime) / 1e9 * outputRate) - renderedFrames
        while due >= 480 {
            do {
                let status = try engine.renderOffline(480, to: buffer)
                guard status == .success else { break }
            } catch {
                break
            }
            monitor.observe(buffer, at: now)
            renderedFrames += 480
            due -= 480
        }
    }

    private func describe() -> String {
        var parts = [microphone.map { "synthetic microphone \(Int($0.sampleRate)) Hz, \($0.chunkFrames)-frame chunks -> 16 kHz" }
            ?? "no capture"]
        parts.append("speech 24 kHz -> headless output \(Int(outputRate)) Hz (manual rendering, real-time pace)")
        return parts.joined(separator: "; ")
    }
}
