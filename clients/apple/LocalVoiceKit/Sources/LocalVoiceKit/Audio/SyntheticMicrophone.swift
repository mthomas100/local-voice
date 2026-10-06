import AVFoundation
import Foundation
import Synchronization

/// A microphone that hears only what it is told to: silence, then clips (a `say` recording, a planted tone) on demand,
/// delivered in real time at 48 kHz like a hardware tap, and converted to 16 kHz by the same `CaptureConverter` the
/// real microphone uses.
///
/// Unattended runs use it so nothing asks for microphone permission (no system prompt on the user's Mac or in the
/// simulator) and the input is known exactly (machine ground truth).
public final class SyntheticMicrophone: @unchecked Sendable {
    // Invariant for @unchecked Sendable: `converter`, `handler`, `startTime` and `emitted` are only touched on `queue`;
    // the clip queue is under its Mutex.
    public let sampleRate: Double
    public let chunkFrames: Int
    private let queue = DispatchQueue(label: "lv.synthetic-mic", qos: .userInteractive)
    private var timer: DispatchSourceTimer?
    private var converter: CaptureConverter
    private var handler: CaptureHandler?
    private var startTime: Nanos = 0
    private var emitted = 0

    private struct Clip {
        var name: String
        var samples: [Float]
        var offset = 0
    }

    private let clips = Mutex<[Clip]>([])
    private let finished = Mutex<(@Sendable (String) -> Void)?>(nil)

    /// `chunkFrames` 960 is 20 ms at 48 kHz; 4800 imitates the 100 ms buffers a tap often delivers.
    public init(sampleRate: Double = 48_000, chunkFrames: Int = 960) {
        self.sampleRate = sampleRate
        self.chunkFrames = chunkFrames
        self.converter = try! CaptureConverter(
            inputFormat: AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!)
    }

    public var onClipFinished: (@Sendable (String) -> Void)? {
        get { finished.withLock { $0 } }
        set { finished.withLock { $0 = newValue } }
    }

    /// Queues samples (mono, at `sampleRate`) to be "heard" after anything already queued.
    public func play(_ samples: [Float], name: String) {
        clips.withLock { $0.append(Clip(name: name, samples: samples)) }
    }

    public var isPlaying: Bool { clips.withLock { !$0.isEmpty } }

    public func start(_ handler: @escaping CaptureHandler) {
        queue.sync {
            guard timer == nil else {
                self.handler = handler
                return
            }
            self.handler = handler
            startTime = MonotonicClock.now()
            emitted = 0
            let t = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
            let period = Double(chunkFrames) / sampleRate
            t.schedule(deadline: .now(), repeating: .nanoseconds(Int(period * 1e9 / 2)), leeway: .milliseconds(1))
            t.setEventHandler { [weak self] in self?.tick() }
            timer = t
            t.resume()
        }
    }

    public func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
            handler = nil
            _ = converter.flush()
        }
    }

    private func tick() {
        let now = MonotonicClock.now()
        let due = Int(Double(now - startTime) / 1e9 * sampleRate) / chunkFrames
        while emitted < due {
            emitted += 1
            var chunk = [Float](repeating: 0, count: chunkFrames)
            var done: [String] = []
            clips.withLock { queue in
                var filled = 0
                while filled < chunkFrames, !queue.isEmpty {
                    let take = min(chunkFrames - filled, queue[0].samples.count - queue[0].offset)
                    for i in 0..<take { chunk[filled + i] = queue[0].samples[queue[0].offset + i] }
                    queue[0].offset += take
                    filled += take
                    if queue[0].offset >= queue[0].samples.count { done.append(queue.removeFirst().name) }
                }
            }
            let samples = chunk.withUnsafeBufferPointer { converter.convert(monoSamples: $0) }
            if !samples.isEmpty { handler?(samples, now) }
            if !done.isEmpty, let callback = onClipFinished { done.forEach(callback) }
        }
    }

    // MARK: Clips

    /// A planted tone with 10 ms fades (machine ground truth: its pitch and length are known exactly).
    public static func tone(frequency: Double, seconds: Double, amplitude: Float = 0.3,
                            sampleRate: Double = 48_000) -> [Float] {
        let n = Int(seconds * sampleRate)
        let fade = Int(0.01 * sampleRate)
        return (0..<n).map { i in
            let edge = min(1, Float(min(i, n - 1 - i)) / Float(fade))
            return amplitude * edge * Float(sin(2 * Double.pi * frequency * Double(i) / sampleRate))
        }
    }

    public static func silence(seconds: Double, sampleRate: Double = 48_000) -> [Float] {
        [Float](repeating: 0, count: Int(seconds * sampleRate))
    }

    public enum LoadError: Error, CustomStringConvertible {
        case unreadable(String)

        public var description: String {
            switch self {
            case let .unreadable(s): return "cannot read audio: \(s)"
            }
        }
    }

    /// Any file AVAudioFile reads (WAV, AIFF from `say`, CAF), as mono Float32 at `sampleRate`.
    public static func load(_ url: URL, sampleRate: Double = 48_000) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        guard let input = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw LoadError.unreadable(url.path)
        }
        try file.read(into: input)
        guard let monoIn = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inFormat.sampleRate, channels: 1,
                                         interleaved: false),
              let monoOut = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let mono = AVAudioPCMBuffer(pcmFormat: monoIn, frameCapacity: input.frameLength)
        else { throw LoadError.unreadable(url.path) }
        mono.frameLength = input.frameLength
        for i in 0..<Int(input.frameLength) { mono.floatChannelData![0][i] = input.floatChannelData![0][i] }
        if inFormat.sampleRate == sampleRate {
            return Array(UnsafeBufferPointer(start: mono.floatChannelData![0], count: Int(mono.frameLength)))
        }
        guard let converter = AVAudioConverter(from: monoIn, to: monoOut),
              let out = AVAudioPCMBuffer(pcmFormat: monoOut,
                                         frameCapacity: AVAudioFrameCount(Double(mono.frameLength) * sampleRate
                                                                          / inFormat.sampleRate) + 1024)
        else { throw LoadError.unreadable(url.path) }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        var handed = false
        var error: NSError?
        _ = converter.convert(to: out, error: &error) { _, status in
            if handed {
                status.pointee = .endOfStream
                return nil
            }
            handed = true
            status.pointee = .haveData
            return mono
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
    }
}
