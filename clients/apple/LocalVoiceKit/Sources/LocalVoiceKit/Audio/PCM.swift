import AVFoundation
import Foundation

/// Monotonic nanoseconds. Every timing decision in the client (mic gate, played_ms, latency) takes the time as a value,
/// so tests can drive it exactly.
public typealias Nanos = UInt64

public enum MonotonicClock {
    /// Uptime that keeps counting while the process sleeps on a timer, but not while the device sleeps.
    public static func now() -> Nanos { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

extension Nanos {
    public static func ms(_ value: Int) -> Nanos { Nanos(value) * 1_000_000 }
    public var milliseconds: Double { Double(self) / 1_000_000 }
}

extension Duration {
    public var nanos: Nanos {
        let (seconds, attoseconds) = components
        return Nanos(seconds) * 1_000_000_000 + Nanos(attoseconds / 1_000_000_000)
    }

    public var milliseconds: Int { Int(nanos / 1_000_000) }
}

/// PCM signed 16-bit little-endian mono, the only sample format on the wire.
public enum PCM16 {
    public static func data(_ samples: some Collection<Int16>) -> Data {
        var data = Data(capacity: samples.count * 2)
        for s in samples {
            let le = s.littleEndian
            withUnsafeBytes(of: le) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Samples from wire bytes. A trailing odd byte is dropped, as the server's serializer does.
    public static func samples(_ data: Data) -> [Int16] {
        let count = data.count / 2
        var out = [Int16](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                out[i] = Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))
            }
        }
        return out
    }

    public static func silence(samples: Int) -> Data { Data(count: samples * 2) }

    /// Wire bytes at `sampleRate` as a Float32 mono buffer for an `AVAudioPlayerNode` connected at that rate.
    public static func floatBuffer(_ data: Data, sampleRate: Double) -> AVAudioPCMBuffer? {
        let count = data.count / 2
        guard count > 0,
              let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)),
              let out = buffer.floatChannelData?[0]
        else { return nil }
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                out[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))) / 32768
            }
        }
        buffer.frameLength = AVAudioFrameCount(count)
        return buffer
    }

    /// RMS level in dBFS (full-scale sine is about -3 dBFS; digital silence is -inf, reported as -120).
    public static func rmsDBFS(_ samples: some Collection<Int16>) -> Double {
        guard !samples.isEmpty else { return -120 }
        var sum = 0.0
        for s in samples {
            let x = Double(s) / 32768
            sum += x * x
        }
        let rms = (sum / Double(samples.count)).squareRoot()
        return rms > 0 ? max(-120, 20 * log10(rms)) : -120
    }

    public static func rmsDBFS(_ data: Data) -> Double { rmsDBFS(samples(data)) }
}

/// Cuts a stream of 16 kHz samples into fixed protocol frames (20 ms, 640 bytes by default).
public struct PCMFramer: Sendable {
    public let frameSamples: Int
    private var pending: [Int16] = []

    public init(frameSamples: Int = ProtocolV1.captureFrameSamples) {
        precondition(frameSamples > 0 && frameSamples * 2 <= ProtocolV1.maxBinaryMessageBytes)
        self.frameSamples = frameSamples
    }

    public var pendingSamples: Int { pending.count }

    /// Appends samples and returns every complete frame.
    public mutating func append(_ samples: some Collection<Int16>) -> [Data] {
        pending.append(contentsOf: samples)
        guard pending.count >= frameSamples else { return [] }
        var frames: [Data] = []
        var start = 0
        while pending.count - start >= frameSamples {
            frames.append(PCM16.data(pending[start..<(start + frameSamples)]))
            start += frameSamples
        }
        pending.removeFirst(start)
        return frames
    }

    /// The partial frame, padded with zeros to a full frame; `nil` when nothing is pending.
    public mutating func flushPadded() -> Data? {
        guard !pending.isEmpty else { return nil }
        let frame = pending + [Int16](repeating: 0, count: frameSamples - pending.count)
        pending.removeAll(keepingCapacity: true)
        return PCM16.data(frame)
    }

    public mutating func reset() { pending.removeAll(keepingCapacity: true) }

    /// Whole frames of digital silence covering at least `duration` (the push-to-talk tail).
    public func silenceFrames(covering duration: Duration, sampleRate: Int = ProtocolV1.captureSampleRate) -> [Data] {
        let samples = Int((Double(duration.nanos) / 1e9 * Double(sampleRate)).rounded(.up))
        let count = (samples + frameSamples - 1) / frameSamples
        return Array(repeating: PCM16.silence(samples: frameSamples), count: count)
    }
}
