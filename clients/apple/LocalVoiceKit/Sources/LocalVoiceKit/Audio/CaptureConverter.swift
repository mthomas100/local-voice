import AVFoundation
import Foundation

/// Converts capture buffers in the hardware format to 16 kHz Int16 mono (PROTOCOL.md "Audio").
///
/// The hardware is never asked for 16 kHz: the input node runs at its own rate (usually 48 kHz, 44.1 kHz on some Macs
/// and Bluetooth routes) and this converts with `AVAudioConverter`. With voice processing on, macOS can present the
/// input node with several channels; channel 0 carries the processed voice, so only it is used.
///
/// One converter instance is one continuous stream: its filter state carries across calls, so chunk boundaries do not
/// click. The input block hands each buffer over once and then answers `.noDataNow`; answering `.endOfStream` would end
/// the stream and the next call would produce nothing. `flush()` ends a stream deliberately (end of a push-to-talk
/// turn) and resets it for the next one.
///
/// Not thread-safe: use it from one thread or queue at a time (the capture thread).
public final class CaptureConverter {
    public let inputFormat: AVAudioFormat
    public let outputFormat: AVAudioFormat
    private let monoFormat: AVAudioFormat
    private let converter: AVAudioConverter
    private var mono: AVAudioPCMBuffer
    private var output: AVAudioPCMBuffer

    public enum ConverterError: Error, CustomStringConvertible {
        case unsupportedFormat(String)
        case noConverter

        public var description: String {
            switch self {
            case let .unsupportedFormat(f): return "unsupported capture format \(f)"
            case .noConverter: return "AVAudioConverter could not be created"
            }
        }
    }

    public init(inputFormat: AVAudioFormat, outputSampleRate: Double = Double(ProtocolV1.captureSampleRate)) throws {
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              inputFormat.commonFormat == .pcmFormatFloat32 || inputFormat.commonFormat == .pcmFormatInt16
        else { throw ConverterError.unsupportedFormat("\(inputFormat)") }
        guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputFormat.sampleRate, channels: 1,
                                       interleaved: false),
              let out = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: outputSampleRate, channels: 1,
                                      interleaved: true)
        else { throw ConverterError.unsupportedFormat("\(inputFormat)") }
        guard let converter = AVAudioConverter(from: mono, to: out) else { throw ConverterError.noConverter }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        self.inputFormat = inputFormat
        self.monoFormat = mono
        self.outputFormat = out
        self.converter = converter
        self.mono = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: 4800)!
        self.output = AVAudioPCMBuffer(pcmFormat: out, frameCapacity: 1600)!
    }

    /// Converts one capture buffer (in `inputFormat`) and returns the 16 kHz samples it produced.
    public func convert(_ buffer: AVAudioPCMBuffer) -> [Int16] {
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return [] }
        ensureCapacity(frames)
        copyChannelZero(of: buffer, frames: frames)
        return run(endOfStream: false)
    }

    /// Converts mono Float32 samples at the input rate (synthetic and file-fed capture).
    public func convert(monoSamples samples: UnsafeBufferPointer<Float>) -> [Int16] {
        guard !samples.isEmpty else { return [] }
        ensureCapacity(samples.count)
        let dst = mono.floatChannelData![0]
        dst.update(from: samples.baseAddress!, count: samples.count)
        mono.frameLength = AVAudioFrameCount(samples.count)
        return run(endOfStream: false)
    }

    /// Drains the resampler's filter tail and resets it, so the next turn starts clean.
    public func flush() -> [Int16] {
        mono.frameLength = 0
        let tail = run(endOfStream: true)
        converter.reset()
        return tail
    }

    private func ensureCapacity(_ frames: Int) {
        if frames > Int(mono.frameCapacity) {
            mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: AVAudioFrameCount(frames))!
        }
        let needed = Int((Double(frames) * outputFormat.sampleRate / inputFormat.sampleRate).rounded(.up)) + 256
        if needed > Int(output.frameCapacity) {
            output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(needed))!
        }
    }

    private func copyChannelZero(of buffer: AVAudioPCMBuffer, frames: Int) {
        let dst = mono.floatChannelData![0]
        let channels = Int(buffer.format.channelCount)
        if let src = buffer.floatChannelData {
            if buffer.format.isInterleaved && channels > 1 {
                for i in 0..<frames { dst[i] = src[0][i * channels] }
            } else {
                dst.update(from: src[0], count: frames)
            }
        } else if let src = buffer.int16ChannelData {
            let stride = buffer.format.isInterleaved ? channels : 1
            for i in 0..<frames { dst[i] = Float(src[0][i * stride]) / 32768 }
        }
        mono.frameLength = AVAudioFrameCount(frames)
    }

    private func run(endOfStream: Bool) -> [Int16] {
        var collected: [Int16] = []
        var handedOver = false
        let input = mono
        // Loop because one call may not drain everything when the output buffer fills.
        while true {
            output.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, inputStatus in
                if handedOver || input.frameLength == 0 {
                    inputStatus.pointee = endOfStream ? .endOfStream : .noDataNow
                    return nil
                }
                handedOver = true
                inputStatus.pointee = .haveData
                return input
            }
            let produced = Int(output.frameLength)
            if produced > 0, let samples = output.int16ChannelData?[0] {
                collected.append(contentsOf: UnsafeBufferPointer(start: samples, count: produced))
            }
            switch status {
            case .haveData where produced == Int(output.frameCapacity):
                continue  // the output filled up; there may be more
            case .error:
                return collected
            default:
                return collected
            }
        }
    }
}
