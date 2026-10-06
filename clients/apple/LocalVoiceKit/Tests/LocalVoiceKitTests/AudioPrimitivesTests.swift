import AVFoundation
import Foundation
import Testing
@testable import LocalVoiceKit

/// Signal helpers with machine ground truth: planted tones whose frequency, level and length are known.
enum Signal {
    static func sine(frequency: Double, sampleRate: Double, seconds: Double, amplitude: Float = 0.5) -> [Float] {
        let n = Int(sampleRate * seconds)
        return (0..<n).map { amplitude * Float(sin(2 * Double.pi * frequency * Double($0) / sampleRate)) }
    }

    /// Frequency from positive-going zero crossings, skipping `skip` samples of resampler warm-up at each end.
    static func frequency(_ samples: [Int16], sampleRate: Double, skip: Int = 200) -> Double {
        let s = Array(samples.dropFirst(skip).dropLast(skip))
        guard s.count > 2 else { return 0 }
        var crossings: [Int] = []
        for i in 1..<s.count where s[i - 1] < 0 && s[i] >= 0 { crossings.append(i) }
        guard crossings.count > 2 else { return 0 }
        let periods = Double(crossings.count - 1)
        return periods / (Double(crossings.last! - crossings.first!) / sampleRate)
    }

    static func maxStep(_ samples: [Int16]) -> Int {
        zip(samples, samples.dropFirst()).map { abs(Int($0) - Int($1)) }.max() ?? 0
    }

    static func buffer(_ samples: [Float], sampleRate: Double, channels: AVAudioChannelCount = 1,
                       otherChannels: (Int) -> Float = { _ in 0 }) -> AVAudioPCMBuffer {
        let buffer = AVAudioPCMBuffer(pcmFormat: format(sampleRate: sampleRate, channels: channels), frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for i in samples.indices {
            buffer.floatChannelData![0][i] = samples[i]
            for c in 1..<Int(channels) { buffer.floatChannelData![c][i] = otherChannels(i) }
        }
        return buffer
    }

    /// Deinterleaved Float32, the engine's own format. Above two channels AVAudioFormat needs a layout (as a
    /// voice-processing input on macOS reports one).
    static func format(sampleRate: Double, channels: AVAudioChannelCount) -> AVAudioFormat {
        if channels <= 2 {
            return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: channels,
                                 interleaved: false)!
        }
        let layout = AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | channels)!
        return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, interleaved: false,
                             channelLayout: layout)
    }

    /// Splits `samples` into chunks of irregular sizes, as a tap or a jittery source would deliver them.
    static func chunks(_ samples: [Float], sizes: [Int]) -> [[Float]] {
        var out: [[Float]] = []
        var i = 0
        var k = 0
        while i < samples.count {
            let n = min(sizes[k % sizes.count], samples.count - i)
            out.append(Array(samples[i..<(i + n)]))
            i += n
            k += 1
        }
        return out
    }
}

@Suite("PCM and framing")
struct PCMTests {
    @Test func littleEndianRoundTrip() {
        let samples: [Int16] = [0, 1, -1, 32767, -32768, 1234]
        let data = PCM16.data(samples)
        #expect(data.count == 12)
        #expect(Array(data.prefix(4)) == [0x00, 0x00, 0x01, 0x00])
        #expect(PCM16.samples(data) == samples)
        #expect(PCM16.samples(data + Data([0x7f])) == samples, "a trailing odd byte is dropped")
    }

    @Test func floatBufferScales() throws {
        let buffer = try #require(PCM16.floatBuffer(PCM16.data([32767, -32768, 0, 16384]), sampleRate: 24_000))
        #expect(buffer.format.sampleRate == 24_000)
        #expect(buffer.frameLength == 4)
        let p = buffer.floatChannelData![0]
        #expect(abs(p[0] - 0.99997) < 0.0001)
        #expect(p[1] == -1)
        #expect(p[3] == 0.5)
    }

    @Test func levels() {
        let full = Signal.sine(frequency: 1000, sampleRate: 16_000, seconds: 0.5, amplitude: 1.0)
            .map { Int16(clamping: Int(($0 * 32767).rounded())) }
        #expect(abs(PCM16.rmsDBFS(full) - (-3.01)) < 0.05)
        let tenth = Signal.sine(frequency: 1000, sampleRate: 16_000, seconds: 0.5, amplitude: 0.1)
            .map { Int16(clamping: Int(($0 * 32767).rounded())) }
        #expect(abs(PCM16.rmsDBFS(tenth) - (-23.01)) < 0.05)
        #expect(PCM16.rmsDBFS([Int16](repeating: 0, count: 320)) == -120)
    }

    @Test func framerCutsTwentyMillisecondFrames() {
        var framer = PCMFramer()
        #expect(framer.frameSamples == 320)
        let frames = framer.append([Int16](repeating: 7, count: 1000))
        #expect(frames.count == 3)
        #expect(frames.allSatisfy { $0.count == 640 })
        #expect(framer.pendingSamples == 40)
        let last = framer.flushPadded()
        #expect(last?.count == 640)
        #expect(PCM16.samples(last!).prefix(40).allSatisfy { $0 == 7 })
        #expect(PCM16.samples(last!).suffix(280).allSatisfy { $0 == 0 })
        #expect(framer.flushPadded() == nil)
    }

    @Test func pushToTalkTailIsAtLeastOneHundredMilliseconds() {
        let framer = PCMFramer()
        let tail = framer.silenceFrames(covering: ProtocolV1.pushToTalkTail)
        #expect(tail.count == 5)
        #expect(tail.allSatisfy { $0.count == 640 && $0.allSatisfy { $0 == 0 } })
        #expect(framer.silenceFrames(covering: .milliseconds(90)).count == 5, "rounded up to whole frames")
    }

    @Test func framesStayUnderTheBinaryLimit() {
        #expect(PCMFramer(frameSamples: 640).frameSamples * 2 <= ProtocolV1.maxBinaryMessageBytes)
    }
}

@Suite("Capture conversion to 16 kHz")
struct CaptureConverterTests {
    @Test("a planted tone keeps its pitch, level and length", arguments: [48_000.0, 44_100.0, 24_000.0, 16_000.0])
    func plantedTone(rate: Double) throws {
        let converter = try CaptureConverter(inputFormat: AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!)
        let tone = Signal.sine(frequency: 1000, sampleRate: rate, seconds: 2.0, amplitude: 0.5)
        var out: [Int16] = []
        // Irregular chunk sizes, like a tap (4800 at 48 kHz is 100 ms) or a jittery source.
        for chunk in Signal.chunks(tone, sizes: [4800, 1024, 937, 2048, 4096, 480, 1]) {
            out += converter.convert(Signal.buffer(chunk, sampleRate: rate))
        }
        out += converter.flush()
        let expected = 32_000.0
        #expect(abs(Double(out.count) - expected) <= expected * 0.005, "got \(out.count) samples")
        let f = Signal.frequency(out, sampleRate: 16_000)
        #expect(abs(f - 1000) < 5, "pitch \(f) Hz: a wrong rate would shift it")
        let level = PCM16.rmsDBFS(out.dropFirst(400).dropLast(400))
        #expect(abs(level - (-9.03)) < 0.3, "level \(level) dBFS")
        // A 1 kHz sine at amplitude 0.5 moves at most 0.5 * 32768 * 2π * 1000/16000 ≈ 6434 per sample at 16 kHz.
        #expect(Signal.maxStep(out) < 7000, "no clicks at chunk boundaries")
    }

    @Test("only channel 0 of a multichannel voice-processing input is used")
    func channelZero() throws {
        let converter = try CaptureConverter(inputFormat: Signal.format(sampleRate: 48_000, channels: 3))
        let tone = Signal.sine(frequency: 440, sampleRate: 48_000, seconds: 1.0)
        let buffer = Signal.buffer(tone, sampleRate: 48_000, channels: 3) { i in i % 2 == 0 ? 0.9 : -0.9 }
        let out = converter.convert(buffer) + converter.flush()
        #expect(abs(Signal.frequency(out, sampleRate: 16_000) - 440) < 3)
    }

    @Test("Int16 interleaved input (files) converts too")
    func int16Input() throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 22_050, channels: 2, interleaved: true)!
        let converter = try CaptureConverter(inputFormat: format)
        let tone = Signal.sine(frequency: 500, sampleRate: 22_050, seconds: 1.0)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(tone.count))!
        buffer.frameLength = AVAudioFrameCount(tone.count)
        let p = buffer.int16ChannelData![0]
        for i in tone.indices {
            p[2 * i] = Int16(tone[i] * 32767)
            p[2 * i + 1] = 0
        }
        let out = converter.convert(buffer) + converter.flush()
        #expect(abs(Double(out.count) - 16_000) < 100)
        #expect(abs(Signal.frequency(out, sampleRate: 16_000) - 500) < 3)
    }

    @Test("flush ends one stream and the next starts clean")
    func flushResets() throws {
        let converter = try CaptureConverter(inputFormat: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!)
        for _ in 0..<2 {
            let tone = Signal.sine(frequency: 1000, sampleRate: 48_000, seconds: 0.5)
            let out = converter.convert(Signal.buffer(tone, sampleRate: 48_000)) + converter.flush()
            #expect(abs(out.count - 8000) <= 40, "\(out.count)")
        }
    }
}

@Suite("Played-time ledger and mic gate")
struct LedgerAndGateTests {
    @Test func countsCompletedAndPartialBuffers() {
        var ledger = PlaybackLedger()  // 24 kHz: 960 frames = 40 ms
        ledger.scheduled(frames: 960, at: .ms(0))
        ledger.scheduled(frames: 960, at: .ms(1))
        ledger.scheduled(frames: 960, at: .ms(2))
        #expect(ledger.playedMs(at: .ms(20)) == 20)
        #expect(ledger.playedMs(at: .ms(100)) == 40, "the partial is capped at the head buffer")
        ledger.completed(at: .ms(40))
        #expect(ledger.playedMs(at: .ms(50)) == 50)
        ledger.completed(at: .ms(80))
        ledger.completed(at: .ms(120))
        #expect(ledger.isDrained)
        #expect(ledger.playedMs(at: .ms(500)) == 120, "a drained player adds nothing while it waits")
    }

    @Test func starvationRestartsTheClockAtScheduling() {
        var ledger = PlaybackLedger()
        ledger.scheduled(frames: 960, at: .ms(0))
        ledger.completed(at: .ms(40))
        ledger.scheduled(frames: 960, at: .ms(300))  // late network chunk
        #expect(ledger.playedMs(at: .ms(310)) == 50)
    }

    @Test func hardGateSilencesThenOpens() {
        var gate = MicGate(durationMs: 600)
        let loud = PCM16.data([Int16](repeating: 12000, count: 320))
        gate.arm(at: .ms(1000))
        let a = gate.filter(loud, at: .ms(1100))
        #expect(a.gated && a.frame.count == 640 && a.frame.allSatisfy { $0 == 0 })
        let b = gate.filter(loud, at: .ms(1600))
        #expect(!b.gated && b.frame == loud)
        #expect(!gate.isArmed(at: .ms(1700)))
        gate.arm(at: .ms(2000))
        gate.disarm()
        #expect(!gate.filter(loud, at: .ms(2001)).gated, "new audio disarms the gate")
    }

    @Test func twoTierGateLetsLoudSpeechThrough() {
        var gate = MicGate(durationMs: 700, mode: .twoTier, openThresholdDBFS: -38)
        let quiet = PCM16.data([Int16](repeating: 200, count: 320))   // about -44 dBFS: echo tail
        let loud = PCM16.data([Int16](repeating: 3000, count: 320))   // about -21 dBFS: the user
        gate.arm(at: .ms(0))
        #expect(gate.filter(quiet, at: .ms(100)).gated)
        #expect(!gate.filter(loud, at: .ms(200)).gated)
        #expect(!gate.filter(quiet, at: .ms(300)).gated, "once the user is heard the gate stays open")
    }
}
