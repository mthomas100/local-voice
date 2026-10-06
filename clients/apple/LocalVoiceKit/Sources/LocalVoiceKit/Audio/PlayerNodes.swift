import AVFoundation
import Foundation
import Synchronization

/// `PlayerDriving` on an `AVAudioPlayerNode` connected at 24 kHz mono; the engine's main mixer converts to the output
/// rate. The node is swapped when the engine is rebuilt (each start builds a fresh graph so voice processing is always
/// enabled in the documented order).
public final class AVPlayerDriver: PlayerDriving, @unchecked Sendable {
    // Invariant for @unchecked Sendable: `attached` is only read and written under its Mutex.
    private struct Attached {
        var node: AVAudioPlayerNode
        var engine: AVAudioEngine
        var callbackType: AVAudioPlayerNodeCompletionCallbackType
    }

    private let attached = Mutex<Attached?>(nil)
    public let sampleRate: Double

    public init(sampleRate: Double = Double(ProtocolV1.playbackSampleRate)) {
        self.sampleRate = sampleRate
    }

    public var format: AVAudioFormat { AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)! }

    /// `.dataPlayedBack` on a device (the count then includes the output latency); `.dataRendered` in manual rendering,
    /// where "played back" never happens.
    func attach(_ node: AVAudioPlayerNode, engine: AVAudioEngine, callbackType: AVAudioPlayerNodeCompletionCallbackType) {
        attached.withLock { $0 = Attached(node: node, engine: engine, callbackType: callbackType) }
    }

    func detach() {
        let old = attached.withLock { a -> Attached? in
            defer { a = nil }
            return a
        }
        old?.node.stop()
    }

    @discardableResult
    public func schedule(_ pcm16: Data, completion: @escaping @Sendable () -> Void) -> Int {
        guard let a = attached.withLock({ $0 }), let buffer = PCM16.floatBuffer(pcm16, sampleRate: sampleRate) else {
            completion()
            return 0
        }
        a.node.scheduleBuffer(buffer, completionCallbackType: a.callbackType) { _ in completion() }
        return Int(buffer.frameLength)
    }

    public func flush() {
        guard let a = attached.withLock({ $0 }) else { return }
        // stop() drops every scheduled buffer and calls their handlers; play() again so new buffers start at once.
        a.node.stop()
        if a.engine.isRunning { a.node.play() }
    }
}

/// The working sound while a tool runs: a soft two-note blip every 1.4 s on its own player node, mixed under speech
/// by the same engine (so the voice-processing echo canceller sees it too).
public final class ToneCue: CueDriving, @unchecked Sendable {
    // Invariant for @unchecked Sendable: all state is under the Mutex.
    private struct State {
        var node: AVAudioPlayerNode?
        var engine: AVAudioEngine?
        var active = false
    }

    private let state = Mutex(State())
    public let volume: Float
    let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!

    public init(volume: Float = 0.12) {
        self.volume = volume
    }

    func attach(_ node: AVAudioPlayerNode, engine: AVAudioEngine) {
        node.volume = volume
        let wasActive = state.withLock { s -> Bool in
            s.node = node
            s.engine = engine
            return s.active
        }
        if wasActive { startLoop(node, engine: engine) }
    }

    func detach() {
        let node = state.withLock { s -> AVAudioPlayerNode? in
            defer {
                s.node = nil
                s.engine = nil
            }
            return s.node
        }
        node?.stop()
    }

    public func setActive(_ active: Bool) {
        let (node, engine, changed) = state.withLock { s -> (AVAudioPlayerNode?, AVAudioEngine?, Bool) in
            defer { s.active = active }
            return (s.node, s.engine, s.active != active)
        }
        guard changed, let node, let engine else { return }
        if active {
            startLoop(node, engine: engine)
        } else {
            node.stop()
        }
    }

    public func chirp() {
        let (node, engine, looping) = state.withLock { s in (s.node, s.engine, s.active) }
        // The working loop already says "busy"; a chirp over it would only muddle it.
        guard !looping, let node, let engine, engine.isRunning, let blip = makeChirp() else { return }
        node.scheduleBuffer(blip, at: nil, options: .interrupts)
        if !node.isPlaying { node.play() }
    }

    private func makeChirp() -> AVAudioPCMBuffer? {
        let rate = format.sampleRate
        let frames = AVAudioFrameCount(rate * 0.06)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        buffer.frameLength = frames
        let out = buffer.floatChannelData![0]
        let n = Int(frames)
        for i in 0..<n {
            let t = Double(i) / rate
            let envelope = sin(Double.pi * Double(i) / Double(n))
            out[i] = Float(0.6 * envelope * sin(2 * Double.pi * (880 + 4000 * t) * t))  // a short rising blip
        }
        return buffer
    }

    private func startLoop(_ node: AVAudioPlayerNode, engine: AVAudioEngine) {
        guard engine.isRunning, let loop = makeLoop() else { return }
        node.stop()
        node.scheduleBuffer(loop, at: nil, options: .loops)
        node.play()
    }

    private func makeLoop() -> AVAudioPCMBuffer? {
        let rate = format.sampleRate
        let frames = AVAudioFrameCount(rate * 1.4)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        buffer.frameLength = frames
        let out = buffer.floatChannelData![0]
        for i in 0..<Int(frames) { out[i] = 0 }
        for (start, frequency) in [(0.0, 659.25), (0.11, 783.99)] {
            let first = Int(start * rate)
            let length = Int(0.07 * rate)
            for i in 0..<length {
                let envelope = sin(Double.pi * Double(i) / Double(length))  // no clicks at either end
                out[first + i] += Float(0.5 * envelope * sin(2 * Double.pi * frequency * Double(i) / rate))
            }
        }
        return buffer
    }
}
