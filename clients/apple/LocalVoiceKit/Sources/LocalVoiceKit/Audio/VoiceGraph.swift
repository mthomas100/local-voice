import AVFoundation
import Foundation

/// The steps of bringing up an engine with Apple's voice processing, as one seam so the order is written once and
/// tested (definition of done M2.3).
protocol VoiceGraphSteps: AnyObject {
    func configureSession() throws
    func attachPlaybackGraph() throws
    func enableVoiceProcessing() throws
    func installCaptureTap() throws
    func prepareAndStart() throws
    func removeCaptureTap()
    func disableVoiceProcessing()
    func stopEngine()
}

/// The one order that works (the project's research notes; Barock, "Why your iOS voice agent still hears itself", 2026-04-22;
/// Apple forum 97679):
///
/// 1. iOS session `.playAndRecord` + `.voiceChat` (`.default` with `.defaultToSpeaker` silently defeats echo
///    cancellation).
/// 2. Attach the playback graph (player → main mixer → output) *before* enabling voice processing: the canceller takes
///    its reference from the output bus, and enabling it first leaves that reference empty, so it cancels nothing,
///    silently.
/// 3. Enable voice processing on the input node while the engine is stopped (AVAudioIONode.h: it can only change then).
/// 4. Install the capture tap in the format the input node reports *after* step 3.
/// 5. `prepare()`, `start()`.
///
/// Teardown: remove the tap, then `setVoiceProcessingEnabled(false)` *before* `engine.stop()`; the reverse order can
/// crash in AURemoteIO teardown on iOS 18+ (field report, same article).
enum VoiceGraph {
    static func bringUp(_ steps: some VoiceGraphSteps, voiceProcessing: Bool, microphone: Bool) throws {
        try steps.configureSession()
        try steps.attachPlaybackGraph()
        if microphone {
            if voiceProcessing { try steps.enableVoiceProcessing() }
            try steps.installCaptureTap()
        }
        try steps.prepareAndStart()
    }

    static func tearDown(_ steps: some VoiceGraphSteps, voiceProcessing: Bool, microphone: Bool) {
        if microphone {
            steps.removeCaptureTap()
            if voiceProcessing { steps.disableVoiceProcessing() }
        }
        steps.stopEngine()
    }
}

extension AVAudioEngine {
    /// `connectNode(_:to:format:)` (macOS and iOS 27) throws when the destination cannot take the format; the older
    /// `connect(_:to:format:)`, deprecated in 27, raises an Objective-C exception that Swift cannot catch.
    func link(_ node: AVAudioNode, to destination: AVAudioNode, format: AVAudioFormat?) throws {
        if #available(macOS 27.0, iOS 27.0, *) {
            try connectNode(node, to: destination, format: format)
        } else {
            connect(node, to: destination, format: format)
        }
    }
}
