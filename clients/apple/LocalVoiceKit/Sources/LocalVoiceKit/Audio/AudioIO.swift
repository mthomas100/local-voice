import Foundation

/// Receives 16 kHz Int16 mono capture samples, on the capture thread, with the time they were delivered.
public typealias CaptureHandler = @Sendable (_ samples: [Int16], _ at: Nanos) -> Void

/// An audio engine for the client: capture converted to 16 kHz, a speech player for 24 kHz replies, and a cue.
///
/// Implementations: `LiveAudioIO` (the device, with Apple's voice processing on the microphone), and
/// `HeadlessAudioIO` (no device at all: a synthetic microphone and an engine in manual rendering mode paced at real
/// time, for unattended tests). `start` and `stop` block, so callers run them off the main thread.
public protocol AudioIO: AnyObject, Sendable {
    var player: any PlayerDriving { get }
    var cue: (any CueDriving)? { get }
    /// Builds the graph, starts capture and playback, and returns a one-line description of the formats in use.
    func start(capture: @escaping CaptureHandler) throws -> String
    func stop()
    /// Set by the client: called when the engine stopped by itself (route change, interruption), with a reason.
    var onStopped: (@Sendable (String) -> Void)? { get set }
}
