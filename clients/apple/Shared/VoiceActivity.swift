// Compiled into both the iPhone app and its widget extension: the Live Activity's data and its End button.

import ActivityKit
import AppIntents
import Foundation

/// The Live Activity: listening, thinking, speaking, what the agent is doing, on the Lock Screen and in the Dynamic
/// Island. It shows state only; it cannot start the microphone (ActivityKit has no capture, research notes).
struct VoiceActivityAttributes: ActivityAttributes {
    struct ContentState: Codable, Hashable {
        /// One line: "Listening", "Thinking…", "Reading your journal", "The Mac is busy".
        var status: String
        /// An SF Symbol name for the compact views.
        var symbol: String
        /// The last thing the agent said, shortened.
        var lastReply: String?
        var handsFree: Bool
        /// "atlas · act": the space and mode (optional, so a state encoded before it existed still decodes).
        var whereLine: String?
        /// The agent's question waiting for an answer (its headline, shortened), and when the server stops waiting.
        /// The card itself is in the app; the Live Activity only says a question waits.
        var question: String?
        var questionDeadline: Date?
    }

    var device: String
}

/// Set by the app at launch. LiveActivityIntents run in the app's process, so the widget extension never calls it.
@MainActor
enum SessionControl {
    static var endSession: (@MainActor () async -> Void)?
}

/// The Live Activity's End button.
struct EndSessionIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "End the conversation"
    static let description = IntentDescription("Stops listening and ends the hands-free conversation.")

    @MainActor
    func perform() async throws -> some IntentResult {
        await SessionControl.endSession?()
        return .result()
    }
}
