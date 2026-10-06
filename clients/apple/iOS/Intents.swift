import AppIntents
import Foundation

/// The Action button (or Shortcuts, or Siri) opens the app and starts a hands-free conversation.
///
/// Recording cannot start from the background ("Target is not foreground", Apple forum 815725), so the intent brings
/// the app to the foreground first: `openAppWhenRun` through iOS 25, `supportedModes` `.foreground(.immediate)` from
/// iOS 26, where `openAppWhenRun` is deprecated (iOS 27 SDK).
struct StartTalkingIntent: AppIntent {
    static let title: LocalizedStringResource = "Talk to my agent"
    static let description = IntentDescription("Opens Local Voice and starts a hands-free conversation with the agent on your Mac.")
    static let openAppWhenRun = true

    @available(iOS 26.0, *)
    static var supportedModes: IntentModes { .foreground(.immediate) }

    @MainActor
    func perform() async throws -> some IntentResult {
        if let hub = SessionHub.shared {
            await hub.startHandsFree()
        } else {
            SessionHub.pendingStart = true
        }
        return .result()
    }
}

struct LocalVoiceShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: StartTalkingIntent(),
                    phrases: ["Talk to \(.applicationName)", "Start \(.applicationName)"],
                    shortTitle: "Talk to my agent",
                    systemImageName: "waveform")
    }
}
