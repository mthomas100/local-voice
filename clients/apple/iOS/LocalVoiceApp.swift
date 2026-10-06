import LocalVoiceUI
import SwiftUI

/// The iPhone client: talk to the agent on your Mac over Tailscale (protocol v1), hold to talk or hands-free, from
/// the Action button, with a Live Activity and an optional CallKit call for long hands-free sessions.
@main
struct LocalVoiceApp: App {
    @State private var hub = SessionHub()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            TalkView(hub: hub)
                .task { hub.launched() }
        }
        .onChange(of: scenePhase) { _, phase in
            hub.scenePhaseChanged(phase)
        }
    }
}
