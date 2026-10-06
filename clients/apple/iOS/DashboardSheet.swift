import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

/// The dashboard on the phone: where the agent is working and in which mode, switches for both, what it is doing,
/// the GPU, the last turn's latency, and the connection. Fetched when it opens and when pulled, otherwise kept current
/// by the session's own messages (no polling).
struct DashboardSheet: View {
    let model: VoiceSessionModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    DashboardCard(dashboard: model.dashboard, connected: model.isReady)
                        .padding(.vertical, 4)
                }
                Section {
                    SpaceModeControls(model: model)
                } header: {
                    Text("Switch")
                } footer: {
                    Text("Act mode lets the agent change things (write files, run commands); in a space whose tier "
                         + "asks first, it asks you aloud before each one.")
                }
                Section("Connection") {
                    LabeledContent("Server", value: model.settings.serverURL.host() ?? model.settings.serverURL.absoluteString)
                    LabeledContent("This device", value: model.settings.device)
                    if let session = model.dashboard.session {
                        LabeledContent("Session", value: session)
                    }
                    ForEach(model.dashboard.clients, id: \.device) { client in
                        LabeledContent(client.device, value: connectedFor(client))
                    }
                }
            }
            .navigationTitle("Status")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .refreshable {
                model.refreshStatus()
                try? await Task.sleep(for: .milliseconds(700))  // the fetch is debounced by 300 ms
            }
            .task { model.refreshStatus() }
        }
    }

    private func connectedFor(_ client: ServerStatus.Client) -> String {
        let kind = client.kind ?? "client"
        guard let s = client.connectedSeconds else { return kind }
        let minutes = Int(s) / 60
        return minutes > 0 ? "\(kind), \(minutes) min" : "\(kind), \(Int(s)) s"
    }
}
