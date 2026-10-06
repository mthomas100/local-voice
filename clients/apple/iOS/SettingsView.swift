import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

struct SettingsView: View {
    let hub: SessionHub
    @Environment(\.dismiss) private var dismiss
    @State private var url = ""
    @State private var device = ""
    @State private var gateMs = 600
    @State private var prerollMs = 100
    @State private var callMode = false
    @State private var liveActivities = true

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("ws://your-mac.tailnet.ts.net:8770/v1/voice", text: $url)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    TextField("Device name", text: $device)
                        .textInputAutocapitalization(.never)
                } header: {
                    Text("Server")
                } footer: {
                    Text("The voice server on your Mac, over Tailscale. One device name per phone: a second connection with the same name takes over the first.")
                }
                Section("Hands-free") {
                    Toggle("Use a call (CallKit)", isOn: $callMode)
                    Toggle("Live Activity", isOn: $liveActivities)
                    Stepper("Mic gate after replies: \(gateMs) ms", value: $gateMs, in: 0...1200, step: 50)
                }
                Section("Playback") {
                    Stepper("Pre-roll: \(prerollMs) ms", value: $prerollMs, in: 40...400, step: 20)
                }
                #if PUSH_TO_TALK
                if let ptt = hub.pushToTalk {
                    Section {
                        Button(ptt.joined ? "Leave the Push to Talk channel" : "Join the Push to Talk channel") {
                            ptt.joined ? ptt.leave() : ptt.join()
                        }
                    } header: {
                        Text("Push to Talk")
                    } footer: {
                        Text("The system talk button on the Lock Screen. Agent-initiated speech also needs the server to send Push to Talk notifications with your APNs key.")
                    }
                }
                #endif
                if let latency = hub.model.latency {
                    Section("Last turn") {
                        LabeledContent("Release to audio start",
                                       value: latency.stopToAudioStartMs.map { "\($0) ms" } ?? "—")
                        LabeledContent("Release to first word", value: latency.stopToPlaybackMs.map { "\($0) ms" } ?? "—")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save", action: save)
                }
            }
            .onAppear(perform: load)
        }
    }

    private func load() {
        let s = hub.model.settings
        url = s.serverURL.absoluteString
        device = s.device
        gateMs = s.gateMs
        prerollMs = s.prerollMs
        callMode = hub.callMode
        liveActivities = hub.liveActivitiesOn
    }

    private func save() {
        var s = hub.model.settings
        if let u = URL(string: url), u.scheme == "ws" || u.scheme == "wss" { s.serverURL = u }
        s.device = device.isEmpty ? "iphone" : device
        s.gateMs = gateMs
        s.prerollMs = prerollMs
        hub.callMode = callMode
        hub.liveActivitiesOn = liveActivities
        Task { await hub.model.update(settings: s) }
        dismiss()
    }
}
