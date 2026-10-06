import AppKit
import LocalVoiceKit
import LocalVoiceUI
import SwiftUI

/// The Mac client: a menu-bar app (no Dock icon) with a global hold-to-talk hotkey and a floating panel, speaking
/// protocol v1 to the orchestrator on this Mac (ws://127.0.0.1:8770/v1/voice by default).
@main
struct LocalVoiceMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        MenuBarExtra {
            MenuContentView(model: delegate.model, hotKeyStatus: delegate.hotKeyStatus,
                            panelPinned: Binding(get: { delegate.panelPinned }, set: { delegate.pinPanel($0) }))
        } label: {
            Image(systemName: delegate.model.symbolName)
                .accessibilityLabel("Local Voice: \(delegate.model.statusLine)")
        }
        .menuBarExtraStyle(.window)

        Settings {
            MacSettingsView(model: delegate.model)
        }
    }
}

@MainActor
@Observable
final class AppDelegate: NSObject, NSApplicationDelegate {
    @ObservationIgnored let model: VoiceSessionModel
    private(set) var hotKeyStatus = "Hotkey not registered"
    /// The floating panel stays on screen: the dashboard as a widget.
    private(set) var panelPinned = false {
        didSet { panel?.setPinned(panelPinned, model: model) }
    }

    /// The menu's toggle, the only place that saves LVPinPanel: a launch argument (unattended runs) must never end up
    /// in the user's preferences.
    func pinPanel(_ on: Bool) {
        panelPinned = on
        UserDefaults.standard.set(on, forKey: "LVPinPanel")
    }
    @ObservationIgnored private var hotKey: HotKey?
    @ObservationIgnored private var panel: FloatingPanelController?

    override init() {
        let settings = AppSettings.load(client: .mac)
        model = VoiceSessionModel(settings: settings, client: .mac)
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let defaults = UserDefaults.standard
        let panel = FloatingPanelController(model: model)
        self.panel = panel
        panelPinned = defaults.bool(forKey: "LVPinPanel")
        model.onChange = { [weak panel] model in panel?.update(for: model) }
        if defaults.bool(forKey: "LVShowPanel") { panel.show() }

        // Unattended runs pass -LVRegisterHotKey NO so a dictation app's Option+Space (such as Handy) is never touched.
        let register = defaults.object(forKey: "LVRegisterHotKey") == nil || defaults.bool(forKey: "LVRegisterHotKey")
        let combo = defaults.string(forKey: "LVHotKey").flatMap(HotKey.Combo.parse) ?? .optionSpace
        let hotKey = HotKey(combo: combo,
                            onPress: { [weak self] in self?.model.pressTalk() },
                            onRelease: { [weak self] in self?.model.releaseTalk() })
        if register { hotKey.register() }
        self.hotKey = hotKey
        hotKeyStatus = register ? hotKey.statusDescription : "Hotkey off (LVRegisterHotKey NO)"
        model.log?.write([("event", "hotkey"), ("combo", .string(combo.description)),
                          ("registered", .bool(hotKey.isRegistered)), ("status", .int(Int(hotKey.status))),
                          ("conflicts", .array(hotKey.conflicts.map { .string($0) }))])

        model.connect()
        let snapshot = defaults.string(forKey: "LVSnapshot")
        let model = self.model
        if let snapshot {
            // A script's snapshot:<label> step draws the panel as it is then: <name>.<label>.png beside <name>.png.
            let base = URL(fileURLWithPath: snapshot).deletingPathExtension()
            model.onSnapshot = { label in
                Self.snapshot(model: model, to: base.appendingPathExtension(label).appendingPathExtension("png"),
                              dashboard: false)
            }
        }
        model.runScriptIfConfigured { ok in
            if let snapshot { Self.snapshot(model: model, to: URL(fileURLWithPath: snapshot)) }
            guard model.settings.quitAfterScript else { return }
            await model.shutdown()
            exit(ok ? 0 : 1)
        }
    }

    /// Unattended evidence: the floating panel and the menu's dashboard card as SwiftUI renders them (no screen
    /// capture, so no Screen Recording permission prompt). ImageRenderer cannot draw scroll views, AppKit text fields or
    /// pickers, so the rest of the menu is not rendered. The card goes next to the panel as `<name>-dashboard.png`.
    static func snapshot(model: VoiceSessionModel, to url: URL, dashboard: Bool = true) {
        let card = url.deletingPathExtension().appendingPathExtension("dashboard").appendingPathExtension("png")
        // An approval card on the panel draws its boxes clipped rather than scrolling (approvalCardSnapshot).
        var views: [(URL, AnyView)] = [(url, AnyView(PanelView(model: model).environment(\.approvalCardSnapshot, true)))]
        if dashboard {
            views.append((card, AnyView(DashboardCard(dashboard: model.dashboard, connected: model.isReady)
                .padding(14).frame(width: 400, alignment: .leading))))
        }
        for (file, view) in views {
            let renderer = ImageRenderer(content: view.background(Color(nsColor: .windowBackgroundColor)))
            renderer.scale = 2
            guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
                  let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { continue }
            try? png.write(to: file)
            model.log?.write([("event", "snapshot"), ("file", .string(file.path))])
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        hotKey?.unregister()
    }
}

struct MenuContentView: View {
    let model: VoiceSessionModel
    let hotKeyStatus: String
    @Binding var panelPinned: Bool
    @State private var typed = ""
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            StatusPill(model: model)
            if let approval = model.pendingApproval {
                // First, above everything else: the agent waits for this answer.
                ApprovalCard(approval: approval, waitingBehind: model.approvals.waiting.count - 1,
                             previewMaxHeight: 200) { model.answer($0, to: approval.id) }
            }
            // The dashboard: where the agent is, how it works, and the switches for both.
            GroupBox {
                VStack(alignment: .leading, spacing: 10) {
                    DashboardCard(dashboard: model.dashboard, connected: model.isReady)
                    SpaceModeControls(model: model)
                }
                .padding(4)
            }
            if model.hold.phase == .held { HoldBanner(hold: model.hold) }
            if let problem = model.problem {
                Text(problem).font(.caption).foregroundStyle(.red)
            }
            TranscriptList(entries: Array(model.entries.suffix(12)), partial: model.partialTranscript)
                .frame(height: 180)
            if let label = model.toolLabel { ToolActivityRow(label: label) }
            HStack {
                TextField("Type to your agent", text: $typed)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(sendTyped)
                Button("Send", action: sendTyped)
                    .disabled(typed.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            HStack {
                Text(hotKeyStatus).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if model.isPlaying {
                    Button("Stop speaking", systemImage: "stop.fill") { model.stopSpeaking() }
                }
            }
            Toggle("Keep the panel on screen", isOn: $panelPinned)
                .font(.caption)
            Divider()
            HStack {
                Button("Reconnect") { model.connect() }
                Button("Settings…") { openSettings() }
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
            }
        }
        .padding(14)
        .frame(width: 400)
        .onAppear { model.refreshStatus() }
    }

    private func sendTyped() {
        model.send(text: typed)
        typed = ""
    }
}

struct MacSettingsView: View {
    let model: VoiceSessionModel
    @State private var url = ""
    @State private var device = ""
    @State private var gateMs = 600
    @State private var prerollMs = 100

    var body: some View {
        Form {
            TextField("Server", text: $url)
            TextField("Device name", text: $device)
            Stepper("Pre-roll: \(prerollMs) ms", value: $prerollMs, in: 40...400, step: 20)
            Stepper("Mic gate after replies: \(gateMs) ms", value: $gateMs, in: 0...1200, step: 50)
            Text("The hotkey is set with `defaults write local.voice.mac LVHotKey \"control+option+space\"`.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Save") {
                    var s = model.settings
                    if let u = URL(string: url) { s.serverURL = u }
                    s.device = device
                    s.gateMs = gateMs
                    s.prerollMs = prerollMs
                    Task { await model.update(settings: s) }
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 460)
        .onAppear {
            url = model.settings.serverURL.absoluteString
            device = model.settings.device
            gateMs = model.settings.gateMs
            prerollMs = model.settings.prerollMs
        }
    }
}
