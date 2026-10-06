import AppKit
import Carbon.HIToolbox
import Foundation

/// A global hold-to-talk hotkey through Carbon's `RegisterEventHotKey`, which reports both the press and the release
/// and needs neither Accessibility nor Input Monitoring permission (`NSEvent` global monitors need
/// Input Monitoring). Key repeat while held is ignored.
///
/// Option+Space is the default. A dictation app such as Handy may be bound to
/// Option+Space too, so the combination is configurable (`LVHotKey`, e.g. "control+option+space") and a failed
/// registration is reported rather than silent.
@MainActor
final class HotKey {
    struct Combo: Equatable, CustomStringConvertible {
        var keyCode: UInt32
        var modifiers: UInt32
        var description: String

        static let optionSpace = Combo(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey), description: "⌥Space")

        /// "option+space", "control+option+space", "command+shift+k", "option+f13".
        static func parse(_ text: String) -> Combo? {
            var modifiers: UInt32 = 0
            var symbols = ""
            var key: UInt32?
            var keyName = ""
            for part in text.lowercased().split(separator: "+").map(String.init) {
                switch part {
                case "control", "ctrl": modifiers |= UInt32(controlKey); symbols += "⌃"
                case "option", "alt": modifiers |= UInt32(optionKey); symbols += "⌥"
                case "shift": modifiers |= UInt32(shiftKey); symbols += "⇧"
                case "command", "cmd": modifiers |= UInt32(cmdKey); symbols += "⌘"
                default:
                    key = keyCodes[part]
                    keyName = part.count == 1 ? part.uppercased() : part.capitalized
                }
            }
            guard let key else { return nil }
            return Combo(keyCode: key, modifiers: modifiers, description: symbols + keyName)
        }

        private static let keyCodes: [String: UInt32] = {
            var map: [String: UInt32] = [
                "space": UInt32(kVK_Space), "return": UInt32(kVK_Return), "escape": UInt32(kVK_Escape),
                "f13": UInt32(kVK_F13), "f14": UInt32(kVK_F14), "f15": UInt32(kVK_F15), "f16": UInt32(kVK_F16),
                "f17": UInt32(kVK_F17), "f18": UInt32(kVK_F18), "f19": UInt32(kVK_F19),
            ]
            let letters: [(String, Int)] = [
                ("a", kVK_ANSI_A), ("b", kVK_ANSI_B), ("c", kVK_ANSI_C), ("d", kVK_ANSI_D), ("e", kVK_ANSI_E),
                ("f", kVK_ANSI_F), ("g", kVK_ANSI_G), ("h", kVK_ANSI_H), ("i", kVK_ANSI_I), ("j", kVK_ANSI_J),
                ("k", kVK_ANSI_K), ("l", kVK_ANSI_L), ("m", kVK_ANSI_M), ("n", kVK_ANSI_N), ("o", kVK_ANSI_O),
                ("p", kVK_ANSI_P), ("q", kVK_ANSI_Q), ("r", kVK_ANSI_R), ("s", kVK_ANSI_S), ("t", kVK_ANSI_T),
                ("u", kVK_ANSI_U), ("v", kVK_ANSI_V), ("w", kVK_ANSI_W), ("x", kVK_ANSI_X), ("y", kVK_ANSI_Y),
                ("z", kVK_ANSI_Z),
            ]
            for (name, code) in letters { map[name] = UInt32(code) }
            return map
        }()
    }

    let combo: Combo
    private(set) var status: OSStatus = noErr
    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private var isDown = false
    private let onPress: @MainActor () -> Void
    private let onRelease: @MainActor () -> Void
    private static let signature: OSType = 0x4C56_484B  // "LVHK"

    init(combo: Combo, onPress: @escaping @MainActor () -> Void, onRelease: @escaping @MainActor () -> Void) {
        self.combo = combo
        self.onPress = onPress
        self.onRelease = onRelease
    }

    var isRegistered: Bool { hotKeyRef != nil }

    @discardableResult
    func register() -> Bool {
        guard hotKeyRef == nil else { return true }
        var specs = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]
        let context = Unmanaged.passUnretained(self).toOpaque()
        status = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
            guard let event, let context else { return OSStatus(eventNotHandledErr) }
            var id = EventHotKeyID()
            let read = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                         nil, MemoryLayout<EventHotKeyID>.size, nil, &id)
            guard read == noErr, id.signature == HotKey.signature else { return OSStatus(eventNotHandledErr) }
            let kind = GetEventKind(event)
            // Carbon delivers hotkey events on the main thread's event loop.
            MainActor.assumeIsolated {
                let hotKey = Unmanaged<HotKey>.fromOpaque(context).takeUnretainedValue()
                hotKey.handle(pressed: kind == UInt32(kEventHotKeyPressed))
            }
            return noErr
        }, specs.count, &specs, context, &handlerRef)
        guard status == noErr else { return false }
        status = RegisterEventHotKey(combo.keyCode, combo.modifiers, EventHotKeyID(signature: Self.signature, id: 1),
                                     GetApplicationEventTarget(), 0, &hotKeyRef)
        if status != noErr {
            hotKeyRef = nil
            if let handlerRef { RemoveEventHandler(handlerRef) }
            handlerRef = nil
        }
        return status == noErr
    }

    func unregister() {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
        if let handlerRef { RemoveEventHandler(handlerRef) }
        hotKeyRef = nil
        handlerRef = nil
    }

    private func handle(pressed: Bool) {
        if pressed {
            guard !isDown else { return }  // key repeat
            isDown = true
            onPress()
        } else {
            guard isDown else { return }
            isDown = false
            onRelease()
        }
    }

    /// Running apps known to bind the same combination. Carbon accepts a second registration of a combination
    /// another process holds (tested 2026-10-05: status 0 both times), so a clash is silent; this names the likely one.
    /// Handy, a dictation app, binds ⌥Space by default.
    var conflicts: [String] {
        guard combo == .optionSpace else { return [] }
        return NSRunningApplication.runningApplications(withBundleIdentifier: "com.pais.handy")
            .map { $0.localizedName ?? "Handy" }
    }

    var statusDescription: String {
        if isRegistered, let other = conflicts.first {
            return "Hold \(combo) to talk; \(other) may also use \(combo)"
        }
        if isRegistered { return "Hold \(combo) to talk" }
        if status == OSStatus(eventHotKeyExistsErr) { return "\(combo) is taken by another app; set LVHotKey" }
        return "Hotkey \(combo) unavailable (\(status))"
    }
}
