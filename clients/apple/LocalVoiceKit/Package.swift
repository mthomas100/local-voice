// swift-tools-version: 6.2
// The core of the Apple clients (PROTOCOL.md v1): codec, WebSocket connection with pre-connect buffering and
// reconnect, the audio engine (voice processing, 16 kHz capture, 24 kHz playback, mic gate, played_ms), and the
// client state machine. The iOS app, the macOS menu-bar app and the `lvclient` command-line client all use it.
import PackageDescription

let package = Package(
    name: "LocalVoiceKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "LocalVoiceKit", targets: ["LocalVoiceKit"]),
        .library(name: "LocalVoiceUI", targets: ["LocalVoiceUI"]),
        .executable(name: "lvclient", targets: ["lvclient"]),
    ],
    targets: [
        .target(name: "LocalVoiceKit"),
        .target(name: "LocalVoiceUI", dependencies: ["LocalVoiceKit"]),
        .executableTarget(name: "lvclient", dependencies: ["LocalVoiceKit"]),
        .testTarget(name: "LocalVoiceKitTests", dependencies: ["LocalVoiceKit"]),
        .testTarget(name: "LocalVoiceUITests", dependencies: ["LocalVoiceUI", "LocalVoiceKit"]),
    ]
)
