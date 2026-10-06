import Foundation
import ImageIO
import LocalVoiceKit
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LocalVoiceUI

/// The dashboard views drawn by `ImageRenderer` from states folded out of real event sequences: evidence of what the
/// owner sees, with no app launched and no simulator booted. With LV_SNAPSHOT_DIR set (`../build.sh snapshots`), the
/// PNGs are kept there.
@MainActor
@Suite("Dashboard snapshots")
struct DashboardSnapshotTests {
    /// What the orchestrator says over a session: welcome, the space after it, a status, a turn's latency, a tool.
    static func working() -> Dashboard {
        var d = Dashboard()
        d.apply(.server(.welcome(Welcome(session: "s3f9a", space: "atlas", mode: "act", tier: "trusted",
                                         state: .listening, hold: HoldStatus(phase: .open)))))
        d.apply(.server(.space(SpaceInfo(name: "atlas", mode: "act", tier: "trusted",
                                         description: "your atlas journal"))))
        let status = ServerStatus(state: .thinking, space: "atlas", mode: "act", tier: "trusted", model: "local/qwen38",
                                  hold: HoldStatus(phase: .open),
                                  lastTurn: .init(eosToFirstAudioMs: 843.7, sttMs: 41.2, llmFirstTokenMs: 301,
                                                  ttsFirstAudioMs: 129.9, tools: ["read"], space: "atlas"),
                                  turns: 7,
                                  spaces: [.init(name: "atlas", description: "your atlas journal", tier: "trusted",
                                                 model: "local/qwen38"),
                                           .init(name: "home", description: "your Mac", tier: "ask",
                                                 model: "local/qwen38")],
                                  clients: [.init(device: "mac", kind: "mac", connectedSeconds: 600)])
        d.apply(.status(.turnEnded, status), now: Date(timeIntervalSince1970: 1_791_234_567))
        d.apply(.latency(TurnLatency(reply: "r1", stopToAudioStartMs: 870, stopToPlaybackMs: 935,
                                     releaseToPlaybackMs: 968)))
        d.apply(.server(.state(.thinking)))
        d.apply(.server(.tool(ToolEvent(phase: .start, name: "read", label: "reading your journal"))))
        return d
    }

    /// The GPU held by a render, and a switch the server never answered (the M1 orchestrator).
    static func held() -> Dashboard {
        var d = Dashboard()
        d.apply(.server(.welcome(Welcome(session: "s1", space: "home", mode: "conversation", tier: "ask",
                                         state: .held, hold: HoldStatus(phase: .held, why: "film render")))))
        d.apply(.server(.space(SpaceInfo(name: "home", mode: "conversation", tier: "ask", description: "your Mac"))))
        d.apply(.switchRequested(SwitchRequest(.space, "atlas")))
        d.apply(.switchOutcome(.noAnswer(SwitchRequest(.space, "atlas"))))
        return d
    }

    @Test("the card and the strip render, in light and dark")
    func renders() throws {
        let cases: [(String, AnyView)] = [
            ("dashboard-working", AnyView(DashboardCard(dashboard: Self.working(), connected: true))),
            ("dashboard-held", AnyView(DashboardCard(dashboard: Self.held(), connected: true))),
            ("strip-working", AnyView(DashboardStrip(dashboard: Self.working()))),
            ("strip-held", AnyView(DashboardStrip(dashboard: Self.held()))),
        ]
        for (name, view) in cases {
            for scheme in [ColorScheme.light, .dark] {
                let image = try #require(render(view, scheme: scheme), "\(name) did not render")
                #expect(image.width > 200 && image.height > 20, "\(name): \(image.width)x\(image.height)")
                save(image, as: "\(name)-\(scheme == .dark ? "dark" : "light").png")
            }
        }
        #expect(DashboardStrip(dashboard: Self.working()).line
                == "atlas · act · trusted · local/qwen38 · last 844 ms")
        #expect(DashboardStrip(dashboard: Self.held()).line == "home · conversation · ask")
    }

    private func render(_ view: AnyView, scheme: ColorScheme) -> CGImage? {
        let content = view
            .padding(14)
            .frame(width: 400, alignment: .leading)
            .background(scheme == .dark ? Color.black : Color.white)
            .environment(\.colorScheme, scheme)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        return renderer.cgImage
    }

    private func save(_ image: CGImage, as name: String) {
        guard let dir = ProcessInfo.processInfo.environment["LV_SNAPSHOT_DIR"], !dir.isEmpty else { return }
        let folder = URL(fileURLWithPath: dir)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        guard let out = CGImageDestinationCreateWithURL(folder.appendingPathComponent(name) as CFURL,
                                                        UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(out, image, nil)
        CGImageDestinationFinalize(out)
    }
}
