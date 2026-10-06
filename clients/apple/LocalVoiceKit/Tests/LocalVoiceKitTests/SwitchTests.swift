import Foundation
import Testing
@testable import LocalVoiceKit

@Suite("Space and mode switches")
struct SwitchTests {
    func outcomes(_ h: CoreHarness) -> [SwitchOutcome] {
        h.events.withLock { $0.compactMap { if case let .switchOutcome(o) = $0 { o } else { nil } } }
    }

    @Test("a switch sends space or mode; the server's next space message answers it")
    func answered() {
        let h = CoreHarness(settings: ClientCoreTests.ptt)
        let atlas = SwitchRequest(.space, "atlas")
        h.run { $0.requestSwitch(atlas) }
        #expect(h.controls == [.space(name: "atlas")])
        let info = SpaceInfo(name: "atlas", mode: "conversation", tier: "trusted", description: "your atlas journal")
        h.server(.space(info))
        #expect(outcomes(h) == [.switched(atlas, info)])

        let act = SwitchRequest(.mode, "act")
        h.run { $0.requestSwitch(act) }
        #expect(h.controls.last == .mode(name: "act"))
        h.server(.space(SpaceInfo(name: "atlas", mode: "conversation")))
        #expect(outcomes(h).last == .refused(act, reason: "the server stayed in conversation"))

        // A space message nobody asked for (a switch by voice, or the one after welcome) is no outcome.
        h.server(.space(SpaceInfo(name: "home")))
        #expect(outcomes(h).count == 2)
    }

    @Test("an error while a switch waits refuses it; 5 s of silence is no answer")
    func refusedAndUnanswered() {
        let h = CoreHarness(settings: ClientCoreTests.ptt)
        let nowhere = SwitchRequest(.space, "nowhere")
        h.run { $0.requestSwitch(nowhere) }
        h.server(.error(code: "unknown_space", message: "There is no space called nowhere"))
        #expect(outcomes(h) == [.refused(nowhere, reason: "There is no space called nowhere")])

        let act = SwitchRequest(.mode, "act")
        h.run { $0.requestSwitch(act) }
        h.clock.advance(ms: 4900)
        h.run { $0.tick(at: h.clock.now) }
        #expect(outcomes(h).count == 1)
        h.clock.advance(ms: 200)
        h.run { $0.tick(at: h.clock.now) }
        #expect(outcomes(h).last == .noAnswer(act))
        // An error after that is only an error.
        h.server(.error(code: "stt", message: "speech recognition failed"))
        #expect(outcomes(h).count == 2)
    }
}
