import AVFoundation
import Foundation
import Synchronization
import Testing
@testable import LocalVoiceKit

final class RecordingSteps: VoiceGraphSteps {
    var log: [String] = []
    func configureSession() throws { log.append("session") }
    func attachPlaybackGraph() throws { log.append("playback graph") }
    func enableVoiceProcessing() throws { log.append("voice processing on") }
    func installCaptureTap() throws { log.append("capture tap") }
    func prepareAndStart() throws { log.append("prepare, start") }
    func removeCaptureTap() { log.append("remove tap") }
    func disableVoiceProcessing() { log.append("voice processing off") }
    func stopEngine() { log.append("stop") }
}

@Suite("Voice-processing order")
struct VoiceGraphTests {
    @Test("playback graph before voice processing, tap after it, voice processing off before stop")
    func documentedOrder() throws {
        let steps = RecordingSteps()
        try VoiceGraph.bringUp(steps, voiceProcessing: true, microphone: true)
        #expect(steps.log == ["session", "playback graph", "voice processing on", "capture tap", "prepare, start"])
        steps.log.removeAll()
        VoiceGraph.tearDown(steps, voiceProcessing: true, microphone: true)
        #expect(steps.log == ["remove tap", "voice processing off", "stop"])
    }

    @Test("without a microphone nothing touches the input node")
    func noMicrophone() throws {
        let steps = RecordingSteps()
        try VoiceGraph.bringUp(steps, voiceProcessing: true, microphone: false)
        VoiceGraph.tearDown(steps, voiceProcessing: true, microphone: false)
        #expect(steps.log == ["session", "playback graph", "prepare, start", "stop"])
    }
}

/// Real AVAudioEngine work, at real-time pace: serialized so timings are not disturbed by each other.
@Suite("Audio engines", .serialized)
struct AudioEngineTests {
    static func toneData(frequency: Double, seconds: Double, rate: Double = 24_000) -> Data {
        PCM16.data(SyntheticMicrophone.tone(frequency: frequency, seconds: seconds, amplitude: 0.5, sampleRate: rate)
            .map { Int16(clamping: Int(($0 * 32767).rounded())) })
    }

    @Test("headless engine: scheduled 24 kHz speech plays at real time through the mixer, and completions fire")
    func headlessPlayback() async throws {
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("lv-headless-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        let io = HeadlessAudioIO(microphone: nil, recordOutputTo: wav)
        _ = try io.start { _, _ in }
        let completed = Mutex(0)
        let data = Self.toneData(frequency: 523.25, seconds: 1.0)
        let t0 = MonotonicClock.now()
        // 25 chunks of 40 ms, as the server sends them.
        for i in stride(from: 0, to: data.count, by: 1920) {
            io.player.schedule(data.subdata(in: i..<min(i + 1920, data.count))) { completed.withLock { $0 += 1 } }
        }
        try await Task.sleep(for: .milliseconds(600))
        let midway = completed.withLock { $0 }
        #expect((11...17).contains(midway), "real-time pace: about 14 of 25 chunks after 0.6 s, got \(midway)")
        #expect(await waitUntil(.seconds(2)) { completed.withLock { $0 } == 25 })
        let elapsed = Double(MonotonicClock.now() - t0) / 1e9
        #expect(elapsed > 0.95, "not faster than real time: \(elapsed) s")
        io.stop()
        io.monitor.close()
        #expect(abs(io.monitor.audibleSeconds - 1.0) < 0.05, "audible \(io.monitor.audibleSeconds) s")
        // The recorded output is the same tone, now at 48 kHz.
        let heard = try SyntheticMicrophone.load(wav, sampleRate: 48_000).map { Int16(clamping: Int($0 * 32767)) }
        #expect(abs(Signal.frequency(heard.filter { _ in true }, sampleRate: 48_000, skip: 2000) - 523.25) < 3)
    }

    @Test("headless engine: a flush silences the output within one render quantum")
    func headlessFlush() async throws {
        let io = HeadlessAudioIO(microphone: nil)
        _ = try io.start { _, _ in }
        let data = Self.toneData(frequency: 440, seconds: 2.0)
        io.player.schedule(data) {}
        try await Task.sleep(for: .milliseconds(400))
        let flushAt = MonotonicClock.now()
        io.player.flush()
        try await Task.sleep(for: .milliseconds(200))
        let last = try #require(io.monitor.lastAudibleAt)
        let tail = Double(Int64(last) - Int64(flushAt)) / 1e6
        #expect(tail < 20, "output went silent \(tail) ms after the flush")
        io.stop()
    }

    @Test("synthetic microphone: real-time pace, converted to 16 kHz, clip end reported")
    func syntheticMicrophone() async throws {
        let mic = SyntheticMicrophone(chunkFrames: 960)
        let samples = Mutex<[Int16]>([])
        let finished = Mutex<[String]>([])
        mic.onClipFinished = { name in finished.withLock { $0.append(name) } }
        mic.play(SyntheticMicrophone.tone(frequency: 1000, seconds: 0.5), name: "tone")
        let t0 = MonotonicClock.now()
        mic.start { s, _ in samples.withLock { $0 += s } }
        #expect(await waitUntil(.seconds(2)) { finished.withLock { $0 } == ["tone"] })
        let elapsed = Double(MonotonicClock.now() - t0) / 1e9
        #expect(elapsed >= 0.45 && elapsed < 0.8, "a 0.5 s clip took \(elapsed) s to be heard")
        try await Task.sleep(for: .milliseconds(200))
        mic.stop()
        let heard = samples.withLock { $0 }
        let first = try #require(heard.firstIndex { abs(Int($0)) > 1000 })
        let last = try #require(heard.lastIndex { abs(Int($0)) > 1000 })
        #expect(abs(Double(last - first) / 16_000 - 0.5) < 0.02, "the clip lasts 0.5 s at 16 kHz")
        #expect(abs(Signal.frequency(Array(heard.prefix(9000)), sampleRate: 16_000, skip: 400) - 1000) < 10)
    }

    @Test("device engine with a synthetic microphone and the output muted: start, play, stop")
    func liveEngineMuted() async throws {
        let io = LiveAudioIO(options: .init(capture: .synthetic(SyntheticMicrophone()), outputVolume: 0))
        let description = try io.start { _, _ in }
        #expect(description.contains("synthetic microphone"))
        #expect(description.contains("(muted)"))
        let completed = Mutex(0)
        io.player.schedule(Self.toneData(frequency: 440, seconds: 0.2)) { completed.withLock { $0 += 1 } }
        #expect(await waitUntil(.seconds(2)) { completed.withLock { $0 } == 1 })
        io.stop()
    }
}

/// The whole client in one process: connection actor, state machine and headless engine, against a scripted socket.
@Suite("Client end to end, in process", .serialized)
struct InProcessClientTests {
    @Test("a push-to-talk turn: tone in, reply out, played_ms back")
    func pushToTalkTurn() async throws {
        let mic = SyntheticMicrophone()
        let io = HeadlessAudioIO(microphone: mic)
        let connector = FakeConnector()
        let client = VoiceClient(
            configuration: .init(url: URL(string: "ws://127.0.0.1:1/v1/voice")!,
                                 hello: Hello(client: .test, device: "inproc", mic: .ptt)),
            audio: io, connector: connector)
        let events = Mutex<[ClientEvent]>([])
        let collector = Task { for await e in client.events { events.withLock { $0.append(e) } } }

        client.pressTalk()
        mic.play(SyntheticMicrophone.tone(frequency: 440, seconds: 0.6), name: "q")
        #expect(await waitUntil { connector.count == 1 })
        let t = try #require(connector.latest)
        t.deliver(ConnectionTests.welcome)
        try await Task.sleep(for: .milliseconds(800))
        client.releaseTalk()
        #expect(await waitUntil { t.sentTypes.contains("stop") })
        let kinds = t.sentTypes
        #expect(kinds.first == "hello")
        #expect(kinds[1] == "start")
        let frames = t.sentMessages.compactMap { if case let .binary(d) = $0 { d } else { nil } }
        #expect(frames.allSatisfy { $0.count == 640 })
        let audio = frames.flatMap { PCM16.samples($0) }
        #expect(abs(Signal.frequency(audio, sampleRate: 16_000, skip: 1000) - 440) < 5, "the tone arrives at 16 kHz")

        // The reply: 0.8 s of tone in 40 ms chunks.
        t.deliver(.state(.thinking))
        t.deliver(.audioStart(replyID: "r1", rate: 24_000))
        let reply = AudioEngineTests.toneData(frequency: 330, seconds: 0.8)
        for i in stride(from: 0, to: reply.count, by: 1920) {
            t.deliverAudio(reply.subdata(in: i..<min(i + 1920, reply.count)))
        }
        t.deliver(.audioEnd(replyID: "r1"))
        t.deliver(.endOfTurn(replyID: "r1"))
        #expect(await waitUntil(.seconds(3)) { t.sentTypes.contains("played_ms") })
        let played = t.sentMessages.compactMap { m -> Int? in
            if case let .text(s) = m, case let .playedMs("r1", ms)?? = try? ProtocolCodec.decodeClient(s) { return ms }
            return nil
        }
        #expect(played.count == 1)
        #expect(abs((played.first ?? 0) - 800) <= 30, "played_ms \(played)")
        #expect(abs(io.monitor.audibleSeconds - 0.8) < 0.06)

        await client.shutdown()
        collector.cancel()
        let latency = events.withLock { $0.compactMap { if case let .latency(l) = $0 { l } else { nil } } }
        #expect(latency.first?.stopToAudioStartMs != nil)
    }
}
