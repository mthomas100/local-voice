# Local Voice: the Apple clients (milestone M2)

The iPhone app and the Mac menu-bar app for the local voice agent. Both speak wire protocol v1 (`../../docs/PROTOCOL.md`)
to the orchestrator on the Mac (`ws://<mac>:8770/v1/voice`), through one Swift package.

## Layout

| Path | What it is |
|---|---|
| `LocalVoiceKit/` | Swift package. `LocalVoiceKit`: protocol v1 codec, WebSocket connection (pre-connect buffering, keepalive, reconnect), audio engines (voice processing, 16 kHz capture, 24 kHz playback, mic gate, `played_ms`), the client state machine. `LocalVoiceUI`: the session model and views both apps share. `lvclient`: the same core on the command line |
| `iOS/`, `Widgets/`, `Shared/` | the iPhone app, its Live Activity widget extension, and what the two share |
| `macOS/` | the menu-bar app: hold-to-talk hotkey, floating panel |
| `project.yml` | XcodeGen spec, the source of the Xcode project (generated, git-ignored) |
| `MockServer/` | a protocol v1 server with no models (Python), and the end-to-end tests |
| `build.sh` | every build and test, one command each |

## Build and test

```bash
cd clients/apple
./build.sh test        # 80 unit tests of the core and 3 UI tests, no device (swift test, ~4 s)
./build.sh e2e         # 23 protocol and status scenarios: lvclient against the mock over a real WebSocket (~2 min)
./build.sh mac         # the menu-bar app  -> build/DerivedData/Build/Products/Debug/LocalVoice.app
./build.sh ios         # the iPhone app for the simulator
./build.sh ios-ptt     # the same with Push to Talk compiled in
./build.sh apps-e2e    # both apps unattended against the mock (the iPhone app in the iOS simulator)
LV_REAL_URL=ws://127.0.0.1:8770/v1/voice ./build.sh real       # lvclient against the real orchestrator (runs models)
LV_REAL_URL=ws://127.0.0.1:8770/v1/voice ./build.sh apps-real  # both apps, one real turn each (runs models)
./build.sh generate    # LocalVoice.xcodeproj from project.yml, to open in Xcode
```

These builds are signed ad hoc so they never ask for the keychain. For the phone, open the generated project in
Xcode: it signs automatically with the team in `project.yml` (set `DEVELOPMENT_TEAM` to your team ID first). Edit
`project.yml`, never the generated project.

Requirements: Xcode 27, `xcodegen`, `uv` (the mock server's venv lands in `MockServer/.venv`, git-ignored).

## Run

- **Mac**: `open build/DerivedData/Build/Products/Debug/LocalVoice.app`. It lives in the menu bar; hold **⌥Space** to
  talk (a dictation app such as Handy may also use ⌥Space; the menu says so when Handy is running, and `LVHotKey`
  moves the key). It connects to `ws://127.0.0.1:8770/v1/voice`.
- **iPhone**: default server `ws://$(LV_MAC_TAILNET_HOST):8770/v1/voice` over Tailscale. Set `LV_MAC_TAILNET_HOST` in
  `project.yml` to your Mac's MagicDNS name (placeholder `your-mac.your-tailnet.ts.net`), or change the server in the
  app's settings.
- **Command line**: `LocalVoiceKit/.build/debug/lvclient --help`. A scripted turn against any server:

  ```bash
  say -v Samantha -o /tmp/q.aiff "what does my knowledge base say about the hold gate"
  LocalVoiceKit/.build/debug/lvclient --url ws://127.0.0.1:8770/v1/voice --mic ptt \
    --script "press; say-wait:file:/tmp/q.aiff; release; wait:end_of_turn:60000; wait:sent:played_ms:60000" \
    --log /tmp/lv/client.jsonl --record-output /tmp/lv/reply.wav
  ```

  `--audio headless` (default) plays replies into a recording with no device; `device` uses the speaker;
  `microphone` uses the real microphone with voice processing (asks for permission once).
- **Mock server**: `MockServer/.venv/bin/python MockServer/mock_server.py --port 18770 --reply-voice say` answers every
  turn with macOS `say`; `--help` lists the scenarios (4403, 4409, 1002, dropped connection, held GPU, ...).
  `--approval write,edit,bash,session,long,cancel,timeout,legacy` asks an approval question before each answer, one
  scenario per turn (PROTOCOL.md "Approvals").

## Settings

UserDefaults keys; launch arguments override them (`-LVServerURL ws://...`), and `defaults write local.voice.mac ...`
sets them on the Mac.

| Key | Default | Meaning |
|---|---|---|
| `LVServerURL` | Mac `ws://127.0.0.1:8770/v1/voice`; iPhone the Mac's tailnet name | the voice server |
| `LVDevice` | `mac` / `iphone` | `hello.device`: one per device; a second connection with the same name takes over (4409) |
| `LVMicMode` | `ptt` | `ptt` (hold to talk) or `vad` (open microphone); hands-free switches to `vad` |
| `LVGateMs`, `LVGateMode` | `600`, `hard` | mic gate after a reply drains (open microphone only); `twoTier` lets loud speech through |
| `LVPrerollMs` | Mac `60`, iPhone `100` | reply audio buffered before playback starts |
| `LVEngineIdleSeconds` | Mac `60`, iPhone `30` | keep the audio engine warm after the last use (its start costs ~0.4-0.5 s); `0` never stops |
| `LVPressChirp` | on with the real microphone | a blip when the microphone is live after a press |
| `LVHotKey` (Mac) | `option+space` | e.g. `control+option+space`, `command+shift+k`, `option+f13` |
| `LVCallMode`, `LVLiveActivity` (iPhone) | off, on | hands-free as a CallKit call; the Live Activity |
| `LVAudio` | `microphone` | `synthetic` or `headless` for unattended runs (no permission prompt) |
| `LVScript`, `LVEventLog`, `LVQuitAfterScript` | | unattended test mode: run a session script, log JSONL events, quit |

Script steps (`lvclient --script`, `-LVScript`): `press`, `release`, `handsfree-on`, `stop-speaking`,
`say-wait:tone:440:1.0`, `say-wait:file:<path>`, `sleep:<ms>`, `wait:<event>[:<ms>]` (e.g. `wait:end_of_turn`,
`wait:sent:played_ms`, `wait:stopped:4403`), `text:<words>`, `connect`, `disconnect`; for the approval card
`wait:approval-shown`, `confirm:yes` (Do it), `confirm:no` (Don't), `choose:<id>` (`choose:allow_session`),
`wait:approval-closed` and `snapshot:<label>` (the Mac app draws its panel to `<LVSnapshot name>.<label>.png`).

## How it works, briefly

- **One ordered outbox.** Everything sent goes through one stream drained by one sender. Until `welcome`, items wait,
  so a press before the socket is up sends `hello`, then `start` and the audio captured meanwhile, in order.
- **One serial queue for the session.** Server messages and reply audio in arrival order, capture, user actions and
  player completions all run on it, so `audio_start` is always seen before its audio and a flush happens before any
  message about it.
- **Voice processing in the documented order**, written once in `VoiceGraph.swift`: session, playback graph, voice
  processing, capture tap in the processed format, start; teardown turns voice processing off before stopping.
- **`played_ms`** counts completed buffers plus the part of the current one, per reply; `AVAudioPlayerNode`'s own
  clock keeps running while starved, so it cannot say what was heard.
- **Unattended tests** use a synthetic microphone (planted tones or a `say` recording at 48 kHz, through the same
  16 kHz converter) and a headless engine (manual rendering paced at real time), so nothing asks for a permission.
