#!/bin/zsh
# Local Voice Apple clients: one entry point for every build and test (2026-10-05).
#
#   ./build.sh generate     write LocalVoice.xcodeproj and Generated/*.plist from project.yml (XcodeGen)
#   ./build.sh test         LocalVoiceKit unit tests (swift test; no device, no network beyond loopback)
#   ./build.sh e2e          protocol v1 end-to-end: lvclient against the mock server (pytest, ~2 min)
#   ./build.sh mac          build the menu-bar app        -> build/DerivedData/.../LocalVoice.app
#   ./build.sh ios          build the iPhone app for the simulator
#   ./build.sh ios-ptt      the same with Push to Talk compiled in (LV_PUSH_TO_TALK=YES)
#   ./build.sh apps-e2e     run both apps unattended against the mock server (scripts/apps_e2e.sh)
#   ./build.sh snapshots    draw the dashboard views and approval cards to PNG (ImageRenderer; no app, no simulator)
#                           -> MockServer/runs/<stamp>/snapshots
#   ./build.sh real         lvclient against the real orchestrator at LV_REAL_URL (it runs models on the GPU:
#                           only when nothing else needs it); LV_REAL_URL=mock is a dry run of the same tests against the mock
#   ./build.sh apps-real    both apps (Mac, iPhone simulator) each complete a turn against LV_REAL_URL (same rules)
#   ./build.sh all          everything above except real and apps-real, in order
#
# Unattended builds sign ad hoc ("-"), so no keychain prompt can appear; open the project in Xcode for automatic
# signing with the team (device installs). One heavy build at a time: Xcode builds here run sequentially.
#
# LV_JOBS=2 ./build.sh test   caps compile jobs (swift -j, xcodebuild -jobs) while the orchestrator measures latency,
# so builds do not skew its numbers. apps-e2e boots the simulator and launches the Mac app: not during those runs.
set -euo pipefail
cd "$(dirname "$0")"
DERIVED=build/DerivedData
SWIFT_JOBS=()
XCB_JOBS=()
if [[ -n "${LV_JOBS:-}" ]]; then
  SWIFT_JOBS=(-j "$LV_JOBS")
  XCB_JOBS=(-jobs "$LV_JOBS")
fi
ADHOC=(CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= PROVISIONING_PROFILE_SPECIFIER=)
SIM_NAME=${LV_SIMULATOR:-iPhone 18 Pro}

generate() {
  command -v xcodegen >/dev/null || { echo "xcodegen not found (brew install xcodegen)"; exit 1; }
  mkdir -p Generated
  xcodegen generate --spec project.yml --quiet
}

mock_venv() {
  if [[ ! -x MockServer/.venv/bin/python ]]; then
    uv venv --python 3.12 MockServer/.venv
    uv pip install --python MockServer/.venv/bin/python -r MockServer/requirements.txt
  fi
}

xcb() {
  xcodebuild -project LocalVoice.xcodeproj -derivedDataPath "$DERIVED" "${XCB_JOBS[@]}" "$@" 2>&1 \
    | grep -E "error:|warning:|BUILD SUCCEEDED|BUILD FAILED|\*\* " | grep -v "appintentsmetadataprocessor" || true
  return ${pipestatus[1]}
}

case "${1:-}" in
  generate) generate ;;
  test) (cd LocalVoiceKit && swift test "${SWIFT_JOBS[@]}") ;;
  e2e)
    mock_venv
    (cd LocalVoiceKit && swift build "${SWIFT_JOBS[@]}" --product lvclient)
    (cd MockServer && .venv/bin/python -m pytest -q "${@:2}")
    ;;
  mac)
    generate
    xcb -scheme LocalVoiceMac -configuration Debug -destination 'platform=macOS' "${ADHOC[@]}" build
    ;;
  ios)
    generate
    xcb -scheme LocalVoice -configuration Debug -destination "platform=iOS Simulator,name=$SIM_NAME" "${ADHOC[@]}" build
    ;;
  ios-ptt)
    generate
    xcb -scheme LocalVoice -configuration Debug -destination "platform=iOS Simulator,name=$SIM_NAME" "${ADHOC[@]}" \
      LV_PUSH_TO_TALK=YES build
    ;;
  apps-e2e)
    mock_venv
    scripts/apps_e2e.sh "${@:2}"
    ;;
  snapshots)
    out="$PWD/MockServer/runs/$(date +%Y%m%d-%H%M%S)/snapshots"
    (cd LocalVoiceKit && LV_SNAPSHOT_DIR="$out" swift test "${SWIFT_JOBS[@]}" --filter "DashboardSnapshotTests|ApprovalSnapshotTests")
    echo "snapshots: $out"
    ;;
  real)
    [[ -n "${LV_REAL_URL:-}" ]] || { echo "set LV_REAL_URL=ws://127.0.0.1:8770/v1/voice (runs models) or LV_REAL_URL=mock"; exit 2; }
    mock_venv
    (cd LocalVoiceKit && swift build "${SWIFT_JOBS[@]}" --product lvclient)
    (cd MockServer && .venv/bin/python -m pytest -m "real and not apps" -q "${@:2}")
    ;;
  apps-real)
    [[ -n "${LV_REAL_URL:-}" ]] || { echo "set LV_REAL_URL=ws://127.0.0.1:8770/v1/voice (runs models) or LV_REAL_URL=mock"; exit 2; }
    mock_venv
    "$0" mac
    "$0" ios
    (cd MockServer && .venv/bin/python -m pytest -m "apps and real" -q "${@:2}")
    ;;
  all)
    "$0" test && "$0" e2e && "$0" mac && "$0" ios && "$0" ios-ptt && "$0" apps-e2e
    ;;
  *)
    sed -n '2,22p' "$0"
    exit 2
    ;;
esac
