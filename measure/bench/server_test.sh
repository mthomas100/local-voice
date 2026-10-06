#!/bin/zsh
# server_test.sh <python> <model> <voice> <lang_code> — start mlx_audio.server on port 18800, probe endpoints,
# time /v1/audio/speech (stream true/false) with curl, then stop the server. Prints everything.
PY=$1; MODEL=$2; VOICE=$3; LANG_CODE=$4
SP=${SP:-${0:a:h:h}}
PORT=18800; SLOG=$SP/raw/server-$(echo $MODEL | tr '/' '_').log
cd $SP/audio
$PY -m mlx_audio.server --host 127.0.0.1 --port $PORT > $SLOG 2>&1 &
SPID=$!
echo "server pid $SPID, log $SLOG"
for i in $(seq 1 60); do curl -s --max-time 1 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 1; done
echo "up after ${i}s"; curl -s http://127.0.0.1:$PORT/v1/models; echo
echo "--- OPTIONS/HEAD probe of /v1/audio/speech (route exists?) ---"
curl -s -o /dev/null -w 'POST empty body -> HTTP %{http_code}\n' -X POST http://127.0.0.1:$PORT/v1/audio/speech -H 'content-type: application/json' -d '{}'
echo "--- warm load: non-stream short (model loads on first request) ---"
TEXT_SHORT="The kettle is on, and the rain has finally stopped outside."
TEXT_LONG="Before we start the meeting, let me summarize where things stand. The new build passed every test last night, the design review is scheduled for Thursday morning, and two customers have already asked when they can try the beta. I think we are in good shape."
body() { printf '{"model":"%s","input":"%s","voice":"%s","lang_code":"%s","response_format":"wav","stream":%s,"streaming_interval":%s}' "$MODEL" "$1" "$VOICE" "$LANG_CODE" "$2" "$3"; }
curl -s -o srv_warm.wav -w 'warm (includes model load): ttfb %{time_starttransfer}s total %{time_total}s bytes %{size_download} http %{http_code}\n' -X POST http://127.0.0.1:$PORT/v1/audio/speech -H 'content-type: application/json' -d "$(body "$TEXT_SHORT" false 2.0)"
for name in short long; do
  if [ $name = short ]; then T=$TEXT_SHORT; else T=$TEXT_LONG; fi
  for mode in "false 2.0" "true 2.0" "true 0.5"; do
    set -- ${=mode}
    for run in 1 2 3; do
      curl -s --no-buffer -o srv_${name}_stream${1}_${2}_${run}.wav -w "$name stream=$1 interval=$2 run$run: ttfb %{time_starttransfer}s total %{time_total}s bytes %{size_download} http %{http_code}\n" -X POST http://127.0.0.1:$PORT/v1/audio/speech -H 'content-type: application/json' -d "$(body "$T" $1 $2)"
    done
  done
done
echo "--- chunk arrival trace (stream=true, interval 0.5, long): python reader timestamps each chunk ---"
$PY - "$PORT" "$MODEL" "$VOICE" "$LANG_CODE" "$TEXT_LONG" <<'PYEOF'
import sys, json, time, urllib.request
port, model, voice, lang, text = sys.argv[1:6]
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech", data=json.dumps(dict(model=model, input=text, voice=voice, lang_code=lang, response_format="wav", stream=True, streaming_interval=0.5)).encode(), headers={"content-type": "application/json"})
t = time.perf_counter(); r = urllib.request.urlopen(req); print(f"headers after {time.perf_counter()-t:.3f}s:", dict(r.headers)); n = 0; tot = 0; data = b""
while True:
    b = r.read1(1 << 20)   # returns whatever has arrived (one HTTP chunk at most), does not wait to fill
    if not b: break
    n += 1; tot += len(b); data += b
    if n <= 6 or len(b) < 40000: print(f"  read {n}: +{time.perf_counter()-t:.3f}s {len(b)} bytes, starts with RIFF={b[:4]==b'RIFF'}")
print(f"done {tot} bytes in {time.perf_counter()-t:.3f}s, {n} reads, RIFF headers in stream: {data.count(b'RIFF')}")
PYEOF
echo "--- file check (afinfo reads only the first RIFF chunk of a streamed file) ---"; for f in srv_short_streamfalse_2.0_1.wav srv_short_streamtrue_2.0_1.wav srv_short_streamtrue_0.5_1.wav srv_long_streamtrue_0.5_1.wav; do printf '%s: %s bytes, RIFF headers %s, ' $f $(stat -f %z $f) $(grep -c RIFF $f); afinfo $f 2>&1 | grep -E 'estimated duration' | tr -d '\n'; echo; done
kill $SPID; sleep 1; kill -9 $SPID 2>/dev/null; echo "server stopped: $(pgrep -f 'mlx_audio.server' | wc -l) left"
