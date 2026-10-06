"""Load and check config.yaml (plus an optional config.local.yaml merged over it).

The file is the single source for every model, voice and timing choice; this module only reads it, fills nothing in
silently, and fails with a message naming the key when something is wrong. Adapter settings stay free-form dicts,
because each adapter reads its own keys (the pi `tts` skill's pattern: `impl: module:Class` plus that class's
settings).
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent.parent          # orchestrator/
DEFAULT_CONFIG = HERE / "config.yaml"


class ConfigError(ValueError):
    """config.yaml (or a local override) is wrong; the message names the key."""


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def expand(p: str | Path, base: Path = HERE) -> Path:
    """`~` expands; a relative path is relative to orchestrator/ (where config.yaml lives)."""
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_impl(spec: str) -> type:
    """`package.module:Class` to the class, with a clear error."""
    mod, _, name = str(spec).partition(":")
    if not mod or not name:
        raise ConfigError(f"impl {spec!r} must look like package.module:Class")
    try:
        return getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"impl {spec!r} cannot be imported: {e}") from e


@dataclass
class Adapter:
    """One model adapter: the class that implements it plus its settings."""
    name: str
    impl: str
    settings: dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.settings.get("kind", ""))

    def build(self):
        return load_impl(self.impl)(dict(self.settings))


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    state_dir: Path
    port: int
    hosts: list[str]
    allowed_logins: list[str]
    keepalive_s: float
    resume_window_s: float
    browser_enabled: bool
    browser_port: int
    working_sound: bool
    browser_prebuilt: bool
    gate: str
    hold_poll_s: float
    busy_notice: str
    stt: Adapter
    tts: Adapter
    trim_leading_silence: bool
    keep_before_onset_ms: float
    vad: dict[str, float]
    smart_turn_enabled: bool
    smart_turn_stop_secs: float
    smart_turn_words: dict[str, Any]     # turn.smart_turn.words (turn_end.py): None values are off
    user_turn_stop_timeout_s: float
    ptt_tail_s: float
    barge_in_pause: dict[str, float]     # turn.barge_in (bargein.py); empty when the pause is off
    spaces_file: Path
    persona_file: Path
    default_space: str
    acks: dict[str, str]
    labels: dict[str, str]
    loading_notice: str
    settle_timeout_s: float
    tool_budget_calls: int               # agent.tool_budget (voice_gate.ts): conversation mode, per run; 0 = off
    tool_budget_s: float
    progress_after_s: float              # agent.progress: said once when tools run this long with nothing said
    progress_text: str
    no_answer_text: str                  # agent.no_answer: a run that called tools and said nothing
    finish_after_barge_in: bool          # agent.barge_in_finish (agent.py): finish a barged-in run silently
    finish_max_chars: int
    finish_wait_s: float
    finish_prefill_max_s: float
    pi_bin: str
    pi_agent_dir: str
    pi_models_source: Path
    pi_max_tokens: int
    pi_skill_dirs: list[Path]
    pi_extensions: dict[str, Path]
    pi_confirm_timeout_ms: int
    pi_env: dict[str, str] = field(default_factory=dict)
    latency_log: bool = True
    turn_log_dir: Path | None = None
    brain_state_dir: Path | None = None
    brain_outbox_module: Path | None = None
    speak_digests: bool = True
    digest_idle_s: float = 1.0
    heard_wait_s: float = 1.5
    tone: dict[str, Any] = field(default_factory=dict)   # tone/README.md ToneHook settings, plus package_dir
    spoken_cap_words: int = 0            # agent.spoken_cap (spoken_cap.py): a reply is said up to this many words; 0 = all
    spoken_cap_long_words: int = 0       # when the person asked for something long, or for more after a cut
    spoken_cap_offer: str = ""           # said after a reply was cut
    saved_text: str = "Saved."           # agent.saved: said once the orchestrator has saved a musing (Atlas)
    journal_join_s: float = 2.0          # agent.journal_join_s: speech this soon after a saved musing goes on with it
    echo: dict[str, Any] = field(default_factory=dict)   # echo: (echo_guard.py, turn_start.py); empty when both are off
    tts_group: dict[str, Any] = field(default_factory=dict)   # tts.group (services/tts.py): min_words (0 = off), hold_s
    record: dict[str, Any] = field(default_factory=dict)      # record: (recorder.py): enabled, dir (a Path), max_minutes

    def adapter(self, section: str, name: str) -> Adapter:
        """Another adapter of the same section (e.g. the Kokoro fallback), by name."""
        return _adapter(self.raw, section, name)


def _get(d: dict, dotted: str, typ: type | tuple[type, ...], *, required: bool = True, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if required:
                raise ConfigError(f"missing key: {dotted}")
            return default
        cur = cur[part]
    if cur is None and not required:
        return default
    if typ is float and isinstance(cur, int) and not isinstance(cur, bool):
        cur = float(cur)
    if not isinstance(cur, typ) or (typ in (int, float) and isinstance(cur, bool)):
        names = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        raise ConfigError(f"{dotted} must be {names}, got {type(cur).__name__} ({cur!r})")
    return cur


def _adapter(raw: dict, section: str, name: str) -> Adapter:
    adapters = _get(raw, f"{section}.adapters", dict)
    if name not in adapters:
        raise ConfigError(f"{section}.adapter is {name!r}, but {section}.adapters has only {', '.join(adapters)}")
    settings = adapters[name]
    if not isinstance(settings, dict) or not settings.get("impl"):
        raise ConfigError(f"{section}.adapters.{name} needs an impl (package.module:Class)")
    s = dict(settings)
    return Adapter(name=name, impl=str(s.pop("impl")), settings=s)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Read config.yaml, merge config.local.yaml beside it if present, then `overrides` (tests), and check."""
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"no config file at {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    local = path.with_name("config.local.yaml")
    if local.exists():
        raw = deep_merge(raw, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    if overrides:
        raw = deep_merge(raw, overrides)
    if raw.get("version") != 1:
        raise ConfigError(f"{path}: version must be 1, got {raw.get('version')!r}")
    base = path.resolve().parent

    stt = _adapter(raw, "stt", _get(raw, "stt.adapter", str))
    if stt.kind not in ("streaming", "segmented"):
        raise ConfigError(f"stt.adapters.{stt.name}.kind must be streaming or segmented, got {stt.kind!r}")
    tts = _adapter(raw, "tts", _get(raw, "tts.adapter", str))
    vad = _get(raw, "turn.vad", dict)
    for k in ("confidence", "start_secs", "stop_secs", "min_volume"):
        _get(raw, f"turn.vad.{k}", float)
    hosts = _get(raw, "server.hosts", list)
    for h in hosts:
        if h not in ("127.0.0.1", "::1", "tailnet"):
            raise ConfigError(f"server.hosts may hold only 127.0.0.1, ::1 and tailnet, not {h!r}")
    exts = {k: expand(v, base) for k, v in (_get(raw, "pi.extensions", dict)).items()}
    for need in ("voice_gate", "voice_mode", "hold", "today"):
        if need not in exts:
            raise ConfigError(f"pi.extensions.{need} is required (SPACES.md: the explicit -e list)")
    if any("film-rig" in str(p) for p in exts.values()):
        raise ConfigError("pi.extensions must never include film-rig.ts (a video rig's Pi extension: it hangs voice turns during a render, 05c)")

    return Config(
        raw=raw, path=path,
        state_dir=expand(_get(raw, "state_dir", str), base),
        port=_get(raw, "server.port", int),
        hosts=list(hosts),
        allowed_logins=[str(x) for x in _get(raw, "server.allowed_logins", list)],
        keepalive_s=_get(raw, "server.keepalive_s", float),
        resume_window_s=_get(raw, "server.resume_window_s", float),
        browser_enabled=_get(raw, "server.browser.enabled", bool),
        browser_port=_get(raw, "server.browser.port", int),
        working_sound=_get(raw, "server.browser.working_sound", bool),
        browser_prebuilt=_get(raw, "server.browser.prebuilt", bool, required=False, default=False),
        gate=_get(raw, "hold.gate", str).rstrip("/"),
        hold_poll_s=_get(raw, "hold.poll_s", float),
        busy_notice=_get(raw, "hold.busy_notice", str),
        stt=stt, tts=tts,
        trim_leading_silence=_get(raw, "tts.trim_leading_silence", bool),
        keep_before_onset_ms=_get(raw, "tts.keep_before_onset_ms", float),
        vad={k: float(v) for k, v in vad.items()},
        smart_turn_enabled=_get(raw, "turn.smart_turn.enabled", bool),
        smart_turn_stop_secs=_get(raw, "turn.smart_turn.stop_secs", float),
        smart_turn_words=_words(raw),
        user_turn_stop_timeout_s=_get(raw, "turn.user_turn_stop_timeout_s", float),
        ptt_tail_s=_get(raw, "turn.ptt_tail_s", float),
        barge_in_pause=_barge_in(raw),
        spaces_file=expand(_get(raw, "agent.spaces_file", str), base),
        persona_file=expand(_get(raw, "agent.persona_file", str), base),
        default_space=_get(raw, "agent.default_space", str),
        acks={str(k): str(v) for k, v in _get(raw, "agent.acks", dict).items()},
        labels={str(k): str(v) for k, v in _get(raw, "agent.labels", dict).items()},
        loading_notice=_get(raw, "agent.loading_notice", str),
        settle_timeout_s=_get(raw, "agent.settle_timeout_s", float),
        tool_budget_calls=_get(raw, "agent.tool_budget.calls", int, required=False, default=0),
        tool_budget_s=_get(raw, "agent.tool_budget.seconds", float, required=False, default=0.0),
        progress_after_s=_get(raw, "agent.progress.after_s", float, required=False, default=0.0),
        progress_text=_get(raw, "agent.progress.text", str, required=False, default=""),
        no_answer_text=_get(raw, "agent.no_answer", str, required=False, default=""),
        finish_after_barge_in=_get(raw, "agent.barge_in_finish.enabled", bool, required=False, default=False),
        finish_max_chars=_get(raw, "agent.barge_in_finish.max_chars", int, required=False, default=400),
        finish_wait_s=_get(raw, "agent.barge_in_finish.wait_s", float, required=False, default=2.0),
        finish_prefill_max_s=_get(raw, "agent.barge_in_finish.prefill_max_s", float, required=False, default=60.0),
        pi_bin=_get(raw, "pi.bin", str),
        pi_agent_dir=_get(raw, "pi.agent_dir", str),
        pi_models_source=expand(_get(raw, "pi.models_source", str), base),
        pi_max_tokens=_get(raw, "pi.max_tokens", int),
        pi_skill_dirs=[expand(p, base) for p in _get(raw, "pi.skill_dirs", list)],
        pi_extensions=exts,
        pi_confirm_timeout_ms=_get(raw, "pi.confirm_timeout_ms", int),
        pi_env={str(k): str(v) for k, v in (_get(raw, "pi.env", dict, required=False, default={}) or {}).items()},
        latency_log=_get(raw, "latency.log", bool),
        turn_log_dir=_opt_path(raw, "brain.turn_log_dir", base),
        brain_state_dir=_opt_path(raw, "brain.state_dir", base),
        brain_outbox_module=_opt_path(raw, "brain.outbox_module", base),
        speak_digests=_get(raw, "brain.speak_digests", bool, required=False, default=True),
        digest_idle_s=_get(raw, "brain.digest_idle_s", float, required=False, default=1.0),
        heard_wait_s=_get(raw, "brain.heard_wait_s", float, required=False, default=1.5),
        tone=_tone(raw, base),
        spoken_cap_words=_get(raw, "agent.spoken_cap.words", int, required=False, default=0),
        spoken_cap_long_words=_get(raw, "agent.spoken_cap.long_words", int, required=False, default=0),
        spoken_cap_offer=_get(raw, "agent.spoken_cap.offer", str, required=False, default=""),
        saved_text=_get(raw, "agent.saved", str, required=False, default="Saved."),
        journal_join_s=_get(raw, "agent.journal_join_s", float, required=False, default=2.0),
        echo=_echo(raw),
        tts_group=_tts_group(raw),
        record=_record(raw, base),
    )


def _barge_in(raw: dict) -> dict[str, float]:
    """turn.barge_in: the reply pause at the first speech frame (bargein.py). Missing or `pause: false` turns it off."""
    if not _get(raw, "turn.barge_in.pause", bool, required=False, default=False):
        return {}
    out = {}
    for k, typ in (("cue_confidence", float), ("cue_frames", int), ("resume_after_s", float), ("max_pause_s", float)):
        out[k] = _get(raw, f"turn.barge_in.{k}", typ)
    if not 0 < out["cue_confidence"] <= 1 or out["cue_frames"] < 1 or out["resume_after_s"] <= 0:
        raise ConfigError("turn.barge_in: cue_confidence in (0, 1], cue_frames >= 1, resume_after_s > 0")
    if out["max_pause_s"] < out["resume_after_s"] or out["max_pause_s"] > 5:
        raise ConfigError("turn.barge_in.max_pause_s must be at least resume_after_s and at most 5 s (Pipecat gives "
                          "up on a transport whose audio write takes 10 s)")
    return out


def _echo(raw: dict) -> dict[str, Any]:
    """echo: the agent's own speech heard back. `guard`: while the agent is busy, a transcript that is mostly what the
    agent said within `window_s` is dropped and never barges in; while it is idle the words go on, marked for the model
    and the turn log (echo_guard.py). `hold_for_words`: a VAD start while the agent is busy waits for words before it
    interrupts (turn_start.py): off, browser (the SmallWebRTC page only) or all; a bool still works, true meaning all
    (and YAML reads a bare off or on as one). `tail_s`: the agent still counts as busy this long after the bot stops
    speaking, for both (turn_start.AgentActivity). Missing, or both off: {} (Pipecat's own turn start, every
    transcript kept)."""
    e = _get(raw, "echo", dict, required=False, default={}) or {}
    hold = e.get("hold_for_words", "off")
    if isinstance(hold, bool):
        hold = "all" if hold else "off"
    if hold not in ("off", "browser", "all"):
        raise ConfigError(f"echo.hold_for_words must be off, browser or all, got {hold!r}")
    out = {"guard": bool(e.get("guard", False)), "hold_for_words": hold}
    if not (out["guard"] or hold != "off"):
        return {}
    for k, typ, default in (("window_s", float, 300.0), ("min_words", int, 3), ("min_share", float, 0.6),
                            ("fragment_sentences", int, 3), ("max_hold_s", float, 1.5), ("final_wait_s", float, 0.6),
                            ("no_words", str, "resume"), ("tail_s", float, 1.0)):
        out[k] = _get(raw, f"echo.{k}", typ, required=False, default=default)
    if out["min_words"] < 2 or not 0 < out["min_share"] <= 1 or out["window_s"] <= 0:
        raise ConfigError("echo: min_words >= 2, min_share in (0, 1], window_s > 0")
    if out["no_words"] not in ("resume", "interrupt"):
        raise ConfigError(f"echo.no_words must be resume or interrupt, got {out['no_words']!r}")
    if not 0 < out["max_hold_s"] <= 5 or not 0 <= out["final_wait_s"] <= 3:
        raise ConfigError("echo: max_hold_s in (0, 5], final_wait_s in [0, 3]")
    if not 0 <= out["tail_s"] <= 5:
        raise ConfigError("echo.tail_s must be in [0, 5] (0 = no tail)")
    return out


def _record(raw: dict, base: Path) -> dict[str, Any]:
    """record: the opt-in session recording (recorder.py; ./run.sh --record turns it on). `enabled` (default false),
    `dir` (relative to this file's folder, default ../state/recordings, local only), `max_minutes` per connection
    (default 30, at most 240: about 140 MB per 30 min of both tracks)."""
    out = {"enabled": _get(raw, "record.enabled", bool, required=False, default=False),
           "dir": expand(_get(raw, "record.dir", str, required=False, default="../state/recordings"), base),
           "max_minutes": _get(raw, "record.max_minutes", float, required=False, default=30.0)}
    if not 0 < out["max_minutes"] <= 240:
        raise ConfigError("record.max_minutes must be in (0, 240]")
    return out


def _tts_group(raw: dict) -> dict[str, Any]:
    """tts.group: a sentence shorter than `min_words` is said in one generation with the next (services/tts.py); one
    with nothing after it for `hold_s` is said alone. Missing: off (min_words 0)."""
    out = {"min_words": _get(raw, "tts.group.min_words", int, required=False, default=0),
           "hold_s": _get(raw, "tts.group.hold_s", float, required=False, default=0.5)}
    if out["min_words"] < 0 or not 0 < out["hold_s"] <= 3:
        raise ConfigError("tts.group: min_words >= 0 (0 = off), hold_s in (0, 3]")
    return out


def _tone(raw: dict, base: Path) -> dict[str, Any]:
    t = dict(_get(raw, "tone", dict, required=False, default={}) or {})
    mode = t.get("mode", "off")
    t["mode"] = {False: "off", True: "on"}.get(mode, mode) if isinstance(mode, bool) else str(mode)   # YAML's bare off
    if t["mode"] not in ("off", "log", "on"):
        raise ConfigError(f"tone.mode must be off, log or on, got {t['mode']!r}")
    t["package_dir"] = str(expand(t.get("package_dir") or "../tone", base))
    return t


def _words(raw: dict) -> dict[str, Any]:
    """turn.smart_turn.words: the transcript as a second opinion on Smart Turn (turn_end.py). Missing means off."""
    out: dict[str, Any] = {}
    for k in ("punctuation_stop_secs", "veto_wait_secs", "command_stop_secs"):
        v = _get(raw, f"turn.smart_turn.words.{k}", float, required=False, default=None)
        if v is not None and not 0 <= v <= 10:
            raise ConfigError(f"turn.smart_turn.words.{k} must be between 0 and 10 s, or null")
        out[k] = v
    out["veto_unpunctuated"] = _get(raw, "turn.smart_turn.words.veto_unpunctuated", bool, required=False, default=False)
    out["hold_dangling"] = _get(raw, "turn.smart_turn.words.hold_dangling", bool, required=False, default=False)
    out["hold_except_questions"] = _get(raw, "turn.smart_turn.words.hold_except_questions", bool, required=False,
                                        default=True)
    return out


def _opt_path(raw: dict, dotted: str, base: Path) -> Path | None:
    """An optional path; null or missing turns the feature off."""
    v = _get(raw, dotted, str, required=False, default=None)
    return expand(v, base) if v else None
