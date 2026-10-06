"""The heartbeat: find a finished day nobody has digested, pass the idle gates, reflect, write, queue.

launchd starts this every 30 minutes (modelled on OpenClaw's heartbeat). Most heartbeats end at the first step:
nothing is due, so no gate is even checked and no model is touched. A due day is digested only when the Mac is idle
(no recent voice turn, nobody at the keyboard, no turn under way) and gpu_clear.sh says CLEAR right before each model
call; otherwise the heartbeat defers and the next one tries again (Hermes' deferred review).

One day per heartbeat: the day's technical pass (kb), then its life pass (Atlas), each a separate Pi run, then the
writes, then the spoken digest for the next conversation, or nothing (NO_REPLY).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from . import atlaswriter, gates, kbwriter, outbox, prompts, state, turnlog, validate
from .config import Config
from .llm import LLMError, PiResult, PiRunner, choose_model
from .report import write_report

EX_OK, EX_FAILED, EX_TEMPFAIL = 0, 1, 75


@dataclass
class Outcome:
    kind: str                 # idle | deferred | locked | done | failed
    detail: str
    day: str | None = None
    code: int = EX_OK


@dataclass
class DayRun:
    """Everything one digest produced, for the report and the tests."""
    day: str
    turns: list[turnlog.Turn]
    problems: list[str]
    model: str = ""
    model_note: str = ""
    tech: validate.TechProposal | None = None
    tech_handles: dict[str, turnlog.Turn] = field(default_factory=dict)
    tech_result: PiResult | None = None
    tech_session: str = ""
    tech_kb: bool = False             # the technical pass ran with kb.ts, so its kb session exists
    tech_actor: str = ""              # pi/<model> that wrote the facts: the kb notes' actor
    life: validate.LifeProposal | None = None
    life_handles: dict[str, turnlog.Turn] = field(default_factory=dict)
    life_result: PiResult | None = None
    life_skipped: list[str] = field(default_factory=list)
    kb: kbwriter.KbResult | None = None
    kb_digest: str = ""
    atlas: atlaswriter.AtlasResult | None = None
    spoken: str | None = None
    outbox_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    gate_log: list[str] = field(default_factory=list)


def today(now: datetime) -> str:
    return now.date().isoformat()


def due_day(cfg: Config, ledger: state.Ledger, now: datetime, *, dry_run: bool = False) -> str | None:
    """The oldest finished day (before today) with turns that is neither done, failed nor skipped. Days past the
    backlog window are marked skipped instead: a week-old spoken digest helps nobody."""
    cutoff = (now.date() - timedelta(days=cfg.max_backlog_days)).isoformat()
    for day in turnlog.days_available(cfg.turns_dir):
        if day >= today(now):
            break
        st = ledger.status(day)
        if st in (state.DONE, state.FAILED, state.SKIPPED):
            continue
        if day < cutoff:
            if not dry_run:
                ledger.day(day).update(status=state.SKIPPED, last_error=f"older than {cfg.max_backlog_days} days")
                ledger.save()
            continue
        return day
    return None


def heartbeat(cfg: Config, *, now: datetime | None = None, day: str | None = None, force: bool = False,
              dry_run: bool = False, log: Callable[[str], None] = print,
              sleep: Callable[[float], None] | None = None) -> Outcome:
    now = now or datetime.now().astimezone()
    with state.Lock(cfg.state_dir / "brain.lock") as got:
        if not got:
            out = Outcome("locked", "another heartbeat is running", code=EX_TEMPFAIL)
        else:
            out = _heartbeat(cfg, now=now, day=day, force=force, dry_run=dry_run, log=log, sleep=sleep)
    if not dry_run:
        state.log_heartbeat(cfg.state_dir, now, out.kind, (f"day={out.day} " if out.day else "") + out.detail)
    return out


def _heartbeat(cfg: Config, *, now: datetime, day: str | None, force: bool, dry_run: bool,
               log: Callable[[str], None], sleep: Callable[[float], None] | None) -> Outcome:
    ledger = state.Ledger.load(cfg.state_dir)
    if day is None:
        day = due_day(cfg, ledger, now, dry_run=dry_run)
        if day is None:
            return Outcome("idle", "nothing to digest")
    elif not turnlog.day_file(cfg.turns_dir, day).exists():
        return Outcome("failed", f"no turn log for {day}", day=day, code=EX_FAILED)
    checks = [] if force else [gates.quiet(cfg.turns_dir, now, cfg.quiet_minutes),
                               gates.user_idle(cfg.user_idle_minutes),
                               gates.orchestrator(cfg.orchestrator_status, cfg.busy_states)]
    for g in checks:
        log(str(g))
        if not g.ok:
            return Outcome("deferred", str(g), day=day, code=EX_TEMPFAIL)
    g = gates.gpu_clear(cfg.gpu_clear)   # never skipped, not even with --force
    log(str(g))
    if not g.ok:
        return Outcome("deferred", str(g), day=day, code=EX_TEMPFAIL)
    entry = ledger.day(day)
    if not dry_run:
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        ledger.save()
    try:
        run = digest_day(cfg, day, now=now, dry_run=dry_run, entry=entry, log=log, sleep=sleep)
    except Exception as e:  # noqa: BLE001 (the job must record any failure and keep the heartbeat alive)
        if dry_run:
            raise
        failed = int(entry["attempts"]) >= cfg.max_attempts
        entry.update(status=state.FAILED if failed else state.RETRY, last_error=f"{type(e).__name__}: {e}",
                     updated=now.isoformat(timespec="seconds"))
        ledger.save()
        return Outcome("failed", f"{type(e).__name__}: {e}" + (" (gave up; `run.sh retry` resets)" if failed else ""),
                       day=day, code=EX_FAILED)
    if not dry_run:
        entry.update(status=state.DONE, last_error=None, updated=now.isoformat(timespec="seconds"))
        ledger.save()
    said = f"spoken digest queued: {run.spoken!r}" if run.spoken else "NO_REPLY"
    return Outcome("done", f"{len(run.tech.facts) if run.tech else 0} facts, "
                           f"{len(run.life.reflections) if run.life else 0} reflections, {said}", day=day)


class _Asker:
    """The model, asked through Pi: chosen once per attempt, gpu_clear right before every call, one retry when a
    reply is not JSON, and the live brain reloaded at the end if a pass ran on another model."""

    def __init__(self, cfg: Config, run: DayRun, *, log: Callable[[str], None],
                 sleep: Callable[[float], None] | None):
        self.cfg, self.run, self.log, self.sleep = cfg, run, log, sleep
        self.runner: PiRunner | None = None
        self.first = True          # the heartbeat checked gpu_clear a moment before the first call
        self.asked = False

    def _setup(self) -> PiRunner:
        if self.runner is None:
            self.runner = PiRunner(self.cfg.pi_bin, agent_dir=self.cfg.agent_dir, session_dir=self.cfg.session_dir,
                                   timeout_s=self.cfg.timeout_s)
            self.runner.check()
            self.run.model, self.run.model_note = choose_model(self.cfg.model, self.cfg.fallback_model,
                                                               self.cfg.agent_dir)
            if self.run.model_note:
                self.log(f"model: {self.run.model_note}")
        return self.runner

    def gpu(self) -> None:
        """Later calls wait out our own previous request: gpu_clear cannot tell whose POST it saw."""
        if self.first:
            self.first = False
            return
        kw = {"sleep": self.sleep} if self.sleep else {}
        g = gates.wait_gpu_clear(self.cfg.gpu_clear, self.cfg.gpu_wait_s, self.cfg.gpu_poll_s, log=self.log, **kw)
        self.log(str(g))
        if not g.ok:
            raise LLMError(f"the GPU did not clear between passes: {g.detail}")

    def ask(self, *, task: str, text: str, sid: str, system_file: Path, cwd: Path, extensions: list[Path],
            env: dict[str, str]) -> tuple[PiResult, dict]:
        runner = self._setup()
        self.gpu()
        res = runner.run(model=self.run.model, task=task, stdin_text=text, session_id=sid,
                         system_prompt_file=system_file, cwd=cwd, extensions=extensions, env_extra=env)
        self.asked = True
        try:
            return res, validate.parse_json_reply(res.text)
        except ValueError:
            self.gpu()
            res = runner.run(model=self.run.model, task=task + " Your previous reply was not valid JSON: reply with "
                             "the JSON object only.", stdin_text=text, session_id=sid + "-r2",
                             system_prompt_file=system_file, cwd=cwd, extensions=extensions, env_extra=env)
            return res, validate.parse_json_reply(res.text)

    def restore(self, cwd: Path) -> None:
        """Leave the rig on its live brain, so the next conversation does not pay a reload."""
        if not self.asked or self.run.model == self.cfg.restore_model or self.runner is None:
            return
        try:
            self.gpu()
            self.runner.run(model=self.cfg.restore_model, task="Reply with the single word OK.", cwd=cwd)
        except LLMError as e:
            self.run.warnings.append(f"could not reload {self.cfg.restore_model}: {e}")


def digest_day(cfg: Config, day: str, *, now: datetime, dry_run: bool = False, entry: dict | None = None,
               log: Callable[[str], None] = print, sleep: Callable[[float], None] | None = None) -> DayRun:
    entry = entry if entry is not None else {"sinks": {}}
    sinks = entry.setdefault("sinks", {})
    turns, problems = turnlog.read_day(cfg.turns_dir, day)
    run = DayRun(day=day, turns=turns, problems=problems)
    ddir = state.day_dir(cfg.state_dir, day)
    ddir.mkdir(parents=True, exist_ok=True)
    tech_turns = [t for t in turns if t.space != cfg.atlas_space]
    atlas_turns = [t for t in turns if t.space == cfg.atlas_space]
    input_hash = turnlog.file_digest(cfg.turns_dir, day)

    # Each pass's validated proposal is saved as soon as it exists: a retry asks the model only for what is missing
    # (a render that takes the GPU between the passes costs the second pass, not both), then retries failed writes.
    saved = None if dry_run else state.load_proposal(cfg.state_dir, day)
    if not saved or saved.get("input_hash") != input_hash:
        saved = {"input_hash": input_hash}
    asker = _Asker(cfg, run, log=log, sleep=sleep)
    tag = secrets.token_hex(2)
    if "tech" in saved:
        _load_tech(run, saved, turns)
    else:
        if tech_turns:
            _tech_pass(cfg, run, day, ddir, tech_turns, asker, tag=tag, dry_run=dry_run)
        saved.update(model=run.model, model_note=run.model_note, tech=run.tech and {
            **state.tech_to_json(run.tech), "handles": state.handle_map(run.tech_handles),
            "session": run.tech_session, "kb_session_registered": run.tech_kb, "actor": run.tech_actor})
        if not dry_run:
            state.save_proposal(cfg.state_dir, day, saved)
    if "life" in saved:
        _load_life(run, saved, turns)
    else:
        if atlas_turns and cfg.atlas_enabled:
            _life_pass(cfg, run, day, ddir, atlas_turns, asker, tag=tag)
        saved.update(life=run.life and {**state.life_to_json(run.life), "handles": state.handle_map(run.life_handles)},
                     life_skipped=run.life_skipped)
        if asker.asked:
            saved.update(model=run.model, model_note=run.model_note)
        if not dry_run:
            state.save_proposal(cfg.state_dir, day, saved)
    run.model = run.model or saved.get("model", "")
    run.model_note = run.model_note or saved.get("model_note", "")
    asker.restore(ddir)

    # --- writes: kb notes, the Atlas page, the spoken digest ---
    facts = run.tech.facts if run.tech else []
    if facts and cfg.kb_enabled and sinks.get("kb", {}).get("status") != state.DONE:
        if dry_run:
            run.warnings.append(f"dry run: {len(facts)} kb notes not written")
        elif not run.tech_kb:
            run.warnings.append(f"{len(facts)} facts not written to the kb: the technical pass had no kb session")
        else:
            done = int(sinks.get("kb", {}).get("notes", 0))
            try:
                run.kb = kbwriter.note_facts(cfg.kb_bin, cfg.kb_home, session_id=run.tech_session,
                                             actor=run.tech_actor, day=day, facts=facts, handles=run.tech_handles,
                                             skip=done)
            except kbwriter.KbError as e:
                sinks["kb"] = {"status": "partial", "notes": done + e.written}   # a retry resumes after these
                raise
            p = kbwriter.digest_path(cfg.kb_home, run.tech_session)
            run.kb_digest = str(p.relative_to(cfg.kb_home)) if p else run.kb.digest.lstrip("/")
            sinks["kb"] = {"status": state.DONE, "notes": len(facts), "digest": run.kb_digest}
    elif sinks.get("kb"):
        run.kb_digest = sinks["kb"].get("digest", "")
    reflections = run.life.reflections if run.life else []
    if reflections and cfg.atlas_enabled and sinks.get("atlas", {}).get("status") != state.DONE:
        run.atlas = atlaswriter.write(cfg.atlas_root, day=day, reflections=reflections, handles=run.life_handles,
                                      by=cfg.atlas_by, dry_run=dry_run)
        run.warnings += run.atlas.warnings
        if not dry_run:
            sinks["atlas"] = {"status": state.DONE, "page": run.atlas.page, "commit": run.atlas.commit}
    run.spoken = compose_spoken(day, len({t.session for t in turns}), run.tech.spoken if run.tech else None,
                                len(reflections) if (cfg.atlas_enabled and (run.atlas or sinks.get("atlas"))) else 0)
    if run.spoken and not dry_run and sinks.get("outbox", {}).get("status") != state.DONE:
        cites = [{"session": run.tech_handles[h].session, "turn": run.tech_handles[h].turn}
                 for f in facts for h in f.cites]
        run.outbox_path = outbox.queue(cfg.state_dir, day=day, text=run.spoken, now=now,
                                       expiry_days=cfg.outbox_expiry_days, cites=cites, kb_digest=run.kb_digest or None,
                                       atlas_page=(run.atlas.page if run.atlas else sinks.get("atlas", {}).get("page")))
        sinks["outbox"] = {"status": state.DONE, "path": str(run.outbox_path)}
    write_report(cfg, run, now=now, path=ddir / ("report.dry-run.md" if dry_run else "report.md"))
    return run


def _load_tech(run: DayRun, saved: dict, turns: list[turnlog.Turn]) -> None:
    t = saved.get("tech")
    if t:
        run.tech = state.tech_from_json(t)
        run.tech_handles = state.resolve_handles(t["handles"], turns)
        run.tech_session, run.tech_kb = t["session"], bool(t.get("kb_session_registered"))
        run.tech_actor = t.get("actor", "")


def _load_life(run: DayRun, saved: dict, turns: list[turnlog.Turn]) -> None:
    life = saved.get("life")
    if life:
        run.life = state.life_from_json(life)
        run.life_handles = state.resolve_handles(life["handles"], turns)
    run.life_skipped = list(saved.get("life_skipped", []))


def _tech_pass(cfg: Config, run: DayRun, day: str, ddir: Path, tech_turns: list[turnlog.Turn], asker: _Asker, *,
               tag: str, dry_run: bool) -> None:
    shown = prompts.render_tech(day, tech_turns, cfg.max_prompt_chars)
    (ddir / "tech.system.md").write_text(prompts.TECH_SYSTEM, encoding="utf-8")
    (ddir / "tech.input.md").write_text(shown.text, encoding="utf-8")
    # kb.ts registers the run as a kb session; a dry run must leave the kb untouched, so it runs without it.
    kb_ok = cfg.kb_enabled and cfg.kb_pi_adapter.is_file() and not dry_run
    if cfg.kb_enabled and not dry_run and not kb_ok:
        run.warnings.append(f"kb adapter missing at {cfg.kb_pi_adapter}: facts will not reach the kb")
    res, obj = asker.ask(task=prompts.TECH_TASK, text=shown.text, sid=f"voice-brain-{day}-tech-{tag}",
                         system_file=ddir / "tech.system.md", cwd=cfg.kb_home if kb_ok else ddir,
                         extensions=[cfg.kb_pi_adapter] if kb_ok else [],
                         env={"KB_HOME": str(cfg.kb_home)} if kb_ok else {})
    (ddir / "tech.reply.txt").write_text(res.text, encoding="utf-8")
    run.tech_result, run.tech_session, run.tech_actor, run.tech_kb = res, f"pi:{res.session_id}", res.actor, kb_ok
    run.tech_handles = shown.handles
    run.tech = validate.validate_tech(obj, shown.handles, max_facts=cfg.max_facts, max_words=cfg.spoken_max_words)
    if shown.omitted:
        run.warnings.append(f"technical pass: {shown.omitted} earlier turns omitted to fit the prompt")


def _life_pass(cfg: Config, run: DayRun, day: str, ddir: Path, atlas_turns: list[turnlog.Turn], asker: _Asker, *,
               tag: str) -> None:
    atlaswriter.check_root(cfg.atlas_root)
    quotable = []
    for t in atlas_turns:
        w, why = atlaswriter.saved_words(cfg.atlas_root, t)
        if w is None:
            if why != "not captured":
                run.life_skipped.append(f"{t.session} turn {t.turn}: {why}")
            continue
        quotable.append((t, w))
    if not quotable:
        return
    shown = prompts.render_life(day, quotable, cfg.max_prompt_chars)
    words = {h: next(w for t2, w in quotable if t2 is t) for h, t in shown.handles.items()}
    tags = atlaswriter.known_tags(cfg.atlas_root)
    system = prompts.LIFE_SYSTEM.format(max_reflections=cfg.max_reflections, tags=", ".join(sorted(tags)) or "none yet")
    (ddir / "life.system.md").write_text(system, encoding="utf-8")
    (ddir / "life.input.md").write_text(shown.text, encoding="utf-8")
    res, obj = asker.ask(task=prompts.LIFE_TASK, text=shown.text, sid=f"voice-brain-{day}-life-{tag}",
                         system_file=ddir / "life.system.md", cwd=ddir, extensions=[], env={})
    (ddir / "life.reply.txt").write_text(res.text, encoding="utf-8")
    run.life_result, run.life_handles = res, shown.handles
    run.life = validate.validate_life(obj, shown.handles, words, known_tags=tags, max_reflections=cfg.max_reflections)


def compose_spoken(day: str, n_sessions: int, tech_spoken: str | None, n_reflections: int) -> str | None:
    """The spoken digest, or None for NO_REPLY. Life content is never spoken: only that a reflection is waiting."""
    parts = [tech_spoken] if tech_spoken else []
    if n_reflections:
        what = "a reflection" if n_reflections == 1 else f"{n_reflections} reflections"
        parts.append(f"I left {what} from your journal talk in your Atlas inbox.")
    if not parts:
        return None
    convs = "conversation" if n_sessions == 1 else "conversations"
    return f"From {date.fromisoformat(day).strftime('%A')}'s voice {convs}: " + " ".join(parts)
