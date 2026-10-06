"""The Pi runner against the stub LLM, the model choice, the kb writer on a kb clone, the Atlas writer on a clone."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import synth
from conftest import write_agent_dir

from local_voice_brain import atlaswriter, kbwriter, turnlog
from local_voice_brain.llm import LLMError, PiRunner, choose_model, effective_compat
from local_voice_brain.validate import Fact, Quote, Reflection

PI = Path(shutil.which("pi") or "/nonexistent/pi")


@pytest.mark.needs_pi
def test_pi_runner_returns_the_answer_session_and_model(tmp_path, stub, agent_dir):
    stub.script([{"text": '{"facts": [], "spoken": "NO_REPLY"}', "chunk_words": 2, "reasoning": "secret thoughts"}])
    sp = tmp_path / "system.md"
    sp.write_text("You are a test.")
    r = PiRunner(PI, agent_dir=agent_dir, session_dir=tmp_path / "sessions", timeout_s=60)
    res = r.run(model="local/qwen38", task="Reply.", stdin_text="TURNS\nline", session_id="voice-brain-t-1",
                system_prompt_file=sp, cwd=tmp_path)
    assert res.text == '{"facts": [], "spoken": "NO_REPLY"}'      # the thinking never leaks into the answer
    assert res.session_id == "voice-brain-t-1" and res.session_file and res.session_file.exists()
    assert (res.model, res.provider, res.actor) == ("qwen38", "local", "pi/qwen38")
    body = stub.requests()[-1]["body"]
    assert body["model"] == "qwen38" and body.get("tools") in (None, [])        # no tools offered to the model
    assert body["messages"][0]["content"].startswith("You are a test.")
    user = body["messages"][-1]["content"]
    user = user if isinstance(user, str) else "".join(p.get("text", "") for p in user)
    assert user.startswith("TURNS\nline") and "\n\nReply." in user
    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.needs_pi
def test_pi_runner_raises_on_a_provider_error(tmp_path, stub, agent_dir):
    stub.script([{"status": 400, "error_message": "bad request"}] * 4)
    r = PiRunner(PI, agent_dir=agent_dir, session_dir=tmp_path / "s", timeout_s=60)
    with pytest.raises(LLMError):
        r.run(model="local/qwen38", task="Reply.", cwd=tmp_path)
    with pytest.raises(LLMError, match="pi not found"):
        PiRunner(tmp_path / "nope", agent_dir=None, session_dir=tmp_path, timeout_s=1).run(model="m", task="t",
                                                                                          cwd=tmp_path)


def test_choose_model_falls_back_while_qwen27_lacks_the_chat_template_compat(tmp_path):
    d = write_agent_dir(tmp_path / "a", "http://127.0.0.1:1/v1")
    m, why = choose_model("local/qwen27-262k", "local/qwen38", d)
    assert m == "local/qwen38" and "qwen-chat-template" in why and "'deepseek'" in why
    d2 = write_agent_dir(tmp_path / "b", "http://127.0.0.1:1/v1", qwen27_compat={"thinkingFormat": "qwen-chat-template"})
    assert choose_model("local/qwen27-262k", "local/qwen38", d2) == ("local/qwen27-262k", "")
    assert choose_model("local/qwen38", "local/x", d) == ("local/qwen38", "")
    assert effective_compat(json.loads((d2 / "models.json").read_text()), "local/qwen27-262k")["supportsStore"] is False


def test_choose_model_reads_the_real_registry_without_writing_it():
    """What the job would pick on this Mac today (read only)."""
    m, why = choose_model("local/qwen27-262k", "local/qwen38", None)
    assert m in ("local/qwen27-262k", "local/qwen38")
    print(f"\nthis Mac today: {m} {why}")


@pytest.mark.needs_pi
@pytest.mark.needs_kb
def test_kb_notes_land_on_the_digest_kb_ts_registered(tmp_path, stub, agent_dir, kb_clone):
    stub.script([{"text": "{}"}])
    r = PiRunner(PI, agent_dir=agent_dir, session_dir=tmp_path / "s", timeout_s=60)
    res = r.run(model="local/qwen38", task="Reply.", stdin_text="voice turns", session_id="voice-brain-2026-10-04-tech-t1",
                cwd=kb_clone, extensions=[kb_clone / "adapters/pi/kb.ts"], env_extra={"KB_HOME": str(kb_clone)})
    assert not list((kb_clone / ".sessions" / "live").glob("*.json"))          # kb.ts closed the session
    synth.write_day(tmp_path / "turns", None)
    turns, _ = turnlog.read_day(tmp_path / "turns", synth.DAY)
    handles = {"S1T2": turns[1]}
    facts = [Fact(text="Kokoro is the fallback voice.", kind="decision", cites=["S1T2"], quote="keep kokoro")]
    out = kbwriter.note_facts(kb_clone / "bin/kb", kb_clone, session_id=f"pi:{res.session_id}", actor=res.actor,
                              day=synth.DAY, facts=facts, handles=handles)
    digest = kbwriter.digest_path(kb_clone, f"pi:{res.session_id}")
    text = digest.read_text()
    assert out.digest.endswith(digest.name)
    assert "pi/qwen38: voice 2026-10-04 · decision: Kokoro is the fallback voice. (voice session s-20261004-0914-a1 " \
           "turn 2, 09:16, space home) — \"keep kokoro\"" in text
    assert "actor: pi/qwen38" in text and f"session_id: pi:{res.session_id}" in text
    status = subprocess.run(["git", "-C", str(kb_clone), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert status.strip().splitlines() == ["?? .sessions/digests/2026/10/"]      # nothing but the digest changed
    with pytest.raises(kbwriter.KbError):
        kbwriter.note_facts(kb_clone / "bin/kb", kb_clone, session_id="pi:never-started", actor="pi/qwen38",
                            day=synth.DAY, facts=facts, handles=handles)


def _reflections(turns):
    c1, c2 = [t for t in turns if t.space == "atlas"][:2]
    handles = {"S1T1": c1, "S1T2": c2}
    return handles, [Reflection(title="Saying yes, and a quiet head",
                                quotes=[Quote("S1T1", "I keep saying yes to things before I've even thought about them"),
                                        Quote("S1T2", "the first time all week my head felt quiet")],
                                notice="You talked about work once and about the river walk once.",
                                questions=["What was different about the walk?"], tags=["area/work"])]


@pytest.mark.needs_atlas
def test_atlas_page_commits_only_itself_and_leaves_the_persons_edits_alone(tmp_path, atlas_clone):
    synth.capture_into_atlas(atlas_clone)
    synth.write_day(tmp_path / "turns", atlas_clone)
    turns, _ = turnlog.read_day(tmp_path / "turns", synth.DAY)
    for t in turns:
        if t.atlas:
            words, why = atlaswriter.saved_words(atlas_clone, t)
            assert words == t.captured_text, why
    # the person is mid-edit in Obsidian: an uncommitted change and a new untracked note
    note = atlas_clone / "journal" / f"{synth.DAY}.md"
    note.write_text(note.read_text() + "\nhalf a thought, not saved yet")
    (atlas_clone / "journal" / "ideas-draft.md").write_text("untracked\n")
    before_head = subprocess.run(["git", "-C", str(atlas_clone), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
    handles, refl = _reflections(turns)
    res = atlaswriter.write(atlas_clone, day=synth.DAY, reflections=refl, handles=handles, by="voice-brain")
    assert res.page == f"inbox/voice-reflections-{synth.DAY}.md" and res.queue_lists_it and not res.warnings
    git = lambda *a: subprocess.run(["git", "-C", str(atlas_clone), *a], capture_output=True, text=True).stdout  # noqa: E731
    assert git("rev-parse", "HEAD~1").strip() == before_head
    assert git("show", "--name-only", "--format=", "HEAD").split() == [res.page]
    assert sorted(git("status", "--porcelain").splitlines()) == [f" M journal/{synth.DAY}.md",
                                                                 "?? journal/ideas-draft.md"]
    assert note.read_text().endswith("half a thought, not saved yet")
    page = (atlas_clone / res.page).read_text()
    atlas = atlaswriter.load_module(atlas_clone)
    lines, body = atlas.split(page)
    meta = atlas.parse(lines)
    assert meta["by"] == "voice-brain" and not atlas.human(meta) and meta["type"] == "Analysis"
    assert meta["tags"] == ["digest", "voice", "area/work"] and meta["related"] == [f"[[{synth.DAY}]]"]
    assert "> I keep saying yes to things before I've even thought about them\n> — [[2026-10-04]], 21:02, said to voice" in body
    for q in refl[0].quotes:
        assert q.text in note.read_text()                       # every quote is in the person's note, verbatim
    # a second run of the same day finds its own page and writes nothing
    again = atlaswriter.write(atlas_clone, day=synth.DAY, reflections=refl, handles=handles, by="voice-brain")
    assert again.commit == "" and git("rev-parse", "HEAD~1").strip() == before_head


@pytest.mark.needs_atlas
def test_atlas_writer_waits_while_git_is_busy_and_checks_the_words(tmp_path, atlas_clone):
    synth.capture_into_atlas(atlas_clone)
    synth.write_day(tmp_path / "turns", atlas_clone)
    turns, _ = turnlog.read_day(tmp_path / "turns", synth.DAY)
    handles, refl = _reflections(turns)
    (atlas_clone / ".git" / "index.lock").write_text("")
    with pytest.raises(atlaswriter.AtlasBusy):
        atlaswriter.write(atlas_clone, day=synth.DAY, reflections=refl, handles=handles, by="voice-brain")
    (atlas_clone / ".git" / "index.lock").unlink()
    # the person edited their entry since: the old words are no longer quotable, and nothing is "fixed"
    note = atlas_clone / "journal" / f"{synth.DAY}.md"
    note.write_text(note.read_text().replace("stretched thin", "stretched"))
    c1 = handles["S1T1"]
    words, why = atlaswriter.saved_words(atlas_clone, c1)
    assert words is None and "not found verbatim" in why
    other = synth.turn("s", 1, "2026-10-04T21:00:00-07:00", None, "atlas", "x", "y",
                       atlas={"root": str(tmp_path / "elsewhere"), "path": "journal/x.md"})
    assert "another root" in atlaswriter.saved_words(atlas_clone, turnlog.parse_record(other))[1]
    escape = synth.turn("s", 1, "2026-10-04T21:00:00-07:00", None, "atlas", "x", "y",
                        atlas={"root": str(atlas_clone), "path": "../../etc/passwd"})
    assert "leaves the root" in atlaswriter.saved_words(atlas_clone, turnlog.parse_record(escape))[1]
