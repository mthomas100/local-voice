"""voice_gate.ts approvals (PROTOCOL.md "Approvals", 2026-10-05), model-free: a real Pi child with the gate and the stub
LLM. Every field a card shows (summary, the action's tool, effect, exact command or absolute path, cwd, space, mode,
preview or unified diff, the 4,000-character cut with its marker), the session scope and the choices offered, for kb,
bash, write and edit; a kb call kb itself would refuse is refused before anyone is asked; and what each answer does to
the run (a plain no or a silence ends it without another model call, a reply of the person's own goes on to the model).
An early version asked "May I change your knowledge base? It starts with kb new. Yes or no?"."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import EXT, HERE
from local_voice.pi_rpc import SelectRequest

pytestmark = pytest.mark.needs_pi
FAKE_KB = HERE / "tests/fixtures/fake_kb_tool.ts"
GATE = EXT / "voice_gate.ts"


def recorder(choice: str = "deny", log: list | None = None, steer: str | None = None):
    """An on_ui handler that keeps each approval (parsed) and answers with `choice` (steering `steer` in first)."""
    async def on_ui(child, ev):
        if isinstance(ev, SelectRequest):
            log.append((json.loads(ev.title), list(ev.options), ev.timeout_ms))
            if steer:
                await child.steer(steer)
        return choice
    return on_ui


async def gated(make_child, tmp_path, *, tools, tier="ask", choice="deny", log, steer=None, **env):
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    e = {"VOICE_TIER": tier, "VOICE_CONFIRM_TIMEOUT_MS": "5000", "KB_HOME": str(kb), "VOICE_SPACE": "home",
         "VOICE_SPACE_DESC": "the scratch folder", **env}
    exts = [FAKE_KB, GATE] if "kb" in tools else [GATE]
    return await make_child(tools=tools, extensions=exts, on_ui=recorder(choice, log, steer), env=e)


def call(name: str, args: dict) -> list[dict]:
    return [{"tool_calls": [{"name": name, "arguments": args}]}, {"text": "ok"}]


async def test_kb_new_says_exactly_which_page_it_creates(make_child, stub, tmp_path):
    """A kb new request, with a type kb accepts: the summary names the page and its title, the card has the exact
    command, the page's absolute path, kb's folder as cwd, and the page as kb writes it."""
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "kb"], log=log)
    args = ["new", "Analysis", "weekly-plan", "--title", "Weekly plan"]
    stub.script(call("kb", {"args": args}))
    r = await child.run_turn("add a weekly plan page")
    (a, options, timeout_ms), = log
    kb = tmp_path / "kb"
    assert a["lv"] == "approval" and a["v"] == 1
    assert a["summary"] == ('Create a new analysis page in your knowledge base titled '
                            '"Weekly plan".')
    assert a["short"] == "create that page"
    act = a["action"]
    assert act["tool"] == "kb" and act["effect"] == "create"
    assert act["command"] == ("kb new Analysis weekly-plan --title "
                              "'Weekly plan'")
    assert act["path"] == str(kb / "wiki/analyses/weekly-plan.md")
    assert act["cwd"] == str(kb) and act["space"] == "home" and act["mode"] == "conversation"
    assert "# Weekly plan\n\n## Summary\n\n## Related\n" in act["preview"]
    assert "type Analysis" in act["preview"]
    assert a["scope"] == {"key": "kb new", "label": "creating knowledge base pages"}
    assert options == ["allow_once", "allow_session", "deny"] and timeout_ms == 5000
    assert r.tool_results[0][1] is False      # answered no: nothing ran


@pytest.mark.parametrize("args,says", [
    (["new", "Issue", "weekly-plan", "--title", "x"], 'not "Issue"'),
    (["new", "Analysis", "Voice Agent", "--title", "x"], "slug must be lowercase"),
    (["new", "Analysis", "already-there"], "already exists"),
])
async def test_a_kb_new_kb_would_refuse_is_refused_without_asking(make_child, stub, tmp_path, args, says):
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "kb"], log=log)
    page = tmp_path / "kb/wiki/analyses/already-there.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("# There\n")
    stub.script(call("kb", {"args": args}))
    r = await child.run_turn("file it")
    assert log == [] and r.tool_results[0][1] is False and says in r.tool_results[0][2]


async def test_bash_commands_carry_effect_command_and_a_scope_only_when_safe(make_child, stub, tmp_path, workdir):
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "bash"], log=log)
    cases = [
        ("git commit -m 'notes'", "run", {"key": "bash git commit", "label": '"git commit" commands'}),
        ("ls -la", "run", {"key": "bash ls", "label": '"ls" commands'}),
        ("rm notes.txt", "delete", None),
        ("curl -s https://example.com", "network", None),
        ("git push origin main", "network", None),
        ("echo hello > made.txt", "run", None),               # a redirection: not one simple command
        ("sudo ls", "run", None),                             # runs another command
        ("python3 -c 'print(1)'", "run", None),               # code inline
    ]
    for command, effect, scope in cases:
        stub.script(call("bash", {"command": command}))
        await child.run_turn("run it")
        a = log[-1][0]
        assert a["action"]["command"] == command and a["action"]["effect"] == effect, command
        assert a["action"]["tool"] == "bash" and a["action"]["cwd"] == str(workdir) and a["action"]["path"] is None
        assert a["scope"] == scope, command
        assert log[-1][1] == (["allow_once", "allow_session", "deny"] if scope else ["allow_once", "deny"]), command
    by_cmd = {e[0]["action"]["command"]: e[0] for e in log}
    assert by_cmd["git commit -m 'notes'"]["summary"] == 'Run a shell command in the scratch folder, starting with "git commit".'
    assert by_cmd["ls -la"]["summary"] == 'Run the command "ls -la" in the scratch folder.'
    assert by_cmd["rm notes.txt"]["summary"] == 'Run the command "rm notes.txt" in the scratch folder. It deletes files.'
    assert by_cmd["curl -s https://example.com"]["summary"].endswith("It uses the network.")
    assert by_cmd["echo hello > made.txt"]["summary"] == 'Run a shell command in the scratch folder, starting with "echo hello".'


async def test_a_heredoc_shows_its_text_and_a_script_is_scoped_to_itself(make_child, stub, tmp_path):
    log = []
    child = await gated(make_child, tmp_path, tools=["bash"], log=log)
    command = "python3 tool.py capture --via voice <<'END'\nThe garden was quiet.\nSecond line.\nEND"
    stub.script(call("bash", {"command": command}))
    await child.run_turn("save it")
    a = log[-1][0]
    assert a["action"]["preview"] == "The garden was quiet.\nSecond line."
    assert a["action"]["command"] == command
    assert a["scope"] == {"key": "bash python3 tool.py capture", "label": '"python3 tool.py capture" commands'}
    assert a["summary"] == 'Run the command "python3 tool.py capture --via voice" in the scratch folder, giving it the text shown.'


async def test_write_names_the_file_its_folder_and_shows_the_text(make_child, stub, tmp_path, workdir):
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "write"], log=log)
    stub.script(call("write", {"path": "Hello.txt", "content": "Hello world from the voice agent\n"}))
    await child.run_turn("make a file")
    stub.script(call("write", {"path": "README.md", "content": "new readme\n"}))
    await child.run_turn("replace the readme")
    stub.script(call("write", {"path": "sub/x.md", "content": "x" * 5000}))
    await child.run_turn("a long one")
    new, old, long = (e[0] for e in log)
    assert new["summary"] == "Create a new file Hello.txt in the scratch folder."
    assert new["short"] == "create Hello.txt"
    assert new["action"] == {"tool": "write", "effect": "create", "command": None, "path": str(workdir / "Hello.txt"),
                             "cwd": str(workdir), "space": "home", "mode": "conversation",
                             "preview": "Hello world from the voice agent\n"}
    assert new["scope"] == {"key": f"write {workdir}", "label": "writing files in the scratch folder"}
    assert old["action"]["effect"] == "modify" and old["summary"].startswith("Replace everything in the file README.md")
    assert long["summary"] == "Create a new file x.md in the folder sub."
    assert long["scope"]["key"] == f"write {workdir / 'sub'}"
    preview = long["action"]["preview"]
    assert preview.startswith("x" * 4000) and preview.endswith("[cut here: 1000 more characters not shown]")
    assert not (workdir / "Hello.txt").exists()


async def test_edit_shows_a_unified_diff_of_the_file_as_it_will_be(make_child, stub, tmp_path, workdir):
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "edit"], log=log)
    stub.script(call("edit", {"path": "README.md", "edits": [{"oldText": "Line two.", "newText": "Line 2, changed."}]}))
    await child.run_turn("change it")
    stub.script(call("edit", {"path": "README.md", "oldText": "not in the file", "newText": "x"}))
    await child.run_turn("change something missing")
    diff, missing = (e[0] for e in log)
    assert diff["summary"] == "Make one change to the file README.md in the scratch folder."
    assert diff["action"]["effect"] == "modify" and diff["action"]["path"] == str(workdir / "README.md")
    p = diff["action"]["preview"]
    assert p.startswith("--- README.md\n+++ README.md\n@@") and "-Line two.\n+Line 2, changed.\n" in p and " Line three.\n" in p
    assert missing["action"]["preview"].startswith("(the changes as asked for")
    assert "--- replace:\nnot in the file\n+++ with:\nx" in missing["action"]["preview"]
    assert (workdir / "README.md").read_text() == "Scratch README.\nLine two.\nLine three.\n"


async def test_act_mode_is_named_in_the_action(make_child, stub, tmp_path):
    log = []
    child = await gated(make_child, tmp_path, tools=["read", "bash"], log=log, VOICE_TOOLS="read")
    stub.script(call("bash", {"command": "ls"}))
    await child.run_turn("list")
    assert log[-1][0]["action"]["mode"] == "act"     # an active tool beyond VOICE_TOOLS: act mode


@pytest.mark.parametrize("choice,ran,model_calls", [
    ("allow_once", True, 2), ("allow_session", True, 2),
    ("deny", False, 1), ("timeout", False, 1),          # ended without another model call: the voice said what was not done
])
async def test_each_answer_runs_or_ends_the_call(make_child, stub, tmp_path, workdir, choice, ran, model_calls):
    log = []
    child = await gated(make_child, tmp_path, tools=["bash"], choice=choice, log=log)
    stub.script(call("bash", {"command": "touch made.txt"}))
    n0 = len(stub.requests())
    r = await child.run_turn("make it")
    assert r.tool_results[0][1] is ran and (workdir / "made.txt").exists() is ran
    assert len(stub.requests()) - n0 == model_calls
    if not ran:
        assert "did not approve" in r.tool_results[0][2] and "have been told so" in r.tool_results[0][2]


async def test_a_reply_of_their_own_goes_on_to_the_model_after_the_refusal(make_child, stub, tmp_path, workdir):
    """deny_said: "no, put it in the inbox instead" is not approval; the person's words reach the model in the same run,
    after the refused call, the way a coding agent's "no, and tell it what to do instead" does."""
    log = []
    child = await gated(make_child, tmp_path, tools=["bash"], choice="deny_said", log=log,
                        steer="No, put it in the inbox instead.")
    stub.script(call("bash", {"command": "touch made.txt"}))
    r = await child.run_turn("make it")
    assert r.tool_results[0][1] is False and not (workdir / "made.txt").exists()
    msgs = stub.requests()[-1]["body"]["messages"]
    tool_i = max(i for i, m in enumerate(msgs) if m["role"] == "tool")
    assert "said something else instead" in str(msgs[tool_i]["content"])
    later = [m for m in msgs[tool_i + 1:] if m["role"] == "user"]
    assert later and "No, put it in the inbox instead." in json.dumps(later[-1]["content"])
