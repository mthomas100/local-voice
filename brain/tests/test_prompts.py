"""How a day is shown to the model: handles, the heard text of an interrupted reply, and the budget."""
from __future__ import annotations

from datetime import datetime, timedelta

from local_voice_brain import prompts
from local_voice_brain.turnlog import Turn

T0 = datetime.fromisoformat("2026-10-04T09:00:00-07:00")


def mk(session, n, user, reply, minutes):
    return Turn(session=session, turn=n, t_start=T0 + timedelta(minutes=minutes), t_end=None, space="home",
                mode="conversation", client="test", input="voice", user_text=user, reply_text=reply, heard_text=None,
                interrupted=False, tools=(("kb", True),), atlas=None, raw={})


def test_handles_number_sessions_in_order_of_appearance():
    turns = [mk("b", 1, "u", "r", 0), mk("a", 1, "u", "r", 1), mk("b", 2, "u", "r", 2)]
    assert {h: (t.session, t.turn) for h, t in prompts.handles_for(turns).items()} == {
        "S1T1": ("b", 1), "S2T1": ("a", 1), "S1T2": ("b", 2)}


def test_a_long_day_shrinks_replies_first_then_drops_the_oldest_turns():
    turns = [mk("s", n, f"question number {n} " + "word " * 40, "answer " * 900, n) for n in range(1, 41)]
    full = prompts.render_tech("2026-10-04", turns, 10**7)
    assert full.omitted == 0 and len(full.text) > 150_000 and "…\nTOOLS" in full.text   # replies start capped at 4000 chars
    fit = prompts.render_tech("2026-10-04", turns, 60_000)
    assert len(fit.text) <= 60_000 and fit.omitted == 0                      # replies shortened, every turn kept
    assert "question number 1 " in fit.text and "…" in fit.text
    tight = prompts.render_tech("2026-10-04", turns, 8_000)
    assert len(tight.text) <= 8_000 + 100 and tight.omitted > 0
    assert "question number 40 " in tight.text and "[S1T1]" not in tight.text  # the newest turns survive
    assert set(tight.handles) == {h for h in prompts.handles_for(turns) if f"[{h}]" in tight.text}
    assert tight.text.startswith(f"({tight.omitted} earlier turns omitted to fit.)")


def test_the_life_input_holds_only_their_saved_words():
    t = mk("c", 1, "note this: the river was quiet", "Saved. Lovely.", 0)
    shown = prompts.render_life("2026-10-04", [(t, "the river was quiet")], 10_000)
    assert "the river was quiet" in shown.text and "note this" not in shown.text and "Saved" not in shown.text
    assert list(shown.handles) == ["S1T1"]
