"""Small text rules the voice layer applies itself: spoken yes/no, how a permission question is said aloud, and
what counts as the loading banner. Deterministic on purpose: an LLM in the final control path of a permission is
what the Home Assistant consensus warns against."""
from __future__ import annotations

import re

YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "please do", "do it", "go ahead", "go for it",
       "absolutely", "of course", "affirmative", "correct", "yes please", "sure thing", "fine", "alright", "all right"}
NO = {"no", "nope", "nah", "don't", "do not", "stop", "cancel", "never mind", "nevermind", "no thanks", "no thank you",
      "negative", "not now", "skip it", "leave it", "wait"}
_WORDS = re.compile(r"[a-z']+")

# llama-swap streams its loading banner as reasoning_content while a model loads (14-20 s, note 05d)
LOADING_BANNER = "llama-swap loading model"


def yes_no(text: str) -> bool | None:
    """True for a plain yes, False for a plain no, None when it is neither (then the question counts as unanswered,
    which voice_gate.ts treats as no). Only the first few words decide: "yes, and also..." is a yes."""
    words = _WORDS.findall(text.lower().replace("’", "'"))
    if not words:
        return None
    for n in (3, 2, 1):
        head = " ".join(words[:n])
        if head in NO:
            return False
        if head in YES:
            return True
    return None


_SHELL = re.compile(r"[<>|&;$`\\(){}\[\]*?~\"']")


_FILE_ACT = re.compile(r"^(write|edit) (.+)$")


def spoken_confirm(title: str, message: str) -> str:
    """The permission question as said aloud: the gate's title, the detail only when it reads as words, then the ask.
    A write or edit is asked about by the file's name: "May I write a file? write Hello.txt." read the gate's detail
    out as it is shown (M3 e2e, 2026-10-05)."""
    title = (title or "May I go ahead?").strip()
    detail = (message or "").strip().split("\n")[0]
    m = _FILE_ACT.match(detail)
    if m:
        return f"May I {m.group(1)} {m.group(2).rstrip('/').rsplit('/', 1)[-1]}? Yes or no?"
    if detail and len(detail) <= 48 and not _SHELL.search(detail):
        return f"{title} {detail}. Yes or no?"
    if detail:
        first = detail.split()[:2]
        if first and not _SHELL.search(" ".join(first)):
            return f"{title} It starts with {' '.join(first)}. Yes or no?"
    return f"{title} Yes or no?"


# Markdown the model writes despite the persona ("no markdown"): asked for a long story through the apps, qwen38 sent
# "Here you go.", then "---", then "**The Weight of Light**" (a real-server run, 2026-10-05), and the voice
# spent 730 ms on the rule. The captions keep the Markdown (the apps render it); only what is spoken loses it.
# Pipecat's MarkdownTextFilter was tried first: it collapses blank lines before converting, so a rule inside a
# multi-line sentence ("---\n\n**Title**") survived it.
_MD_RULES = [
    (re.compile(r"^[ \t]*(?:[-*_=][ \t]*){3,}$", re.M), ""),                    # ---, ***, ___, === (and setext lines)
    (re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$", re.M), ""),   # |---|---|
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", re.M), r"\1"),  # ## Heading
    (re.compile(r"^[ \t]*>[ \t]?", re.M), ""),                                  # > quote
    (re.compile(r"^[ \t]*[-*+•][ \t]+", re.M), ""),                             # - bullet (numbered items are read)
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),                            # [text](url), ![alt](url)
    (re.compile(r"`+([^`]*)`+"), r"\1"),                                        # `code`
    (re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"), r"\2"),                      # **strong**, __strong__
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),          # *emphasis*
    (re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])"), r"\1"),            # _emphasis_ (not snake_case)
    (re.compile(r"\*\*|__"), ""),                                               # a pair left open across sentences
    (re.compile(r"[ \t]*\|[ \t]*"), ", "),                                      # table cells
]


def speakable(text: str) -> str:
    """Text as it should be spoken: Markdown markup removed (rules, headings, bullets, emphasis, inline code, links,
    tables), lines joined. Words are never changed. Empty when nothing is left to say (a rule on its own)."""
    for pattern, repl in _MD_RULES:
        text = pattern.sub(repl, text)
    # a line that ends without punctuation (a title, a list item) gets a full stop, so it is said as its own phrase
    text = re.sub(r"(\w)[ \t]*\n\s*(?=\S)", r"\1. ", text.strip())
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"(?:,\s*){2,}", ", ", text).strip(" ,")


def word_count(text: str) -> int:
    return len(_WORDS.findall(text.lower()))


def group_sentences(sentences: list[str], *, min_words: int = 0, max_sentences: int = 1) -> list[str]:
    """A reply's sentences joined into the texts the voice generates one at a time: a run of sentences shorter than
    `min_words` words is said together with the sentence after it, and up to `max_sentences` sentences go into one
    generation. A short last sentence joins the group before it. Each generation restarts the voice's pitch and pace,
    and Qwen3-TTS given a lone "Done." or "Hey!" is where it goes off (an early live test opened with "Hey!"
    generated alone, 2026-10-05). min_words 0 and max_sentences 1: one generation per sentence, as before."""
    groups: list[str] = []
    cur: list[str] = []
    for s in (s.strip() for s in sentences):
        if not s:
            continue
        cur.append(s)
        if word_count(" ".join(cur)) >= min_words and len(cur) >= max(1, max_sentences):
            groups.append(" ".join(cur))
            cur = []
    if cur:
        if groups and word_count(" ".join(cur)) < min_words:
            groups[-1] = f"{groups[-1]} {' '.join(cur)}"
        else:
            groups.append(" ".join(cur))
    return groups


# Words a listener says to keep a speaker going, not to take the turn (the barge-in bench's backchannels, 2026-10-05).
BACKCHANNEL_WORDS = frozenset("""mm mmm mhm mm-hm mm-hmm mmhm hm hmm uh-huh uh huh yeah yep yes yup right okay ok sure
i see got it oh ah cool nice alright true""".split())


def is_backchannel(text: str | None) -> bool:
    """Every word is a backchannel ("Mm-hmm.", "Yeah, yeah.", "I see."); empty text is not one."""
    import re
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", (text or "").lower())
    return bool(words) and all(w in BACKCHANNEL_WORDS for w in words)
