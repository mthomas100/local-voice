"""Life reflections into Atlas `inbox/`, in Atlas's own shapes, never touching a word the person wrote.

atlas.py has no verb for inbox pages: every Atlas skill (atlas-process, atlas-review, atlas-ask) writes them directly
in documented shapes. This writer does the same, and goes through atlas.py wherever atlas.py has the logic: its own
`render` for the frontmatter, `split`/`parse`/`human` to prove the page parses and counts as an agent's page, `context`
for the tags to reuse, and `queue` to see the page listed as waiting on the person.

It deliberately does not run `atlas.py start`/`finish`: both run `git add -A` over the whole iCloud-synced repo, which
from a background job would commit the person's half-finished edits (and any iCloud placeholder state) under the
brain's name. Instead it creates only new files in inbox/, commits only those paths (atlas-review's precedent:
`git add inbox && git commit`), and proves afterwards that the commit holds nothing else and that the rest of the
working tree is exactly as it was.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from .prompts import weekday
from .turnlog import Turn
from .validate import Reflection


class AtlasError(RuntimeError):
    pass


class AtlasBusy(AtlasError):
    """The repo is locked or mid-operation (Obsidian's git, a rebase): try again later."""


@dataclass
class AtlasResult:
    page: str            # relative to the root, e.g. inbox/voice-reflections-2026-10-04.md
    commit: str
    queue_lists_it: bool
    warnings: list[str] = field(default_factory=list)


def check_root(root: Path) -> None:
    for need in ("atlas.py", "AGENTS.md", ".git"):
        if not (root / need).exists():
            raise AtlasError(f"{root} is not an Atlas repo ({need} missing)")


def load_module(root: Path) -> ModuleType:
    """Import the repo's own atlas.py (standard library only; its main() is guarded)."""
    name = f"_atlas_{abs(hash(str(root)))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, root / "atlas.py")
    if not spec or not spec.loader:
        raise AtlasError(f"cannot load {root / 'atlas.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("render", "split", "parse", "human"):
        if not hasattr(mod, fn):
            raise AtlasError(f"atlas.py has no {fn}(); the brain's Atlas writer needs updating")
    sys.modules[name] = mod
    return mod


def _run(root: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), cwd=str(root), capture_output=True, text=True, timeout=timeout)


def known_tags(root: Path) -> set[str]:
    """Tags already in use, from `python3 atlas.py context` (its first two lines)."""
    r = _run(root, sys.executable, "atlas.py", "context")
    if r.returncode != 0:
        return set()
    lines = r.stdout.splitlines()
    if len(lines) < 2 or "none yet" in lines[1]:
        return set()
    return {part.strip().rsplit(" (", 1)[0] for part in lines[1].split(",") if part.strip()}


def saved_words(root: Path, t: Turn) -> tuple[str | None, str]:
    """The turn's captured words if the note holds them verbatim now, else None and why. A read error is reported,
    never 'fixed' (SPACES.md: an iCloud file can be an evicted placeholder)."""
    words = t.captured_text
    if not words or not t.atlas:
        return None, "not captured"
    if Path(t.atlas.root).resolve() != root.resolve():
        return None, f"captured into another root ({t.atlas.root})"
    note = (root / t.atlas.path).resolve()
    try:
        note.relative_to(root.resolve())
    except ValueError:
        return None, f"note path leaves the root ({t.atlas.path})"
    try:
        text = note.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"cannot read {t.atlas.path}: {e}"
    if words not in text:
        return None, f"words not found verbatim in {t.atlas.path} (edited since, or captured elsewhere)"
    return words, "ok"


def _status(root: Path) -> set[str]:
    """The working tree's state as a set of porcelain entries (paths are NUL-separated, so any name is safe)."""
    r = _run(root, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if r.returncode != 0:
        raise AtlasError(f"git status failed: {r.stderr.strip()}")
    return {e for e in r.stdout.split("\0") if e}


def _busy(root: Path) -> str | None:
    g = root / ".git"
    for marker in ("index.lock", "MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"):
        if (g / marker).exists():
            return marker
    return None


def page_text(atlas: ModuleType, *, day: str, reflections: list[Reflection], handles: dict[str, Turn], by: str
              ) -> str:
    notes: list[str] = []
    for r in reflections:
        for q in r.quotes:
            stem = Path(handles[q.handle].atlas.path).stem  # type: ignore[union-attr]
            if stem not in notes:
                notes.append(stem)
    tags = ["digest", "voice"] + [t for r in reflections for t in r.tags]
    tags = list(dict.fromkeys(tags))
    title = f"Voice reflections {day}"
    front = (atlas.render("type", "Analysis") + atlas.render("title", title)
             + atlas.render("description", f"What you said aloud to the voice agent on {weekday(day)} {day} and kept "
                                           f"in your journal, with what the voice brain noticed.")
             + atlas.render("tags", tags) + atlas.render("status", "draft") + atlas.render("date", day)
             + atlas.render("by", by) + atlas.render("related", [f"[[{n}]]" for n in notes]))
    body = [f"# {title}", "",
            f"Drafted by the voice agent's background brain from what you said on {weekday(day)}. Your words are "
            f"quoted exactly, each linked to the note that holds them. Keep a reflection by moving this page to "
            f"wiki/, or delete it.", ""]
    for r in reflections:
        body += [f"## {r.title}", "", "### In your words", ""]
        for q in r.quotes:
            t = handles[q.handle]
            stem = Path(t.atlas.path).stem  # type: ignore[union-attr]
            body += ["> " + line if line else ">" for line in q.text.split("\n")]
            body += [f"> — [[{stem}]], {t.t_start.strftime('%H:%M')}, said to voice", ""]
        if r.notice:
            body += ["### What the AI notices", "", f"- {r.notice}", ""]
        if r.questions:
            body += ["### Questions", ""] + [f"- {q}" for q in r.questions] + [""]
    return "---\n" + "\n".join(front) + "\n---\n" + "\n".join(body).rstrip("\n") + "\n"


def _commit(root: Path, rel: str, day: str, by: str) -> AtlasResult:
    """Commit exactly one new inbox page, then prove the commit holds nothing else and nothing else moved."""
    before = _status(root) - {f"?? {rel}"}
    try:
        for args in (("git", "add", "--", rel),
                     ("git", "commit", "-q", "-m", f"inbox: voice reflections {day} via {by}", "--", rel)):
            r = _run(root, *args)
            if r.returncode != 0:
                raise AtlasError(f"{' '.join(args[:2])} failed: {r.stderr.strip() or r.stdout.strip()}")
    except AtlasError:
        _run(root, "git", "reset", "-q", "--", rel)
        (root / rel).unlink(missing_ok=True)
        raise
    sha = _run(root, "git", "rev-parse", "HEAD").stdout.strip()
    files = _run(root, "git", "show", "--name-only", "--format=", "HEAD").stdout.split()
    if files != [rel]:
        raise AtlasError(f"commit {sha[:12]} holds {files}, not only {rel}: look at it now")
    warnings = []
    if _status(root) != before:
        # Not ours to fix: the person (or Obsidian) changed something during the write. The commit is still only ours.
        warnings.append(f"the working tree changed during the write (someone else's edit?); commit {sha[:12]} "
                        f"holds only {rel}")
    q = _run(root, sys.executable, "atlas.py", "queue")
    return AtlasResult(page=rel, commit=sha, queue_lists_it=rel in q.stdout, warnings=warnings)


def write(root: Path, *, day: str, reflections: list[Reflection], handles: dict[str, Turn], by: str,
          dry_run: bool = False) -> AtlasResult:
    check_root(root)
    if (m := _busy(root)) is not None:
        raise AtlasBusy(f"Atlas git is busy ({m})")
    atlas = load_module(root)
    text = page_text(atlas, day=day, reflections=reflections, handles=handles, by=by)
    lines, body = atlas.split(text)
    data = atlas.parse(lines)
    if lines is None or data.get("by") != by or atlas.human(data):
        raise AtlasError("the page would not parse as an agent's page")
    rel = f"inbox/voice-reflections-{day}.md"
    path = root / rel
    if path.exists():
        existing = atlas.parse(atlas.split(path.read_text(encoding="utf-8"))[0])
        if existing.get("by") == by:
            # An earlier run of this day wrote it. Committed: done. Left uncommitted by a crash between the write
            # and the commit: commit it as it is (it is the brain's own page, never the person's).
            tracked = _run(root, "git", "ls-files", "--error-unmatch", "--", rel).returncode == 0
            clean = not _run(root, "git", "status", "--porcelain", "--", rel).stdout.strip()
            if (tracked and clean) or dry_run:
                return AtlasResult(page=rel, commit="", queue_lists_it=True)
            return _commit(root, rel, day, by)
        rel = f"inbox/voice-reflections-{day}-{by}.md"
        path = root / rel
        if path.exists():
            raise AtlasError(f"{rel} exists and is not the brain's")
    if dry_run:
        return AtlasResult(page=rel, commit="(dry run)", queue_lists_it=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _commit(root, rel, day, by)
