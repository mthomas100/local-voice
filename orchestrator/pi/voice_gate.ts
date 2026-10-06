// voice_gate.ts: the per-space risk gate from SPACES.md as a Pi extension (05c prototype, 2026-10-05).
//
// Every tool call is allowed, asked about, or refused, by the space's tier (env, set per child):
//   readonly  read-only tools run; anything else is refused without asking.
//   ask       read-only tools run; anything else needs a spoken yes.
//   trusted   read-only tools, writes inside the space root that match VOICE_WRITE_ALLOW, and bash
//             commands matching VOICE_BASH_ALLOW run; anything else needs a spoken yes.
// "Needs a yes" is an approval ("Approvals" below): ctx.ui.select(approval JSON, choices, {timeout}), in RPC mode an
// extension_ui_request (method "select") the voice orchestrator shows as a card, speaks, and answers with
// extension_ui_response; with no answer Pi resolves it undefined at the timeout, so silence means no.
// A refusal returns {block, reason}: Pi turns it into an error tool result the model reads.
//
// Why the bash check parses instead of globbing the raw string: "python3 atlas.py *" as a plain glob also
// matches "python3 atlas.py queue; rm -rf ~". So the command's first line must be free of shell control
// characters before the glob is tried, and the only multi-line form accepted is a quoted heredoc
// (<<'MARK') whose marker line appears exactly once, as the last line: its body is inert text (the
// person's words for atlas.py capture), never commands. Anything else falls through to asking.
//
// Env: VOICE_TIER (readonly|ask|trusted, default ask), VOICE_CONFIRM_TIMEOUT_MS (default 20000; a backstop behind the
// orchestrator's own wait), VOICE_BASH_ALLOW and VOICE_WRITE_ALLOW (JSON arrays of globs; write globs are relative to
// the root), VOICE_READ_TOOLS (comma list, default read,grep,find,ls), VOICE_DENY_TERMINATES=1 to stop the run after a
// refusal instead of letting the model answer it, KB_HOME (the knowledge base kb.ts uses, default ~/kb), VOICE_SPACE
// and VOICE_SPACE_DESC (the space's name, and its root as said aloud: "your atlas journal").
//
// Two speed rules for a spoken turn (M1 e2e, 2026-10-05: one tool turn took 2 min 10 s). kb prints a page's path
// from the kb's root ("/wiki/decisions/x.md"), which the read tools take for an absolute path that does not exist;
// such a path is mapped to the page in place (Pi: "mutate event.input"). And find or grep over the whole home folder
// or the disk is refused with a pointer: one such find took 105 s.
//
// A budget for looking things up in conversation mode (2026-10-05): asked "And how much does one cost?" about a heat
// pump, qwen38 searched the kb and then grepped the rig's docs for a way to search the web for over 90 s, and "Let me
// look that up." was all the person heard (e2e 14:03). Once a run has made VOICE_TOOL_BUDGET_CALLS tool calls or
// spent VOICE_TOOL_BUDGET_S seconds, further calls are refused with an instruction to answer with what it has and
// offer to look further; after two such refusals the run is stopped (the orchestrator then says so). Act mode (any
// active tool beyond VOICE_TOOLS) has no budget: there the person asked for work to be done. 0 turns a limit off.
//
// Approvals (PROTOCOL.md "Approvals", 2026-10-05): an early version asked "May I change your knowledge base? It
// starts with kb new. Yes or no?", which said neither what would be created nor where. So every call the gate asks about carries what a coding agent's prompt shows: a one-sentence `summary` of
// exactly what will happen, the `action` (tool, effect, the exact command or the absolute path, cwd, space, mode, and a
// preview: the text to be written or a unified diff, cut at 4,000 characters with a visible marker) and the `scope` an
// "allow for this session" answer would cover (the same tool and command prefix or folder, never broader; none for a
// compound command, a delete or the network). It goes out as ctx.ui.select(<the approval as JSON>, <the choices>):
// select, not confirm, because the answer must say which choice was made. The orchestrator answers with a choice id,
// or with `deny_said` (not approved, and the person said something else, which it steers into the run) or `timeout`
// (no answer: it has already said aloud what was not done). A plain no or a silence then ends the run without another
// model call (`terminate`), because the orchestrator has said "Okay, I didn't …" itself. Session grants are kept by
// the orchestrator, per space and voice session, not here: this child outlives a voice session.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

type Tier = "readonly" | "ask" | "trusted";
type Effect = "create" | "modify" | "delete" | "run" | "network";
export type Action = { tool: string; effect: Effect; command: string | null; path: string | null; cwd: string;
                       space: string; mode: string; preview: string | null };
export type Scope = { key: string; label: string };
export type Approval = { title: string; message: string; summary: string; short: string; action: Action; scope: Scope | null };
type Verdict = { action: "allow" } | { action: "ask"; approval: Approval } | { action: "refuse"; reason: string };

const TIER = (["readonly", "ask", "trusted"].includes(process.env.VOICE_TIER ?? "") ? process.env.VOICE_TIER : "ask") as Tier;
const TIMEOUT_MS = Number(process.env.VOICE_CONFIRM_TIMEOUT_MS ?? 20000);
const SPACE = process.env.VOICE_SPACE ?? "";
const SPACE_DESC = process.env.VOICE_SPACE_DESC ?? "";   // "your atlas journal": how the space's root is named aloud
export const PREVIEW_MAX = 4000;
const BASH_ALLOW = jsonList(process.env.VOICE_BASH_ALLOW);
const WRITE_ALLOW = process.env.VOICE_WRITE_ALLOW ? jsonList(process.env.VOICE_WRITE_ALLOW) : ["*"];
const READ_TOOLS = new Set((process.env.VOICE_READ_TOOLS ?? "read,grep,find,ls").split(",").map((s) => s.trim()).filter(Boolean));
// The kb CLI's verbs (`kb --help`, 2026-10-05). Read-only ones run without asking; the rest change the kb. A verb kb
// does not have is refused at once with a pointer, instead of a spoken question about something that cannot run:
// qwen38 called `kb read <page>` in the M1 e2e run (2026-10-05) and the gate asked "May I change your knowledge
// base?" for 20 s of silence before the model fell back to the read tool.
const KB_READ_VERBS = new Set(["search", "trace", "stale"]);
const KB_WRITE_VERBS = new Set(["new", "ingest", "link", "index", "supersede", "verify", "log", "session", "harness", "doctor"]);
// kb new's page types and their folders, read from kb's own TYPES table so the gate never disagrees with kb; the table
// as of 2026-10-05 when it cannot be read. An early version asked about `kb new Issue ...`, which kb would have
// refused (Issue is not a type): a call that cannot run is refused at once with a pointer, never asked about.
const KB_TYPES_FALLBACK: Record<string, [string, string]> = {
  concept: ["Concept", "concepts"], entity: ["Entity", "entities"], analysis: ["Analysis", "analyses"],
  decision: ["Decision", "decisions"],
};
const KB_SLUG = /^[a-z0-9][a-z0-9-]*$/;
const DENY_TERMINATES = process.env.VOICE_DENY_TERMINATES === "1";
const BUDGET_CALLS = Number(process.env.VOICE_TOOL_BUDGET_CALLS ?? 0);
const BUDGET_S = Number(process.env.VOICE_TOOL_BUDGET_S ?? 0);
const CONVERSATION_TOOLS = new Set((process.env.VOICE_TOOLS ?? "").split(",").map((s) => s.trim()).filter(Boolean));
const KB_HOME = path.resolve(process.env.KB_HOME ?? path.join(os.homedir(), "kb"));
const SHELL_CONTROL = /[;&|`$<>\\(){}\r\n]/;

function jsonList(s: string | undefined): string[] {
  if (!s) return [];
  try {
    const v = JSON.parse(s);
    return Array.isArray(v) ? v.map(String) : [];
  } catch {
    return [];
  }
}

export function globToRegExp(glob: string): RegExp {
  const body = glob.split("*").map((p) => p.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join("[^\\n]*");
  return new RegExp(`^${body}$`);
}

/** The command's first line, minus a trailing quoted-heredoc redirect, if the command is "simple":
 *  one line, or one line plus a quoted heredoc whose marker is the last line and appears only there. */
export function simpleHead(command: string): string | null {
  if (command.includes("\r")) return null;
  const nl = command.indexOf("\n");
  if (nl < 0) return SHELL_CONTROL.test(command) ? null : command.trim();
  const first = command.slice(0, nl);
  const m = /<<\s*(['"])([A-Za-z_][A-Za-z0-9_]*)\1\s*$/.exec(first);
  if (!m) return null;
  const lines = command.slice(nl + 1).replace(/\n+$/, "").split("\n");
  if (lines.indexOf(m[2]) !== lines.length - 1) return null;
  const head = first.slice(0, m.index).trimEnd();
  return SHELL_CONTROL.test(head) ? null : head;
}

/** `cd <dir> && rest`, where <dir> is the space root or inside it, returns `rest`; anything else returns null.
 *  Models add this prefix even when the cwd already is the root: qwen38 wrapped every Atlas capture in
 *  `cd <atlas root> && python3 atlas.py capture ...` or `cd "$(pwd)" && ...` (05d, 2026-10-05), which the bare
 *  check sent to a spoken confirmation on every musing. Only one leading cd is stripped; what follows gets the
 *  full check. */
// `cd` to the current directory spelled the ways models spell it (qwen38 wrote `cd "$(pwd)" && ...`, 05d): exact
// literal forms only, so no other expansion can ride along.
const CD_HERE = /^\s*cd\s+(?:"\$\(pwd\)"|\$\(pwd\)|"\$PWD"|\$PWD|"\$\{PWD\}"|\$\{PWD\}|\.\/?)\s*&&\s*/;

export function stripCdIntoRoot(command: string, cwd: string): string | null {
  const here = CD_HERE.exec(command);
  if (here) return command.slice(here[0].length);
  // double quotes still expand $, ` and \ in bash; unquoted ~user and globs expand too: none of those is accepted
  const m = /^\s*cd\s+(?:"([^"$`\\]*)"|'([^']*)'|((?:\\ |[^\s;&|'"`$<>(){}\\*?[\]])+))\s*&&\s*/.exec(command);
  if (!m) return null;
  let dir = m[1] ?? m[2] ?? (m[3] ?? "").replace(/\\ /g, " ");
  if (m[3] !== undefined && dir.startsWith("~") && dir !== "~" && !dir.startsWith("~/")) return null;
  if (m[3] !== undefined && (dir === "~" || dir.startsWith("~/"))) dir = path.join(os.homedir(), dir.slice(1));
  const rel = path.relative(path.resolve(cwd), path.resolve(cwd, dir));
  if (rel.startsWith("..") || path.isAbsolute(rel)) return null;
  return command.slice(m[0].length);
}

function expandHome(p: string): string {
  return p === "~" || p.startsWith("~/") ? path.join(os.homedir(), p.slice(1)) : p;
}

/** A path as kb prints it ("/wiki/decisions/x.md", from the kb's root) that does not exist as given but does inside
 *  the kb: that page's real path. Anything else (a real path, a missing one outside the kb): null, left alone. */
export function kbPath(p: string, kbHome: string = KB_HOME): string | null {
  if (!p.startsWith("/") || fs.existsSync(p)) return null;
  const inKb = path.join(kbHome, p);
  return inKb.startsWith(kbHome + path.sep) && fs.existsSync(inKb) ? inKb : null;
}

/** find or grep rooted at the home folder or the disk root: minutes of walking, not an answer to wait for. */
export function tooBroad(tool: string, input: any, cwd: string): string | null {
  if (tool !== "find" && tool !== "grep") return null;
  const root = path.resolve(cwd, expandHome(String(input?.path ?? ".")));
  if (root !== path.resolve(os.homedir()) && root !== path.parse(root).root) return null;
  const where = root === path.parse(root).root ? "the whole disk" : "the whole home folder";
  return `${tool} over ${where} takes minutes, too long for a spoken answer. Search a narrower folder (for example ~/repos), or the knowledge base with kb search, or ask the user where to look.`;
}

// ------------------------------------------------------------------------------------------------- approvals

/** Text cut at `max` characters with a marker that says how much was left out (PROTOCOL.md: a visible marker). */
export function cut(text: string, max: number = PREVIEW_MAX): string {
  return text.length <= max ? text : `${text.slice(0, max)}\n… [cut here: ${text.length - max} more characters not shown]`;
}

/** One argv word as a shell would need it written, so the command shown is the command that runs. */
export function shellQuote(arg: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, `'\\''`)}'`;
}

/** How a folder is named in a spoken sentence: the space's own name for its root, else plainly by its name. */
export function folderPhrase(dir: string, root: string, spaceDesc: string = SPACE_DESC): string {
  const d = path.resolve(dir);
  if (d === path.resolve(os.homedir())) return "your home folder";
  if (d === path.resolve(root) && spaceDesc) return spaceDesc;
  return `the folder ${path.basename(d) || d}`;
}

export function kbTypes(kbHome: string = KB_HOME): Record<string, [string, string]> {
  try {
    const src = fs.readFileSync(path.join(kbHome, "bin", "kb"), "utf8");
    const table = /^TYPES\s*=\s*\{([\s\S]*?)\}/m.exec(src)?.[1] ?? "";
    const out: Record<string, [string, string]> = {};
    for (const m of table.matchAll(/"(\w+)"\s*:\s*\(\s*"(\w+)"\s*,\s*"([\w-]+)"\s*\)/g)) out[m[1]] = [m[2], m[3]];
    if (Object.keys(out).length) return out;
  } catch {
    // no kb source to read (a test's KB_HOME): the table as of 2026-10-05
  }
  return KB_TYPES_FALLBACK;
}

/** kb's argv parsed the way its argparse reads `kb new type slug [--title T]`: positionals and --name value pairs. */
function kbArgs(args: string[]): { pos: string[]; opts: Record<string, string> } {
  const pos: string[] = [];
  const opts: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq > 0) opts[a.slice(2, eq)] = a.slice(eq + 1);
      else opts[a.slice(2)] = args[++i] ?? "";
    } else pos.push(a);
  }
  return { pos, opts };
}

function kbApproval(args: string[], mode: string): Approval | { refuse: string } {
  const verb = args[0];
  const command = `kb ${args.map(shellQuote).join(" ")}`;
  const action = (effect: Effect, p: string | null, preview: string | null): Action =>
    ({ tool: "kb", effect, command: cut(command), path: p, cwd: KB_HOME, space: SPACE, mode, preview });
  if (verb === "new") {
    const { pos, opts } = kbArgs(args.slice(1));
    const types = kbTypes();
    const [typeArg = "", slug = ""] = pos;
    const t = types[typeArg.toLowerCase()];
    if (!t) {
      const names = Object.values(types).map((x) => x[0]).join(", ");
      return { refuse: `kb new takes a page type of ${names}, not "${typeArg}" (an open issue or a finding is usually an Analysis). Call it again with one of those; nothing was asked or written.` };
    }
    if (!KB_SLUG.test(slug)) {
      return { refuse: `kb new's slug must be lowercase letters, digits and hyphens (got "${slug}"). Call it again with such a slug; nothing was asked or written.` };
    }
    const page = path.join(KB_HOME, "wiki", t[1], `${slug}.md`);
    if (fs.existsSync(page)) {
      return { refuse: `${path.relative(KB_HOME, page)} already exists, and kb new never overwrites a page (kb supersede replaces one). Nothing was asked or written.` };
    }
    // kb's own default title (bin/kb cmd_new): the slug with spaces, first letter capitalised
    const title = opts.title || (slug.replace(/-/g, " ").charAt(0).toUpperCase() + slug.replace(/-/g, " ").slice(1));
    const preview = `(kb adds its front matter: type ${t[0]}, this title, status draft, today's date and this session)\n# ${title}\n\n## Summary\n\n## Related\n`;
    return {
      title: "May I create a page in your knowledge base?", message: command,
      summary: `Create a new ${t[0].toLowerCase()} page in your knowledge base titled "${title}".`,
      short: "create that page", action: action("create", page, cut(preview)),
      scope: { key: "kb new", label: "creating knowledge base pages" },
    };
  }
  return {
    title: "May I change your knowledge base?", message: command,
    summary: `Run the knowledge base command "kb ${verb}", which changes your knowledge base.`,
    short: "run that kb command", action: action(verb === "ingest" ? "create" : "modify", null, null),
    scope: { key: `kb ${verb}`, label: `"kb ${verb}" commands` },
  };
}

// The first word of a command that deletes or reaches the network: such commands are asked about every time (no
// session scope), and the spoken question says so.
const DELETERS = new Set(["rm", "rmdir", "unlink", "trash", "shred", "srm"]);
const NETWORK = new Set(["curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ftp", "telnet", "http", "https"]);
const GIT_NETWORK = new Set(["push", "pull", "fetch", "clone"]);
// Interpreters and wrappers whose next word is arbitrary code or another command: no session scope for a prefix of it,
// except `python3 script.py sub` style calls, which are scoped to that script (and its subcommand).
const INTERPRETERS = new Set(["python", "python3", "node", "ruby", "perl", "deno", "bun", "uv", "npx", "pnpm", "php"]);
const WRAPPERS = new Set(["bash", "sh", "zsh", "fish", "eval", "exec", "source", ".", "sudo", "doas", "env", "xargs",
                          "nohup", "time", "nice", "command", "builtin", "osascript", "open"]);

export function bashEffect(head: string): Effect {
  const w = head.trim().split(/\s+/);
  if (DELETERS.has(w[0]) || (w[0] === "git" && ["rm", "clean"].includes(w[1] ?? ""))) return "delete";
  if (NETWORK.has(w[0]) || (w[0] === "git" && GIT_NETWORK.has(w[1] ?? "")) ||
      (["brew", "pip", "pip3", "npm", "pnpm", "yarn", "gem", "cargo"].includes(w[0]) && ["install", "add", "update", "upgrade"].includes(w[1] ?? ""))) {
    return "network";
  }
  return "run";
}

/** The argv prefix an "allow for this session" answer would cover: the program and its subcommand word, or for an
 *  interpreter the script and its subcommand. null when no prefix is safe to grant (a wrapper that runs other code,
 *  an interpreter given code inline, a delete or the network). */
export function bashPrefix(head: string): string[] | null {
  const w = head.trim().split(/\s+/).filter(Boolean);
  if (!w.length || WRAPPERS.has(w[0]) || w[0].includes("=") || bashEffect(head) !== "run") return null;
  const word = (s: string | undefined) => s !== undefined && /^[a-z][a-z0-9_-]*$/.test(s);
  if (INTERPRETERS.has(w[0])) {
    const script = w[1];
    if (!script || script.startsWith("-") || !/\.(py|js|mjs|ts|rb|pl|php)$/.test(script)) return null;
    return word(w[2]) ? [w[0], script, w[2]] : [w[0], script];
  }
  return word(w[1]) ? [w[0], w[1]] : [w[0]];
}

function bashApproval(command: string, cwd: string, root: string, mode: string): Approval {
  const stripped = stripCdIntoRoot(command, cwd) ?? command;
  const head = simpleHead(stripped);               // null: more than one simple command
  const first = (head ?? stripped.split("\n")[0]).trim();
  const effect = bashEffect(head ?? first);
  const where = folderPhrase(cwd, root);
  // a quoted heredoc's text (simpleHead accepted its shape: the marker line once, last)
  const body = head !== null && stripped.includes("\n")
    ? stripped.slice(stripped.indexOf("\n") + 1).replace(/\n+$/, "").split("\n").slice(0, -1).join("\n") : null;
  const speakable = head !== null && first.length <= 60 && !/["'`\\]/.test(first);
  let summary = speakable ? `Run the command "${first}" in ${where}` : `Run a shell command in ${where}, starting with "${first.split(/\s+/).slice(0, 2).join(" ")}"`;
  summary += body !== null ? ", giving it the text shown." : ".";
  if (effect === "delete") summary += " It deletes files.";
  if (effect === "network") summary += " It uses the network.";
  const prefix = head !== null ? bashPrefix(head) : null;
  return {
    title: "May I run a command?", message: cut(command.split("\n")[0], 160), summary, short: "run that command",
    action: { tool: "bash", effect, command: cut(command), path: null, cwd, space: SPACE, mode, preview: body === null ? null : cut(body) },
    scope: prefix ? { key: `bash ${prefix.join(" ")}`, label: `"${prefix.join(" ")}" commands` } : null,
  };
}

/** Pi's edit input in its accepted shapes: {edits: [{oldText, newText}]}, edits as a JSON string or one object, or the
 *  legacy {oldText, newText} (Pi's edit.js normalises the same ones). */
export function editList(input: any): { oldText: string; newText: string }[] {
  let edits: any = input?.edits;
  if (typeof edits === "string") {
    try { edits = JSON.parse(edits); } catch { edits = []; }
  }
  if (edits && !Array.isArray(edits) && typeof edits === "object") edits = [edits];
  const out = Array.isArray(edits) ? edits.filter((e: any) => typeof e?.oldText === "string" && typeof e?.newText === "string") : [];
  if (typeof input?.oldText === "string" && typeof input?.newText === "string") out.push({ oldText: input.oldText, newText: input.newText });
  return out;
}

/** The file as it will be after the edits, when each oldText occurs exactly once (Pi also matches fuzzily; then null). */
export function applyEdits(content: string, edits: { oldText: string; newText: string }[]): string | null {
  const spans: [number, number, string][] = [];
  for (const e of edits) {
    const at = content.indexOf(e.oldText);
    if (!e.oldText || at < 0 || content.indexOf(e.oldText, at + 1) >= 0) return null;
    spans.push([at, at + e.oldText.length, e.newText]);
  }
  spans.sort((a, b) => b[0] - a[0]);
  let out = content;
  for (const [a, b, text] of spans) out = out.slice(0, a) + text + out.slice(b);
  return out;
}

type Patch = (file: string, before: string, after: string) => string;

function fileApproval(tool: "write" | "edit", input: any, cwd: string, root: string, mode: string, patch: Patch | null): Approval {
  const abs = path.resolve(cwd, expandHome(String(input?.path ?? input?.file_path ?? "")));
  const name = path.basename(abs);
  const dir = path.dirname(abs);
  const where = folderPhrase(dir, root);
  const scope = { key: `${tool} ${dir}`, label: `${tool === "write" ? "writing" : "editing"} files in ${where}` };
  const base = { tool, command: null, path: abs, cwd, space: SPACE, mode };
  if (tool === "write") {
    const exists = fs.existsSync(abs);
    return {
      title: `May I ${exists ? "replace" : "create"} ${name}?`, message: abs,
      summary: exists ? `Replace everything in the file ${name} in ${where} with the text shown.` : `Create a new file ${name} in ${where}.`,
      short: `${exists ? "replace" : "create"} ${name}`,
      action: { ...base, effect: exists ? "modify" : "create", preview: cut(String(input?.content ?? "")) }, scope,
    };
  }
  const edits = editList(input);
  let preview: string;
  let before: string | null = null;
  try {
    before = fs.statSync(abs).size <= 2_000_000 ? fs.readFileSync(abs, "utf8") : null;
  } catch {
    before = null;
  }
  const after = before === null ? null : applyEdits(before.replace(/\r\n/g, "\n"), edits);
  if (before !== null && after !== null && patch) {
    preview = patch(path.relative(cwd, abs) || name, before.replace(/\r\n/g, "\n"), after);
  } else {
    // shown as the model wrote them: the file could not be read, or an old text is not in it exactly once
    preview = "(the changes as asked for; they could not be laid over the file as it is now)\n" +
      edits.map((e) => `--- replace:\n${e.oldText}\n+++ with:\n${e.newText}`).join("\n\n");
  }
  const n = edits.length;
  return {
    title: `May I change ${name}?`, message: abs,
    summary: `Make ${n === 1 ? "one change" : `${n} changes`} to the file ${name} in ${where}.`,
    short: `change ${name}`, action: { ...base, effect: "modify", preview: cut(preview) }, scope,
  };
}

export function judge(tool: string, input: any, cwd: string, readOnlyHint: boolean | undefined,
                      mode: string = "conversation", patch: Patch | null = null): Verdict {
  const broad = tooBroad(tool, input, cwd);
  if (broad) return { action: "refuse", reason: broad };
  if (READ_TOOLS.has(tool)) return { action: "allow" };
  if (tool === "kb") {
    const args: string[] = (input?.args ?? []).map(String);
    const verb = args[0] ?? "";
    if (KB_READ_VERBS.has(verb)) return { action: "allow" };
    if (!KB_WRITE_VERBS.has(verb)) {
      return { action: "refuse", reason: `kb has no "${verb}" verb (it has ${[...KB_READ_VERBS, ...KB_WRITE_VERBS].join(", ")}). To read a page, use the read tool with the path kb printed (such as /wiki/decisions/x.md).` };
    }
    if (TIER === "readonly") return { action: "refuse", reason: `kb ${verb} changes the knowledge base; this space is read-only` };
    const a = kbApproval(args, mode);
    return "refuse" in a ? { action: "refuse", reason: a.refuse } : { action: "ask", approval: a };
  }
  if (tool === "bash") {
    const command = String(input?.command ?? "");
    const head = simpleHead(stripCdIntoRoot(command, cwd) ?? command);
    if (TIER === "readonly") return { action: "refuse", reason: "this space is read-only: no shell commands" };
    if (TIER === "trusted" && head !== null && BASH_ALLOW.some((g) => globToRegExp(g).test(head))) return { action: "allow" };
    return { action: "ask", approval: bashApproval(command, cwd, cwd, mode) };
  }
  if (tool === "write" || tool === "edit") {
    const p = String(input?.path ?? input?.file_path ?? "");
    const abs = path.resolve(cwd, p);
    const rel = path.relative(cwd, abs);
    const inside = !!rel && !rel.startsWith("..") && !path.isAbsolute(rel);
    if (TIER === "readonly") return { action: "refuse", reason: `this space is read-only: no ${tool}` };
    if (TIER === "trusted" && inside && WRITE_ALLOW.some((g) => globToRegExp(g).test(rel))) return { action: "allow" };
    return { action: "ask", approval: fileApproval(tool, input, cwd, cwd, mode, patch) };
  }
  if (readOnlyHint === true) return { action: "allow" };
  if (TIER === "readonly") return { action: "refuse", reason: `${tool} is not read-only; this space is read-only` };
  return {
    action: "ask", approval: {
      title: `May I use ${tool}?`, message: cut(JSON.stringify(input ?? {}), 160), summary: `Use the tool "${tool}".`,
      short: `use ${tool}`, scope: null,
      action: { tool, effect: "run", command: null, path: null, cwd, space: SPACE, mode, preview: cut(JSON.stringify(input ?? {}, null, 2)) },
    },
  };
}

/** Over budget: the reason a further call is refused, or null. `calls` counts the calls made before this one. */
export function overBudget(calls: number, elapsedS: number, maxCalls: number = BUDGET_CALLS, maxS: number = BUDGET_S): string | null {
  const byCalls = maxCalls > 0 && calls >= maxCalls;
  const byTime = maxS > 0 && elapsedS >= maxS;
  if (!byCalls && !byTime) return null;
  const spent = byCalls ? `${calls} tool calls` : `${Math.round(elapsedS)} seconds`;
  return `Stop looking: you have spent ${spent} on this spoken question, as long as a person waits on a call. Do not call any more tools now. Answer in one or two short sentences with what you already know or found, say plainly what you could not check, and offer to look further if they want.`;
}

/** What the gate tells the model for each answer the orchestrator can give (see "Approvals" at the top). */
export function answerVerdict(answer: string | undefined): { block: true; reason: string; terminate: boolean } | undefined {
  switch (answer) {
    case "allow_once":
    case "allow_session":
      return undefined;
    case "deny":
      return { block: true, terminate: true, reason: "The user did not approve this: they said no. It was not done, and they have been told so. Do not retry it." };
    case "timeout":
      return { block: true, terminate: true, reason: "The user did not approve this: they did not answer. It was not done, and they have been told so. Do not retry it unless they ask." };
    case "deny_said":
      return { block: true, terminate: false, reason: "The user did not approve this: it was not done, and they have been told so. They said something else instead, which follows as their next message: answer that. Do not retry this call unless they ask you to." };
    default:   // no answer from the orchestrator before this gate's own timeout, or the dialog was cancelled (an abort)
      return { block: true, terminate: DENY_TERMINATES, reason: `The user did not approve this (said no, or did not answer within ${+(TIMEOUT_MS / 1000).toFixed(1)} s). Do not retry it; say you did not do it.` };
  }
}

export default function voiceGate(pi: ExtensionAPI) {
  let calls = 0;
  let refusals = 0;
  let started = Date.now();
  let patch: Patch | null = null;
  // Pi's own unified diff (the package root is an extension's virtual module; deep imports are not)
  import("@earendil-works/pi-coding-agent").then((m: any) => {
    if (typeof m.generateUnifiedPatch === "function") patch = m.generateUnifiedPatch;
  }).catch(() => undefined);
  pi.on("agent_start", async () => {
    calls = 0;
    refusals = 0;
    started = Date.now();
  });
  pi.on("tool_call", async (event, ctx) => {
    const input = event.input as any;
    const conversation = CONVERSATION_TOOLS.size > 0 && pi.getActiveTools().every((t) => CONVERSATION_TOOLS.has(t));
    if (conversation) {
      const over = overBudget(calls, (Date.now() - started) / 1000);
      calls += 1;
      if (over) {
        refusals += 1;
        return { block: true, reason: over, terminate: refusals > 2 };
      }
    }
    if (READ_TOOLS.has(event.toolName) && typeof input?.path === "string") {
      const page = kbPath(input.path);
      if (page) input.path = page;
    }
    const hint = pi.getAllTools().find((t) => t.name === event.toolName)?.annotations?.readOnlyHint;
    // the mode as voice_mode.ts set it: act once a tool beyond VOICE_TOOLS is active (no VOICE_TOOLS: no modes)
    const mode = CONVERSATION_TOOLS.size === 0 || conversation ? "conversation" : "act";
    const v = judge(event.toolName, event.input, ctx.cwd, hint, mode, patch);
    if (v.action === "allow") return undefined;
    if (v.action === "refuse") return { block: true, reason: v.reason, terminate: DENY_TERMINATES };
    if (!ctx.hasUI) return { block: true, reason: `${event.toolName} needs a yes and there is nobody to ask`, terminate: DENY_TERMINATES };
    const choices = v.approval.scope ? ["allow_once", "allow_session", "deny"] : ["allow_once", "deny"];
    // signal: an abort (the user barged in) dismisses the dialog at once. Without it the abort waits until the
    // dialog times out (05c E10). The timeout is a backstop: the orchestrator answers `timeout` itself before it.
    const answer = await ctx.ui.select(JSON.stringify({ lv: "approval", v: 1, ...v.approval }), choices,
                                       { timeout: TIMEOUT_MS, signal: ctx.signal });
    return answerVerdict(answer);
  });
}
