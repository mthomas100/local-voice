// voice_gate.ts: the per-space risk gate from SPACES.md as a Pi extension (05c prototype, 2026-10-05).
//
// Every tool call is allowed, asked about, or refused, by the space's tier (env, set per child):
//   readonly  read-only tools run; anything else is refused without asking.
//   ask       read-only tools run; anything else needs a spoken yes.
//   trusted   read-only tools, writes inside the space root that match VOICE_WRITE_ALLOW, and bash
//             commands matching VOICE_BASH_ALLOW run; anything else needs a spoken yes.
// "Needs a spoken yes" is ctx.ui.confirm(title, message, {timeout}). In RPC mode that is an
// extension_ui_request (method "confirm", with "timeout") the voice orchestrator speaks and answers with
// extension_ui_response; with no answer Pi resolves it false at the timeout, so silence means no.
// A refusal returns {block, reason}: Pi turns it into an error tool result the model reads.
//
// Why the bash check parses instead of globbing the raw string: "python3 atlas.py *" as a plain glob also
// matches "python3 atlas.py queue; rm -rf ~". So the command's first line must be free of shell control
// characters before the glob is tried, and the only multi-line form accepted is a quoted heredoc
// (<<'MARK') whose marker line appears exactly once, as the last line: its body is inert text (the
// person's words for atlas.py capture), never commands. Anything else falls through to asking.
//
// Env: VOICE_TIER (readonly|ask|trusted, default ask), VOICE_CONFIRM_TIMEOUT_MS (default 20000),
// VOICE_BASH_ALLOW and VOICE_WRITE_ALLOW (JSON arrays of globs; write globs are relative to the root),
// VOICE_READ_TOOLS (comma list, default read,grep,find,ls), VOICE_DENY_TERMINATES=1 to stop the run after a
// refusal instead of letting the model answer it.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as os from "node:os";
import * as path from "node:path";

type Tier = "readonly" | "ask" | "trusted";
type Verdict = { action: "allow" } | { action: "ask"; title: string; message: string } | { action: "refuse"; reason: string };

const TIER = (["readonly", "ask", "trusted"].includes(process.env.VOICE_TIER ?? "") ? process.env.VOICE_TIER : "ask") as Tier;
const TIMEOUT_MS = Number(process.env.VOICE_CONFIRM_TIMEOUT_MS ?? 20000);
const BASH_ALLOW = jsonList(process.env.VOICE_BASH_ALLOW);
const WRITE_ALLOW = process.env.VOICE_WRITE_ALLOW ? jsonList(process.env.VOICE_WRITE_ALLOW) : ["*"];
const READ_TOOLS = new Set((process.env.VOICE_READ_TOOLS ?? "read,grep,find,ls").split(",").map((s) => s.trim()).filter(Boolean));
const KB_READ_VERBS = new Set(["search", "trace"]);
const DENY_TERMINATES = process.env.VOICE_DENY_TERMINATES === "1";
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

export function judge(tool: string, input: any, cwd: string, readOnlyHint: boolean | undefined): Verdict {
  if (READ_TOOLS.has(tool)) return { action: "allow" };
  if (tool === "kb") {
    const verb = String(input?.args?.[0] ?? "");
    if (KB_READ_VERBS.has(verb)) return { action: "allow" };
    if (TIER === "readonly") return { action: "refuse", reason: `kb ${verb} changes the knowledge base; this space is read-only` };
    return { action: "ask", title: "May I change your knowledge base?", message: `kb ${(input?.args ?? []).join(" ").slice(0, 160)}` };
  }
  if (tool === "bash") {
    const command = String(input?.command ?? "");
    const head = simpleHead(stripCdIntoRoot(command, cwd) ?? command);
    if (TIER === "readonly") return { action: "refuse", reason: "this space is read-only: no shell commands" };
    if (TIER === "trusted" && head !== null && BASH_ALLOW.some((g) => globToRegExp(g).test(head))) return { action: "allow" };
    const firstLine = command.split("\n")[0].slice(0, 160);
    return { action: "ask", title: "May I run a command?", message: firstLine };
  }
  if (tool === "write" || tool === "edit") {
    const p = String(input?.path ?? input?.file_path ?? "");
    const abs = path.resolve(cwd, p);
    const rel = path.relative(cwd, abs);
    const inside = !!rel && !rel.startsWith("..") && !path.isAbsolute(rel);
    if (TIER === "readonly") return { action: "refuse", reason: `this space is read-only: no ${tool}` };
    if (TIER === "trusted" && inside && WRITE_ALLOW.some((g) => globToRegExp(g).test(rel))) return { action: "allow" };
    return { action: "ask", title: `May I ${tool} a file?`, message: `${tool} ${inside ? rel : abs}` };
  }
  if (readOnlyHint === true) return { action: "allow" };
  if (TIER === "readonly") return { action: "refuse", reason: `${tool} is not read-only; this space is read-only` };
  return { action: "ask", title: `May I use ${tool}?`, message: JSON.stringify(input ?? {}).slice(0, 160) };
}

export default function voiceGate(pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    const hint = pi.getAllTools().find((t) => t.name === event.toolName)?.annotations?.readOnlyHint;
    const v = judge(event.toolName, event.input, ctx.cwd, hint);
    if (v.action === "allow") return undefined;
    if (v.action === "refuse") return { block: true, reason: v.reason, terminate: DENY_TERMINATES };
    if (!ctx.hasUI) return { block: true, reason: `${event.toolName} needs a yes and there is nobody to ask`, terminate: DENY_TERMINATES };
    // signal: an abort (the user barged in) dismisses the dialog at once. Without it the abort waits until the
    // dialog times out, 20 s by default (05c E10).
    const ok = await ctx.ui.confirm(v.title, v.message, { timeout: TIMEOUT_MS, signal: ctx.signal });
    if (ok) return undefined;
    return { block: true, reason: `The user did not approve this (said no, or did not answer within ${+(TIMEOUT_MS / 1000).toFixed(1)} s). Do not retry it; say you did not do it.`, terminate: DENY_TERMINATES };
  });
}
