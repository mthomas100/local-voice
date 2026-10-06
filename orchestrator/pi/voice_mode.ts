// voice_mode.ts: conversation vs act mode as an extension command (05c prototype, 2026-10-05).
//
// Pi's --tools is an allowlist of what gets REGISTERED: a tool left out of it cannot be activated later
// (setActiveTools ignores unknown names; measured in 05c). So the child is spawned with every tool the space
// may ever use (--tools = tools + act_tools) and this extension narrows the active set to VOICE_TOOLS at
// session_start. `/voice-mode act` activates everything the child registered; `/voice-mode conversation`
// goes back to VOICE_TOOLS. The orchestrator sends the command as a `prompt`: Pi runs extension commands at
// once, even during a run, and answers disposition "handled" with no model call. The new mode is reported
// with setStatus("voice-mode", ...), an extension_ui_request the orchestrator can show.
//
// Cost (05c E7b): the active set is rendered into the system message's <tools> section and the request's
// tools array, both at the head of the prompt, so the first request after a switch re-processes the whole
// conversation on llama.cpp / ds4 (no prefix-cache hit). Switch modes rarely, not per turn.
//
// space_switch (SPACES.md "Switching", 2026-10-05): the orchestrator's router switches spaces on plain words ("go to my
// journal") before any model call; this tool is the model's way to switch when the words were less direct ("go where my
// journal is", "can we do this in the atlas instead"). It only reports: the orchestrator sees the call (its arguments
// come with tool_execution_start), and once the run settles it says "Switching to ...", makes that space active, and
// when the person asked for something there, sends their own words (never the model's paraphrase) as that space's first
// prompt. The result ends the run (`terminate`), so this child says nothing more. Registered only where VOICE_SPACES
// names another space; read-only by its annotation, so voice_gate.ts lets it run.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type SpaceInfo = { name: string; description: string };

function list(s: string | undefined): string[] {
  return (s ?? "").split(",").map((x) => x.trim()).filter(Boolean);
}

function otherSpaces(): SpaceInfo[] {
  try {
    const v = JSON.parse(process.env.VOICE_SPACES ?? "[]");
    return Array.isArray(v) ? v.filter((s) => s?.name && s.name !== process.env.VOICE_SPACE) : [];
  } catch {
    return [];
  }
}

export default function voiceMode(pi: ExtensionAPI) {
  let all: string[] = [];
  let conversation: string[] = [];
  let mode = "conversation";
  pi.on("session_start", async () => {
    all = pi.getActiveTools();
    const want = list(process.env.VOICE_TOOLS);
    conversation = want.length ? all.filter((t) => want.includes(t)) : all;
    pi.setActiveTools(conversation);
  });
  pi.registerCommand("voice-mode", {
    description: "Switch the voice agent between conversation (read-only tools) and act mode",
    handler: async (args, ctx) => {
      const want = args.trim();
      if (want === "act") {
        pi.setActiveTools(all);
        mode = "act";
      } else if (want === "conversation") {
        pi.setActiveTools(conversation);
        mode = "conversation";
      }
      ctx.ui.setStatus("voice-mode", `${mode}: ${pi.getActiveTools().join(",")}`);
    },
  });
  const others = otherSpaces();
  if (!others.length) return;
  const names = others.map((s) => `${s.name} (${s.description})`).join(", ");
  pi.registerTool({
    name: "space_switch",
    label: "space_switch",
    description: `Move this conversation to another of the person's spaces: ${names}. Use it when they ask to go to one, ` +
      `or to work on something that lives there, in words other than a plain "go to ..." (the voice already switches on ` +
      `those). If it is unclear which space they mean, ask them instead. The voice says it is switching, and their request ` +
      `continues in that space.`,
    parameters: Type.Object({
      space: Type.String({ description: `the space's name: ${others.map((s) => s.name).join(", ")}` }),
      request: Type.Optional(Type.String({ description: "what they want done there, if anything beyond going there; empty if they only asked to go" })),
    }),
    annotations: { readOnlyHint: true },
    async execute(_id, params) {
      const p = params as { space: string; request?: string };
      const target = others.find((s) => s.name === String(p.space).trim().toLowerCase());
      if (!target) throw new Error(`There is no space "${p.space}" to switch to; the spaces are ${others.map((s) => s.name).join(", ")}.`);
      return {
        content: [{ type: "text", text: `Switching to ${target.description}; their request continues there. Say nothing more.` }],
        details: { space: target.name, request: String(p.request ?? "") }, terminate: true,
      };
    },
  });
}
