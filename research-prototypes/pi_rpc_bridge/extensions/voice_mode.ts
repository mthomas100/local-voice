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
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function list(s: string | undefined): string[] {
  return (s ?? "").split(",").map((x) => x.trim()).filter(Boolean);
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
}
