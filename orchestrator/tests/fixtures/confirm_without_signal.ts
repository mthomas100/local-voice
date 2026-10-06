// Test fixture (05c): an extension that asks before every bash call WITHOUT passing ctx.signal to the dialog,
// as third-party extensions may. An abort alone then waits until the dialog times out (20 s here); the bridge's
// interrupt also cancels open dialogs so the run settles at once.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "bash") return undefined;
    const ok = await ctx.ui.confirm("Run it?", String((event.input as { command?: string }).command ?? ""), { timeout: 20000 });
    return ok ? undefined : { block: true, reason: "not approved" };
  });
}
