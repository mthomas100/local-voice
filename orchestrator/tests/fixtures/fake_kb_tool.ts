// Test fixture (2026-10-05): a `kb` tool with kb.ts's parameter shape (args = the argv after `kb`) that runs nothing,
// so the voice gate's kb rules can be tested without the kb CLI, KB_HOME or a session digest.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "kb",
    label: "kb",
    description: "fake kb for tests",
    parameters: Type.Object({ args: Type.Array(Type.String()) }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: `ran kb ${(params as { args: string[] }).args.join(" ")}` }], details: {} };
    },
  });
}
