import Foundation

// Approvals (PROTOCOL.md, added 2026-10-05). A spoken question alone ("May I change your knowledge base? It starts
// with kb new. Yes or no?") is too vague to answer safely: nothing on screen says what it means, and silence refuses
// it. So the card gives what coding agents give: exactly what will happen, and a choice you can see and pick. The
// card's content and the questions waiting are worked out here, once, so the iPhone app, the Mac app and lvclient's
// log agree.

/// What an approval card shows for one `confirm_request`.
public struct ApprovalContent: Sendable, Equatable {
    /// A monospaced block: the exact command, the exact path, or an older server's message.
    public struct Exact: Sendable, Equatable, Identifiable {
        public enum Kind: String, Sendable { case command, path, details }
        public let kind: Kind
        public let label: String
        public let text: String
        /// A card has at most one block of each kind.
        public var id: Kind { kind }
    }

    public struct Preview: Sendable, Equatable {
        public enum Kind: String, Sendable { case text, diff }
        public let kind: Kind
        public let label: String
        /// As the server sent it, its cut marker included.
        public let text: String
        /// The server's marker when it cut the preview at 4,000 characters: the card pins it under the scroll box,
        /// where it is seen without scrolling.
        public let cutMarker: String?
    }

    public struct Choice: Sendable, Equatable, Identifiable {
        /// `primary` is the first allow; no choice is ever a default action (no key answers a card).
        public enum Role: String, Sendable { case primary, secondary, deny }
        public let id: String
        public let label: String
        public let allows: Bool
        public let role: Role
        /// The allow of a delete, drawn in red.
        public let destructive: Bool
    }

    public let headline: String
    /// Only `title` and `message`: a server from before Approvals.
    public let isLegacy: Bool
    public let effect: ApprovalEffect?
    /// "Creates", "Runs a command", ...: the effect in words, for the badge.
    public let effectLabel: String?
    /// The tool, the space and the mode ("kb", "home space", "act mode"), for the line under the headline.
    public let facts: [String]
    public let exact: [Exact]
    /// The working directory, shown under a command.
    public let cwd: String?
    public let preview: Preview?
    /// One button each, in the server's order.
    public let choices: [Choice]
    /// Whether the server offered `choices`; a request that offered none is answered with `confirmed` alone.
    public let choicesOffered: Bool
    /// What the voice takes instead of a button.
    public let spokenHint: String

    public init(_ request: ConfirmRequest) {
        let action = request.action
        let summary = Self.nonEmpty(request.summary)
        headline = summary ?? Self.nonEmpty(request.title) ?? "Your agent asks for permission"
        isLegacy = request.summary == nil && action == nil && request.choices.isEmpty
        effect = action?.effect
        effectLabel = action?.effect.map(Self.words)

        var facts: [String] = []
        if let tool = Self.nonEmpty(action?.tool) { facts.append(tool) }
        if let space = Self.nonEmpty(action?.space) { facts.append("\(space) space") }
        if let mode = Self.nonEmpty(action?.mode) { facts.append("\(mode) mode") }
        self.facts = facts

        // Exact means as sent: the blocks keep the server's text untrimmed.
        var exact: [Exact] = []
        let command = action?.command.flatMap { Self.nonEmpty($0) == nil ? nil : $0 }
        if let command { exact.append(Exact(kind: .command, label: "Command", text: command)) }
        if let path = action?.path, Self.nonEmpty(path) != nil {
            exact.append(Exact(kind: .path, label: Self.pathLabel(action?.effect), text: path))
        }
        // An older server's message is the command or the path (voice_gate.ts before 2026-10-05: "kb new …",
        // "write /path", the first line of a shell command), so it is shown as exactly as a command is.
        if exact.isEmpty, let message = Self.nonEmpty(request.message), message != headline {
            exact.append(Exact(kind: .details, label: "Details", text: request.message))
        }
        self.exact = exact
        cwd = command == nil ? nil : Self.nonEmpty(action?.cwd)

        if let text = action?.preview, Self.nonEmpty(text) != nil {
            let diff = Self.isDiff(text)
            let label: String
            switch (diff, action?.effect) {
            case (true, _): label = "Changes"
            case (false, .create?), (false, .modify?): label = "Text to be written"
            default: label = "Preview"
            }
            preview = Preview(kind: diff ? .diff : .text, label: label, text: text, cutMarker: Self.cutMarker(in: text))
        } else {
            preview = nil
        }

        let resolved = Self.resolveChoices(request.choices, effect: action?.effect)
        choices = resolved
        choicesOffered = !request.choices.isEmpty
        spokenHint = resolved.contains { $0.id == ApprovalChoice.allowSession }
            ? "Or say “yes”, “yes, for this session” or “no”."
            : "Or say “yes” or “no”."
    }

    // MARK: Choices

    static let defaultLabels = [ApprovalChoice.allowOnce: "Do it",
                                ApprovalChoice.allowSession: "Allow this for the rest of the session",
                                ApprovalChoice.deny: "Don't"]

    /// The server's choices in its order, each id once. An older server offered none and takes yes or no: "Do it" and
    /// "Don't". A "Don't" is always there, so the person can refuse on screen whatever the server sent.
    static func resolveChoices(_ offered: [ApprovalChoice], effect: ApprovalEffect?) -> [Choice] {
        var seen = Set<String>()
        var list = offered.filter { seen.insert($0.id).inserted }
        if list.isEmpty { list = [ApprovalChoice(id: ApprovalChoice.allowOnce), ApprovalChoice(id: ApprovalChoice.deny)] }
        if !list.contains(where: { $0.id == ApprovalChoice.deny }) { list.append(ApprovalChoice(id: ApprovalChoice.deny)) }
        var primaryTaken = false
        return list.map { c in
            let role: Choice.Role
            if c.id == ApprovalChoice.deny {
                role = .deny
            } else if c.allows && !primaryTaken {
                role = .primary
                primaryTaken = true
            } else {
                role = .secondary
            }
            let label = nonEmpty(c.label) ?? defaultLabels[c.id]
                ?? c.id.replacingOccurrences(of: "_", with: " ").capitalizedFirst
            return Choice(id: c.id, label: label, allows: c.allows, role: role,
                          destructive: role == .primary && effect == .delete)
        }
    }

    // MARK: Words

    static func words(_ effect: ApprovalEffect) -> String {
        switch effect {
        case .create: return "Creates"
        case .modify: return "Changes"
        case .delete: return "Deletes"
        case .run: return "Runs a command"
        case .network: return "Uses the network"
        case let .other(s): return s.capitalizedFirst
        }
    }

    static func pathLabel(_ effect: ApprovalEffect?) -> String {
        switch effect {
        case .create?: return "New file"
        case .modify?: return "File to change"
        case .delete?: return "File to delete"
        default: return "File"
        }
    }

    /// `confirm_cancel.why` in words for the note in the conversation. The orchestrator sends codes
    /// (2026-10-05): "answered" (by voice), "timeout" (silence is no), "overtaken" (the run ended with the question
    /// open); anything else is shown as written.
    public static func withdrawal(_ why: String) -> (words: String, timedOut: Bool) {
        switch why.trimmingCharacters(in: .whitespaces).lowercased() {
        case "answered": return ("answered by voice", false)
        case "timeout", "timed out": return ("no answer in time", true)
        case "overtaken": return ("overtaken by what came next", false)
        default: return (why, false)
        }
    }

    static func nonEmpty(_ s: String?) -> String? {
        guard let t = s?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
        return t
    }

    // MARK: Previews

    /// A unified diff has hunk headers ("@@ -1,2 +1,2 @@"); "+" and "-" alone also start Markdown lists.
    public static func isDiff(_ text: String) -> Bool {
        text.split(whereSeparator: \.isNewline).contains { $0.hasPrefix("@@ -") }
    }

    public enum DiffLine: String, Sendable, Equatable { case header, hunk, added, removed, context }

    /// Each line of a unified diff with its kind: before the first hunk, `---`/`+++` and the like are headers; after
    /// it, a line's first character says what it is (so a removed "-- note" is not mistaken for a header).
    public static func diffLines(_ text: String) -> [(DiffLine, Substring)] {
        var inHunk = false
        return text.split(separator: "\n", omittingEmptySubsequences: false).map { line in
            if line.hasPrefix("@@") {
                inHunk = true
                return (.hunk, line)
            }
            if !inHunk {
                let header = ["--- ", "+++ ", "diff ", "index "].contains { line.hasPrefix($0) }
                return (header ? .header : .context, line)
            }
            if line.hasPrefix("+") { return (.added, line) }
            if line.hasPrefix("-") { return (.removed, line) }
            return (.context, line)
        }
    }

    /// The server cuts a long preview at 4,000 characters and says so in a marker at its end (PROTOCOL.md); a scroll
    /// box would hide it below the fold, so the card pins it. The words are the server's: matched is a last line that
    /// starts with "…", "...", "[" or "(", or a bracketed end of the last line, that speaks of a cut. Below 2,000
    /// characters a preview was never cut (2,000, not 4,000: the server may count UTF-16 units, as JavaScript does).
    public static func cutMarker(in preview: String) -> String? {
        guard preview.unicodeScalars.count >= 2_000,
              let last = preview.split(whereSeparator: \.isNewline).last(where: { $0.contains { !$0.isWhitespace } })
        else { return nil }
        let line = last.trimmingCharacters(in: .whitespaces)
        var candidate = Substring(line)
        if !["…", "...", "[", "("].contains(where: { line.hasPrefix($0) }) {
            // "…the last words … [cut: 2,180 more characters]": the bracketed end, with an ellipsis just before it.
            guard line.last == "]" || line.last == ")",
                  let open = line.lastIndex(where: { $0 == "[" || $0 == "(" }) else { return nil }
            var start = open
            let before = line[..<open].trimmingCharacters(in: .whitespaces)
            if before.hasSuffix("…") || before.hasSuffix("...") {
                start = line.range(of: before.hasSuffix("…") ? "…" : "...", options: .backwards,
                                   range: line.startIndex..<open)?.lowerBound ?? open
            }
            candidate = line[start...]
        }
        guard candidate.count <= 160 else { return nil }
        let lower = candidate.lowercased()
        let words = ["cut", "truncat", "omitted", "not shown", "more characters", "more lines", "more bytes"]
        return words.contains(where: lower.contains) ? String(candidate) : nil
    }
}

/// A question on screen: the request, its card, and when it came.
public struct PendingApproval: Sendable, Equatable, Identifiable {
    public let request: ConfirmRequest
    public let content: ApprovalContent
    public let receivedAt: Date

    public init(request: ConfirmRequest, receivedAt: Date) {
        self.request = request
        self.content = ApprovalContent(request)
        self.receivedAt = receivedAt
    }

    public var id: ScalarID { request.id }

    /// When the server stops waiting (silence is no); nil when the request gave no time.
    public var deadline: Date? {
        guard let ms = request.timeoutMs, ms > 0 else { return nil }
        return receivedAt.addingTimeInterval(Double(ms) / 1000)
    }

    /// Whole seconds left, rounded up; 0 once the time is up.
    public func secondsLeft(at now: Date) -> Int? {
        deadline.map { max(0, Int(($0.timeIntervalSince(now)).rounded(.up))) }
    }
}

/// The questions waiting for the person, oldest first; the card shows the first.
public struct ApprovalQueue: Sendable, Equatable {
    /// How long after its time runs out a card stays when no `confirm_cancel` comes (a server from before Approvals
    /// sends none): long enough for a current server's cancel, with its reason, to arrive first.
    public static let grace: TimeInterval = 3

    public private(set) var waiting: [PendingApproval] = []

    public init() {}

    public var current: PendingApproval? { waiting.first }

    public enum Received: String, Sendable { case new, repeated }

    /// A request whose id is already waiting is the server asking again: it replaces the old one in place, and its
    /// time starts again.
    @discardableResult
    public mutating func receive(_ request: ConfirmRequest, at now: Date) -> Received {
        let approval = PendingApproval(request: request, receivedAt: now)
        if let i = waiting.firstIndex(where: { $0.id == request.id }) {
            waiting[i] = approval
            return .repeated
        }
        waiting.append(approval)
        return .new
    }

    public struct Answer: Sendable, Equatable {
        public let approval: PendingApproval
        public let choice: ApprovalContent.Choice
        /// The `confirm_response` to send: `confirmed` true for any allow and false for deny, and the choice's id when
        /// the server offered choices.
        public let message: ClientMessage
    }

    /// The person picked one of a card's choices: the question leaves the queue and `message` goes to the server. Nil,
    /// and nothing to send, when the question is no longer waiting or the choice is not on its card.
    public mutating func answer(id: ScalarID, choice: String) -> Answer? {
        guard let i = waiting.firstIndex(where: { $0.id == id }),
              let picked = waiting[i].content.choices.first(where: { $0.id == choice }) else { return nil }
        let approval = waiting.remove(at: i)
        let message = ClientMessage.confirmResponse(id: id, confirmed: picked.allows,
                                                    choice: approval.content.choicesOffered ? picked.id : nil)
        return Answer(approval: approval, choice: picked, message: message)
    }

    /// `confirm_cancel`: the question is withdrawn. Nil when it was not waiting (answered here already).
    public mutating func cancel(id: ScalarID) -> PendingApproval? {
        guard let i = waiting.firstIndex(where: { $0.id == id }) else { return nil }
        return waiting.remove(at: i)
    }

    /// The questions whose time ran out at least `grace` ago with no `confirm_cancel`, closed: silence is no.
    public mutating func expire(now: Date, grace: TimeInterval = ApprovalQueue.grace) -> [PendingApproval] {
        let gone = waiting.filter { a in a.deadline.map { now >= $0.addingTimeInterval(grace) } ?? false }
        waiting.removeAll { a in gone.contains { $0.id == a.id } }
        return gone
    }

    /// Every waiting question, closed: the connection dropped, and the orchestrator ends the turn that asked (a
    /// question it still holds after a resume comes back as a new `confirm_request`).
    public mutating func closeAll() -> [PendingApproval] {
        defer { waiting.removeAll() }
        return waiting
    }

    /// When `expire` next has something to close.
    public func nextExpiry(grace: TimeInterval = ApprovalQueue.grace) -> Date? {
        waiting.compactMap { $0.deadline?.addingTimeInterval(grace) }.min()
    }
}

// MARK: Event log

extension PendingApproval {
    /// One `approval` line for the JSONL log (the apps' and lvclient's): `shown` carries the whole card, as the e2e
    /// tests check it; `answered` and `closed` name the question and add their own fields.
    public func logFields(_ phase: String, _ extra: [(String, JSONValue)] = []) -> [(String, JSONValue)] {
        var o: [(String, JSONValue)] = [("event", "approval"), ("phase", .string(phase)), ("id", id.json),
                                        ("headline", .string(content.headline))]
        if phase == "shown" {
            let c = content
            o += [
                ("legacy", .bool(c.isLegacy)),
                ("effect", c.effect.map { .string($0.wire) } ?? .null),
                ("facts", .array(c.facts.map(JSONValue.string))),
                ("exact", .array(c.exact.map { .object([("kind", .string($0.kind.rawValue)),
                                                         ("label", .string($0.label)), ("text", .string($0.text))]) })),
                ("cwd", c.cwd.map(JSONValue.string) ?? .null),
                ("preview", c.preview.map { p in
                    .object([("kind", .string(p.kind.rawValue)), ("label", .string(p.label)),
                             ("chars", .int(p.text.count)), ("cut_marker", p.cutMarker.map(JSONValue.string) ?? .null)])
                } ?? .null),
                ("choices", .array(c.choices.map { .object([("id", .string($0.id)), ("label", .string($0.label)),
                                                            ("role", .string($0.role.rawValue)),
                                                            ("allows", .bool($0.allows))]) })),
                ("choices_offered", .bool(c.choicesOffered)),
                ("timeout_ms", request.timeoutMs.map(JSONValue.int) ?? .null),
                ("spoken_hint", .string(c.spokenHint)),
            ]
        }
        return o + extra
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
