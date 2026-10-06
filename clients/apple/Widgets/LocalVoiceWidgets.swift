import ActivityKit
import AppIntents
import SwiftUI
import WidgetKit

@main
struct LocalVoiceWidgets: WidgetBundle {
    var body: some Widget {
        VoiceLiveActivity()
    }
}

struct VoiceLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: VoiceActivityAttributes.self) { context in
            LockScreenView(state: context.state)
                .activityBackgroundTint(Color.black.opacity(0.6))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: context.state.symbol)
                        .font(.title2)
                        .foregroundStyle(.tint)
                }
                DynamicIslandExpandedRegion(.center) {
                    VStack(spacing: 1) {
                        Text(context.state.status)
                            .font(.headline)
                            .lineLimit(1)
                        if let whereLine = context.state.whereLine {
                            Text(whereLine)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Button(intent: EndSessionIntent()) {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("End the conversation")
                }
                DynamicIslandExpandedRegion(.bottom) {
                    if let question = context.state.question {
                        QuestionLine(question: question, deadline: context.state.questionDeadline)
                    } else if let reply = context.state.lastReply {
                        Text(reply)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
            } compactLeading: {
                Image(systemName: context.state.symbol)
            } compactTrailing: {
                if context.state.question != nil, let deadline = context.state.questionDeadline, deadline > .now {
                    Text(timerInterval: Date.now...deadline, countsDown: true)
                        .font(.caption2.monospacedDigit())
                        .frame(maxWidth: 44)
                } else {
                    Text(context.state.status)
                        .font(.caption2)
                        .lineLimit(1)
                        .frame(maxWidth: 64)
                }
            } minimal: {
                Image(systemName: context.state.symbol)
            }
        }
    }
}

struct LockScreenView: View {
    let state: VoiceActivityAttributes.ContentState

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: state.symbol)
                .font(.title)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(state.status)
                        .font(.headline)
                    if let whereLine = state.whereLine {
                        Text(whereLine)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                if let question = state.question {
                    QuestionLine(question: question, deadline: state.questionDeadline)
                } else if let reply = state.lastReply {
                    Text(reply)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer()
            Button(intent: EndSessionIntent()) {
                Label("End", systemImage: "xmark")
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}

/// A question waits (PROTOCOL.md "Approvals"): what it is, and the time left, counting down without updates. The
/// card and its buttons are in the app; a tap on the Live Activity opens it.
struct QuestionLine: View {
    let question: String
    let deadline: Date?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(question)
                .font(.caption)
                .lineLimit(2)
            HStack(spacing: 4) {
                Text("Open to answer, or say yes or no")
                if let deadline, deadline > .now {
                    Text("·")
                    Text(timerInterval: Date.now...deadline, countsDown: true)
                        .monospacedDigit()
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }
}
