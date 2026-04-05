import Foundation
import Testing

@testable import Project2501Core

@MainActor
struct ContentBlockDisplayTests {
    @Test
    func assistantVisibleContent_hidesGeminiMetadata() {
        let assistant = ChatTurn(
            role: .assistant,
            content: "\u{200B}ts:CiQabcDEF123+/=_\u{200B}Dependencies installed."
        )

        #expect(assistant.visibleContent == "Dependencies installed.")
    }

    @Test
    func assistantParagraphs_useVisibleContent() {
        let assistant = ChatTurn(
            role: .assistant,
            content: "\u{200B}ts:CiQabcDEF123+/=_\u{200B}Dependencies installed."
        )

        let blocks = ContentBlock.generateBlocks(
            from: [assistant],
            streamingTurnId: nil,
            agentName: "Assistant"
        )

        let paragraphText = blocks.compactMap { block -> String? in
            guard case let .paragraph(_, text, _, _) = block.kind else { return nil }
            return text
        }.first

        #expect(paragraphText == "Dependencies installed.")
    }

    @Test
    func userVisibleContent_preservesOriginalText() {
        let user = ChatTurn(role: .user, content: "ts:debug-token should stay visible for user content")

        #expect(user.visibleContent == "ts:debug-token should stay visible for user content")
    }

    @Test
    func userParagraphs_preserveOriginalText() {
        let user = ChatTurn(role: .user, content: "ts:debug-token should stay visible for user content")

        let blocks = ContentBlock.generateBlocks(
            from: [user],
            streamingTurnId: nil,
            agentName: "Assistant"
        )

        let userText = blocks.compactMap { block -> String? in
            guard case let .userMessage(text, _) = block.kind else { return nil }
            return text
        }.first

        #expect(userText == "ts:debug-token should stay visible for user content")
    }
}
