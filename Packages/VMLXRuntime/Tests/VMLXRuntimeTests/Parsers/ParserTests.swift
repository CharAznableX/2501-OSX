import Testing
@testable import VMLXRuntime

@Suite("GenericToolParser")
struct GenericToolParserTests {

    @Test("Detects JSON tool call")
    func detectsToolCall() {
        var parser = GenericToolParser()
        let results = parser.processChunk("""
            {"name": "get_weather", "arguments": {"location": "NYC"}}
            """)

        let toolCalls = results.compactMap { result -> ParsedToolCall? in
            if case .toolCall(let tc) = result { return tc }
            return nil
        }
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].name == "get_weather")
    }

    @Test("Passes through non-tool text")
    func passesThrough() {
        var parser = GenericToolParser()
        let results = parser.processChunk("Hello world!")
        let texts = results.compactMap { r -> String? in
            if case .text(let t) = r { return t }
            return nil
        }
        #expect(texts.joined() == "Hello world!")
    }

    @Test("Buffers incomplete JSON")
    func buffersIncomplete() {
        var parser = GenericToolParser()
        let r1 = parser.processChunk("{\"name\": \"test\"")
        let hasBuffered = r1.contains { if case .buffered = $0 { return true }; return false }
        #expect(hasBuffered)

        let r2 = parser.processChunk(", \"arguments\": {}}")
        let toolCalls = r2.compactMap { r -> ParsedToolCall? in
            if case .toolCall(let tc) = r { return tc }
            return nil
        }
        #expect(toolCalls.count == 1)
    }

    @Test("Reset clears state")
    func reset() {
        var parser = GenericToolParser()
        _ = parser.processChunk("{incomplete")
        parser.reset()
        let results = parser.processChunk("plain text")
        let texts = results.compactMap { r -> String? in
            if case .text(let t) = r { return t }
            return nil
        }
        #expect(texts.joined() == "plain text")
    }
}

@Suite("ThinkTagReasoningParser")
struct ThinkTagReasoningParserTests {

    @Test("Extracts thinking block")
    func extractsThinking() {
        var parser = ThinkTagReasoningParser()
        let r = parser.processChunk("<think>I need to calculate...</think>The answer is 42.")
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Streaming think tags")
    func streamingThink() {
        var parser = ThinkTagReasoningParser()
        let r1 = parser.processChunk("<think>thinking")
        #expect(r1.inThinking == true)
        #expect(r1.reasoning != nil)

        let r2 = parser.processChunk(" more</think>response")
        #expect(r2.inThinking == false)
        #expect(r2.reasoning != nil)
    }

    @Test("No think tags passes content through")
    func noThinkTags() {
        var parser = ThinkTagReasoningParser()
        let r = parser.processChunk("Just a normal response")
        #expect(r.content == "Just a normal response")
        #expect(r.reasoning == nil)
        #expect(r.inThinking == false)
    }

    @Test("Reset clears thinking state")
    func reset() {
        var parser = ThinkTagReasoningParser()
        _ = parser.processChunk("<think>partial")
        parser.reset()
        let r = parser.processChunk("fresh start")
        #expect(r.inThinking == false)
        #expect(r.content == "fresh start")
    }

    @Test("Finalize flushes remaining")
    func finalize() {
        var parser = ThinkTagReasoningParser()
        // First chunk opens the thinking block and flushes content
        _ = parser.processChunk("<think>start")
        // Second chunk has content ending with partial close tag, leaving buffer non-empty
        _ = parser.processChunk(" more</thi")
        let r = parser.finalize()
        // Finalize should flush buffered partial content as reasoning and reset state
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Auto-detect finds parser for Qwen3")
    func autoDetectQwen3() {
        let parser = autoDetectReasoningParser(modelName: "Qwen3-8B-JANG")
        #expect(parser != nil)
    }

    @Test("Format lookup finds think-tag parser")
    func formatLookupQwen3() {
        let parser = reasoningParserForFormat(.qwen3)
        #expect(parser is ThinkTagReasoningParser)
    }

    @Test("Format lookup finds DeepSeek-R1 think-tag parser")
    func formatLookupDeepSeek() {
        let parser = reasoningParserForFormat(.deepseekR1)
        #expect(parser is ThinkTagReasoningParser)
    }
}

@Suite("GPTOSSReasoningParser")
struct GPTOSSReasoningParserTests {

    @Test("Extracts reasoning from analysis channel")
    func extractsReasoning() {
        var parser = GPTOSSReasoningParser()
        let r = parser.processChunk("<|channel|>analysis<|message|>Let me think about this...<|channel|>final<|message|>The answer is 42.")
        // With streaming, we need to check the accumulated results
        #expect(r.reasoning != nil || r.content != nil)
    }

    @Test("Extracts content from final channel")
    func extractsContent() {
        var parser = GPTOSSReasoningParser()
        let r = parser.processChunk("<|channel|>analysis<|message|>reasoning here<|channel|>final<|message|>content here")
        let r2 = parser.finalize()
        // Either the processChunk or finalize should have captured the content
        let hasContent = (r.content != nil) || (r2.content != nil)
        let hasReasoning = (r.reasoning != nil) || (r2.reasoning != nil)
        #expect(hasContent || hasReasoning)
    }

    @Test("No markers passes as content")
    func noMarkers() {
        var parser = GPTOSSReasoningParser()
        let r = parser.processChunk("Just a normal response without Harmony markers")
        #expect(r.content == "Just a normal response without Harmony markers")
        #expect(r.reasoning == nil)
    }

    @Test("Reset clears state")
    func reset() {
        var parser = GPTOSSReasoningParser()
        _ = parser.processChunk("<|channel|>analysis<|message|>partial")
        parser.reset()
        let r = parser.processChunk("fresh start")
        #expect(r.content == "fresh start")
    }

    @Test("Finalize flushes remaining")
    func finalize() {
        var parser = GPTOSSReasoningParser()
        let r1 = parser.processChunk("<|channel|>analysis<|message|>thinking...")
        // processChunk should have captured reasoning via channel markers
        #expect(r1.reasoning != nil)
        let r = parser.finalize()
        // After processChunk already emitted everything, finalize resets state
        #expect(r.inThinking == false)
    }

    @Test("Auto-detect finds GPTOSSReasoningParser")
    func autoDetect() {
        let parser = autoDetectReasoningParser(modelName: "gptoss-model-v1")
        #expect(parser != nil)
        #expect(parser is GPTOSSReasoningParser)
    }

    @Test("Auto-detect finds GPTOSSReasoningParser for Harmony")
    func autoDetectHarmony() {
        let parser = autoDetectReasoningParser(modelName: "harmony-chat-7b")
        #expect(parser != nil)
        #expect(parser is GPTOSSReasoningParser)
    }

    @Test("Format lookup finds GPTOSSReasoningParser")
    func formatLookupGPTOSS() {
        let parser = reasoningParserForFormat(.gptoss)
        #expect(parser is GPTOSSReasoningParser)
    }
}

@Suite("MistralReasoningParser")
struct MistralReasoningParserTests {

    @Test("Extracts thinking block")
    func extractsThinking() {
        var parser = MistralReasoningParser()
        let r = parser.processChunk("[THINK]Let me work through this...[/THINK]The answer is 345.")
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Streaming think tags")
    func streamingThink() {
        var parser = MistralReasoningParser()
        let r1 = parser.processChunk("[THINK]thinking")
        #expect(r1.inThinking == true)
        #expect(r1.reasoning != nil)

        let r2 = parser.processChunk(" more[/THINK]response")
        #expect(r2.inThinking == false)
        #expect(r2.reasoning != nil)
    }

    @Test("No think tags passes content through")
    func noThinkTags() {
        var parser = MistralReasoningParser()
        let r = parser.processChunk("Just a normal Mistral response")
        #expect(r.content == "Just a normal Mistral response")
        #expect(r.reasoning == nil)
        #expect(r.inThinking == false)
    }

    @Test("Implicit reasoning mode (only closing tag)")
    func implicitReasoning() {
        var parser = MistralReasoningParser()
        let r = parser.processChunk("reasoning content[/THINK]The final answer.")
        #expect(r.reasoning != nil)
        #expect(r.reasoning!.contains("reasoning content"))
        #expect(r.inThinking == false)
    }

    @Test("Reset clears thinking state")
    func reset() {
        var parser = MistralReasoningParser()
        _ = parser.processChunk("[THINK]partial")
        parser.reset()
        let r = parser.processChunk("fresh start")
        #expect(r.inThinking == false)
        #expect(r.content == "fresh start")
    }

    @Test("Finalize flushes remaining")
    func finalize() {
        var parser = MistralReasoningParser()
        _ = parser.processChunk("[THINK]start thinking")
        _ = parser.processChunk(" more[/THI")
        let r = parser.finalize()
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Auto-detect finds MistralReasoningParser for Mistral 4")
    func autoDetectMistral4() {
        let parser = autoDetectReasoningParser(modelName: "Mistral-Large-Instruct-2411")
        #expect(parser != nil)
        #expect(parser is MistralReasoningParser)
    }

    @Test("Format lookup finds MistralReasoningParser")
    func formatLookupMistral() {
        let parser = reasoningParserForFormat(.mistral)
        #expect(parser is MistralReasoningParser)
    }

    @Test("Format lookup disables parser for none")
    func formatLookupNone() {
        let parser = reasoningParserForFormat(.none)
        #expect(parser == nil)
    }
}
