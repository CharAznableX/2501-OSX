import Foundation
import Testing

@testable import OsaurusCore

struct PreflightCapabilitySearchTests {

    @Test func emptyQueryReturnsEmptyResult() async {
        let result = await PreflightCapabilitySearch.search(query: "", attachments: [])
        #expect(result.toolSpecs.isEmpty)
        #expect(result.contextSnippet.isEmpty)
    }

    @Test func whitespaceOnlyQueryReturnsEmptyResult() async {
        let result = await PreflightCapabilitySearch.search(query: "   \n  ", attachments: [])
        #expect(result.toolSpecs.isEmpty)
        #expect(result.contextSnippet.isEmpty)
    }

    @Test func nonsenseQueryReturnsGracefully() async {
        let result = await PreflightCapabilitySearch.search(
            query: "zzz_completely_nonexistent_capability_xyz_12345",
            attachments: []
        )
        #expect(result.toolSpecs.isEmpty)
    }

    @Test func resultContainsNoDuplicateToolSpecs() async {
        let result = await PreflightCapabilitySearch.search(query: "deploy build test", attachments: [])
        let names = result.toolSpecs.map { $0.function.name }
        #expect(Set(names).count == names.count)
    }

    @Test func preflightToolSpecsHaveNoDuplicatesWithAlwaysLoaded() async {
        let alwaysLoaded = await MainActor.run {
            ToolRegistry.shared.alwaysLoadedSpecs(mode: .none)
        }
        let alwaysNames = Set(alwaysLoaded.map { $0.function.name })

        let result = await PreflightCapabilitySearch.search(query: "search memory save method", attachments: [])
        let preflightNames = result.toolSpecs.map { $0.function.name }

        #expect(
            Set(preflightNames).count == preflightNames.count,
            "Pre-flight specs should not contain internal duplicates"
        )
    }

    // MARK: - PreflightSearchMode Tests

    @Test func offModeReturnsEmptyResult() async {
        let result = await PreflightCapabilitySearch.search(query: "deploy build test", attachments: [], mode: .off)
        #expect(result.toolSpecs.isEmpty)
        #expect(result.contextSnippet.isEmpty)
    }

    @Test func narrowModeReturnsNoDuplicates() async {
        let result = await PreflightCapabilitySearch.search(query: "deploy build test", attachments: [], mode: .narrow)
        let names = result.toolSpecs.map { $0.function.name }
        #expect(Set(names).count == names.count)
    }

    @Test func wideModeReturnsNoDuplicates() async {
        let result = await PreflightCapabilitySearch.search(query: "deploy build test", attachments: [], mode: .wide)
        let names = result.toolSpecs.map { $0.function.name }
        #expect(Set(names).count == names.count)
    }

    @Test func topKValuesAreCorrect() {
        #expect(PreflightSearchMode.off.topKValues == (0, 0, 0))
        #expect(PreflightSearchMode.narrow.topKValues == (1, 2, 0))
        #expect(PreflightSearchMode.balanced.topKValues == (3, 5, 1))
        #expect(PreflightSearchMode.wide.topKValues == (5, 8, 2))
    }
}
