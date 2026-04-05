//
//  ToolIndexServiceTests.swift
//  project2501
//
//  Unit tests for ToolDatabase (CRUD, upsert, runtime field) and
//  ToolIndexService's buildCompactIndex logic.
//

import Foundation
import Testing

@testable import Project2501Core

struct ToolDatabaseTests {

    private func makeTempDB() throws -> ToolDatabase {
        let db = ToolDatabase()
        try db.openInMemory()
        return db
    }

    private func sampleEntry(
        id: String = "test-tool",
        name: String = "test-tool",
        description: String = "A test tool",
        runtime: ToolRuntime = .builtin
    ) -> ToolIndexEntry {
        ToolIndexEntry(
            id: id,
            name: name,
            description: description,
            runtime: runtime,
            toolsJSON: "{}",
            source: .system,
            tokenCount: 50
        )
    }

    // MARK: - Insert and Query

    @Test func upsertAndLoadEntryRoundtrip() throws {
        let db = try makeTempDB()
        let entry = sampleEntry()
        try db.upsertEntry(entry)

        let loaded = try db.loadEntry(id: entry.id)
        #expect(loaded != nil)
        #expect(loaded?.id == entry.id)
        #expect(loaded?.name == entry.name)
        #expect(loaded?.description == entry.description)
        #expect(loaded?.runtime == .builtin)
        #expect(loaded?.tokenCount == 50)
    }

    @Test func upsertOverwritesExisting() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry(description: "Version 1"))
        try db.upsertEntry(sampleEntry(description: "Version 2"))

        let loaded = try db.loadEntry(id: "test-tool")
        #expect(loaded?.description == "Version 2")

        let count = try db.entryCount()
        #expect(count == 1)
    }

    @Test func deleteEntryRemovesFromDB() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry())
        #expect(try db.loadEntry(id: "test-tool") != nil)

        try db.deleteEntry(id: "test-tool")
        #expect(try db.loadEntry(id: "test-tool") == nil)
    }

    @Test func loadAllEntriesReturnsAll() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry(id: "a", name: "alpha"))
        try db.upsertEntry(sampleEntry(id: "b", name: "beta"))
        try db.upsertEntry(sampleEntry(id: "c", name: "gamma"))

        let all = try db.loadAllEntries()
        #expect(all.count == 3)
    }

    @Test func entryCountIsAccurate() throws {
        let db = try makeTempDB()
        #expect(try db.entryCount() == 0)

        try db.upsertEntry(sampleEntry(id: "a", name: "alpha"))
        try db.upsertEntry(sampleEntry(id: "b", name: "beta"))
        #expect(try db.entryCount() == 2)
    }

    @Test func deleteAllClearsTable() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry(id: "a", name: "alpha"))
        try db.upsertEntry(sampleEntry(id: "b", name: "beta"))
        #expect(try db.entryCount() == 2)

        try db.deleteAll()
        #expect(try db.entryCount() == 0)
    }

    @Test func loadEntryNotFoundReturnsNil() throws {
        let db = try makeTempDB()
        #expect(try db.loadEntry(id: "nonexistent") == nil)
    }

    // MARK: - Runtime Field

    @Test func runtimeFieldStoredCorrectly() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry(id: "native-tool", runtime: .native))
        try db.upsertEntry(sampleEntry(id: "sandbox-tool", runtime: .sandbox))
        try db.upsertEntry(sampleEntry(id: "builtin-tool", runtime: .builtin))
        try db.upsertEntry(sampleEntry(id: "mcp-tool", runtime: .mcp))

        #expect(try db.loadEntry(id: "native-tool")?.runtime == .native)
        #expect(try db.loadEntry(id: "sandbox-tool")?.runtime == .sandbox)
        #expect(try db.loadEntry(id: "builtin-tool")?.runtime == .builtin)
        #expect(try db.loadEntry(id: "mcp-tool")?.runtime == .mcp)
    }

    @Test func mcpRuntimePersistsAndRoundtrips() throws {
        let db = try makeTempDB()
        let entry = ToolIndexEntry(
            id: "github_search",
            name: "github_search",
            description: "Search GitHub repositories via MCP",
            runtime: .mcp,
            toolsJSON: "{}",
            source: .system,
            tokenCount: 40
        )
        try db.upsertEntry(entry)

        let loaded = try db.loadEntry(id: "github_search")
        #expect(loaded != nil)
        #expect(loaded?.runtime == .mcp)
        #expect(loaded?.name == "github_search")
        #expect(loaded?.description == "Search GitHub repositories via MCP")
    }

    // MARK: - Migrations

    @Test func openInMemoryCreatesSchema() throws {
        let db = try makeTempDB()
        try db.upsertEntry(sampleEntry())
        let entries = try db.loadAllEntries()
        #expect(entries.count == 1)
    }

    // MARK: - Source Field

    @Test func sourceFieldPersists() throws {
        let db = try makeTempDB()
        let entry = ToolIndexEntry(
            id: "manual-tool",
            name: "manual-tool",
            description: "Manually added",
            runtime: .native,
            source: .manual,
            tokenCount: 25
        )
        try db.upsertEntry(entry)

        let loaded = try db.loadEntry(id: "manual-tool")
        #expect(loaded?.source == .manual)
    }

    @Test func communitySourceFieldPersists() throws {
        let db = try makeTempDB()
        let entry = ToolIndexEntry(
            id: "community-tool",
            name: "community-tool",
            description: "From community",
            runtime: .native,
            source: .community,
            tokenCount: 30
        )
        try db.upsertEntry(entry)

        let loaded = try db.loadEntry(id: "community-tool")
        #expect(loaded?.source == .community)
    }

    @Test func allSourceTypesDistinct() throws {
        let db = try makeTempDB()
        try db.upsertEntry(
            ToolIndexEntry(
                id: "sys",
                name: "sys",
                description: "system",
                runtime: .builtin,
                source: .system
            )
        )
        try db.upsertEntry(
            ToolIndexEntry(
                id: "man",
                name: "man",
                description: "manual",
                runtime: .native,
                source: .manual
            )
        )
        try db.upsertEntry(
            ToolIndexEntry(
                id: "comm",
                name: "comm",
                description: "community",
                runtime: .native,
                source: .community
            )
        )

        #expect(try db.loadEntry(id: "sys")?.source == .system)
        #expect(try db.loadEntry(id: "man")?.source == .manual)
        #expect(try db.loadEntry(id: "comm")?.source == .community)
        #expect(try db.entryCount() == 3)
    }

}
