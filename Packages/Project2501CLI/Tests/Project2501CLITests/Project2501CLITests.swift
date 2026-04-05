//
//  Project2501CLITests.swift
//  project2501
//
//  Unit tests for the Project2501 CLI core functionality.
//

import XCTest
@testable import Project2501CLICore

final class Project2501CLITests: XCTestCase {
    func testConfiguration() {
        // Just a smoke test to ensure things link
        let root = Configuration.toolsRootDirectory()
        XCTAssertFalse(root.path.isEmpty)
    }
}
