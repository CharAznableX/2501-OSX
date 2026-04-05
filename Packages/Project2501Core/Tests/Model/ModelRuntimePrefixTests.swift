//
//  ModelRuntimePrefixTests.swift
//  project2501Tests
//
//  Tests for the background prefix-cache build path in ModelRuntime.
//
//  The key invariant: buildPrefixCache (always called from a background
//  prefix-cache task) awaits activeGenerationTask before touching the GPU,
//  and never overwrites it.  This prevents concurrent Metal command buffer
//  submissions that cause EXC_BAD_ACCESS / SIGSEGV on Apple Silicon.
//
//  The actor isolation on ModelRuntime ensures all state mutations are
//  serialised, so generateEventStream and buildPrefixCache never interleave
//  within a single suspension point.
//

import Foundation
import Testing

@testable import Project2501Core

/// Verifies that ModelRuntime exposes the API surface expected by the
/// prefix-cache GPU serialisation fix.
struct ModelRuntimePrefixTests {

    /// ModelRuntime must be an actor so that generateEventStream,
    /// buildPrefixCache, and the stale-task cleanup are serialised.
    @Test func modelRuntimeIsAnActor() {
        #expect(ModelRuntime.self is any Actor.Type)
    }
}
