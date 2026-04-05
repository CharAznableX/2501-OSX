//
//  Method.swift
//  project2501
//
//  Models for the methods subsystem: recorded tool-call sequences,
//  scoring events, and computed scores.
//

import Foundation

// MARK: - Method

public struct Method: Identifiable, Codable, Sendable {
    public let id: String
    public var name: String
    public var description: String
    public var triggerText: String?
    public var body: String
    public var source: MethodSource
    public var sourceModel: String?
    public var tier: MethodTier
    public var toolsUsed: [String]
    public var skillsUsed: [String]
    public var tokenCount: Int
    public var version: Int
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String = UUID().uuidString,
        name: String,
        description: String,
        triggerText: String? = nil,
        body: String,
        source: MethodSource,
        sourceModel: String? = nil,
        tier: MethodTier = .active,
        toolsUsed: [String] = [],
        skillsUsed: [String] = [],
        tokenCount: Int = 0,
        version: Int = 1,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.triggerText = triggerText
        self.body = body
        self.source = source
        self.sourceModel = sourceModel
        self.tier = tier
        self.toolsUsed = toolsUsed
        self.skillsUsed = skillsUsed
        self.tokenCount = tokenCount
        self.version = version
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

// MARK: - MethodSource

public enum MethodSource: String, Codable, Sendable {
    case user
}

// MARK: - MethodTier

public enum MethodTier: String, Codable, Sendable {
    case active
}

// MARK: - MethodEventType

public enum MethodEventType: String, Codable, Sendable {
    case loaded
    case succeeded
    case failed
}

// MARK: - MethodEvent

public struct MethodEvent: Identifiable, Sendable {
    public let id: Int
    public let methodId: String
    public let eventType: MethodEventType
    public let modelUsed: String?
    /// For `.loaded` events this stores the issue ID, linking the method to the work session.
    public let agentId: String?
    public let notes: String?
    public let createdAt: Date

    public init(
        id: Int = 0,
        methodId: String,
        eventType: MethodEventType,
        modelUsed: String? = nil,
        agentId: String? = nil,
        notes: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.methodId = methodId
        self.eventType = eventType
        self.modelUsed = modelUsed
        self.agentId = agentId
        self.notes = notes
        self.createdAt = createdAt
    }
}

// MARK: - MethodScore

public struct MethodScore: Sendable {
    public let methodId: String
    public var timesLoaded: Int
    public var timesSucceeded: Int
    public var timesFailed: Int
    public var successRate: Double
    public var lastUsedAt: Date?
    public var score: Double

    public init(
        methodId: String,
        timesLoaded: Int = 0,
        timesSucceeded: Int = 0,
        timesFailed: Int = 0,
        successRate: Double = 0.0,
        lastUsedAt: Date? = nil,
        score: Double = 0.0
    ) {
        self.methodId = methodId
        self.timesLoaded = timesLoaded
        self.timesSucceeded = timesSucceeded
        self.timesFailed = timesFailed
        self.successRate = successRate
        self.lastUsedAt = lastUsedAt
        self.score = score
    }

    /// Recomputes `successRate` and `score` from the current counts and `lastUsedAt`.
    public mutating func recalculate() {
        let total = timesSucceeded + timesFailed
        successRate = total > 0 ? Double(timesSucceeded) / Double(total) : 0.0

        let daysSinceUsed: Double
        if let last = lastUsedAt {
            daysSinceUsed = max(0, Date().timeIntervalSince(last) / 86400.0)
        } else {
            daysSinceUsed = 365
        }
        let recencyWeight = 1.0 / (1.0 + daysSinceUsed / 30.0)
        score = successRate * recencyWeight
    }
}

// MARK: - MethodSearchResult

public struct MethodSearchResult: Sendable {
    public let method: Method
    public let searchScore: Float
    public let score: Double

    public init(method: Method, searchScore: Float, score: Double) {
        self.method = method
        self.searchScore = searchScore
        self.score = score
    }
}
