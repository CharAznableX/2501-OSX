//
//  MemoryLogger.swift
//  project2501
//
//  Structured logger for the memory subsystem using os.Logger.
//  Zero-cost when not collected; filterable in Console.app / Instruments.
//

import Foundation
import os

public enum MemoryLogger {
    static let service = Logger(subsystem: "ai.project2501", category: "memory.service")
    static let search = Logger(subsystem: "ai.project2501", category: "memory.search")
    static let database = Logger(subsystem: "ai.project2501", category: "memory.database")
    static let config = Logger(subsystem: "ai.project2501", category: "memory.config")
}
