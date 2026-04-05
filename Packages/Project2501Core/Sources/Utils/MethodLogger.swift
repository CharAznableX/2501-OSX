//
//  MethodLogger.swift
//  project2501
//
//  Structured logger for the methods subsystem using os.Logger.
//  Zero-cost when not collected; filterable in Console.app / Instruments.
//

import Foundation
import os

public enum MethodLogger {
    static let service = Logger(subsystem: "ai.project2501", category: "method.service")
    static let search = Logger(subsystem: "ai.project2501", category: "method.search")
    static let database = Logger(subsystem: "ai.project2501", category: "method.database")
}
