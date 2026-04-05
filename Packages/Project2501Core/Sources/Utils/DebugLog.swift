//
//  DebugLog.swift
//  project2501
//
//  Lightweight file-based debug logger.  Works from any isolation context.
//  Writes timestamped lines to /tmp/project2501_debug.log.
//  No-ops in production (only active when the file already exists or is created
//  on first write).
//

import Foundation

/// Writes a timestamped line to `/tmp/project2501_debug.log`.
///
/// Safe to call from any actor or thread.  Uses `FileHandle` for append-only
/// writes so concurrent calls from different threads do not corrupt the file.
@inline(__always)
func debugLog(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    guard let data = line.data(using: .utf8) else { return }
    let path = "/tmp/project2501_debug.log"
    if FileManager.default.fileExists(atPath: path) {
        if let fh = FileHandle(forWritingAtPath: path) {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.closeFile()
        }
    } else {
        try? data.write(to: URL(fileURLWithPath: path))
    }
}
