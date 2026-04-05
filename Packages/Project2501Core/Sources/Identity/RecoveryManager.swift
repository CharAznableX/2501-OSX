//
//  RecoveryManager.swift
//  project2501
//
//  Generates a one-time recovery code at identity creation.
//  The plaintext is shown once, then discarded. Server stores bcrypt hash only (Phase 1b).
//

import Foundation
import Security

public struct RecoveryManager: Sendable {

    /// Generate a recovery code for the given Project2501 ID.
    /// The code is returned in `RecoveryInfo`; caller must present it to the user
    /// and discard it from memory afterward.
    public static func configure(address: Project2501ID) -> RecoveryInfo {
        let code = generateRecoveryCode()
        return RecoveryInfo(code: code)
    }

    /// Format: PROJECT2501-XXXX-XXXX-XXXX-XXXX (uppercase hex, 4 groups of 4).
    private static func generateRecoveryCode() -> String {
        var bytes = [UInt8](repeating: 0, count: 8)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)

        let hex = bytes.map { String(format: "%02X", $0) }.joined()
        let chunks = stride(from: 0, to: hex.count, by: 4).map { i -> String in
            let start = hex.index(hex.startIndex, offsetBy: i)
            let end = hex.index(start, offsetBy: 4)
            return String(hex[start ..< end])
        }
        return "PROJECT2501-\(chunks.joined(separator: "-"))"
    }
}
