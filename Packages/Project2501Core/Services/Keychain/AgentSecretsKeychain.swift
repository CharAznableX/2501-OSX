//
//  AgentSecretsKeychain.swift
//  project2501
//
//  Secure Keychain storage for agent-level secrets (API keys, tokens, etc.).
//  Unlike ToolSecretsKeychain which is plugin-scoped, this stores secrets
//  per-agent only, making them accessible to any sandbox plugin.
//

import Foundation
import Security

/// Keychain wrapper for agent-scoped secret storage.
/// Account format: `"{agentId}.{key}"` — no plugin scoping.
public enum AgentSecretsKeychain {
    private static let service = "ai.project2501.agent-secrets"

    @discardableResult
    public static func saveSecret(_ value: String, id: String, agentId: UUID) -> Bool {
        let account = "\(agentId.uuidString).\(id)"
        guard let valueData = value.data(using: .utf8) else { return false }

        deleteSecret(id: id, agentId: agentId)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: valueData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    public static func getSecret(id: String, agentId: UUID) -> String? {
        let account = "\(agentId.uuidString).\(id)"

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
            let data = result as? Data
        else { return nil }

        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    public static func deleteSecret(id: String, agentId: UUID) -> Bool {
        let account = "\(agentId.uuidString).\(id)"

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    /// Enumerates accounts then fetches each value individually.
    public static func getAllSecrets(agentId: UUID) -> [String: String] {
        let prefix = "\(agentId.uuidString)."

        var secrets: [String: String] = [:]
        for account in allAccounts() where account.hasPrefix(prefix) {
            let key = String(account.dropFirst(prefix.count))
            if let value = getSecret(id: key, agentId: agentId) {
                secrets[key] = value
            }
        }
        return secrets
    }

    public static func deleteAllSecrets(agentId: UUID) {
        let prefix = "\(agentId.uuidString)."
        for account in allAccounts() where account.hasPrefix(prefix) {
            let query: [String: Any] = [
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service,
                kSecAttrAccount as String: account,
            ]
            SecItemDelete(query as CFDictionary)
        }
    }

    // MARK: - Environment Safety

    /// Env var names that must never be overridden by user-defined secrets.
    private static let reservedEnvVarNames: Set<String> = [
        "PATH", "HOME", "SHELL", "USER", "LOGNAME",
        "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
        "VIRTUAL_ENV", "PROJECT2501_PLUGIN",
    ]

    /// Returns agent secrets with reserved env var names stripped out.
    static func getFilteredSecrets(agentId: UUID) -> [String: String] {
        getAllSecrets(agentId: agentId).filter { !reservedEnvVarNames.contains($0.key) }
    }

    /// Returns merged agent + plugin secrets with reserved names stripped out.
    /// Plugin secrets override agent secrets of the same name.
    static func mergedSecretsEnvironment(agentId: UUID, pluginId: String) -> [String: String] {
        var env = getFilteredSecrets(agentId: agentId)
        let pluginSecrets =
            ToolSecretsKeychain
            .getAllSecrets(for: pluginId, agentId: agentId)
            .filter { !reservedEnvVarNames.contains($0.key) }
        env.merge(pluginSecrets) { _, new in new }
        return env
    }

    // MARK: - Private

    private static func allAccounts() -> [String] {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecMatchLimit as String: kSecMatchLimitAll,
            kSecReturnAttributes as String: true,
        ]

        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
            let items = result as? [[String: Any]]
        else { return [] }

        return items.compactMap { $0[kSecAttrAccount as String] as? String }
    }
}
