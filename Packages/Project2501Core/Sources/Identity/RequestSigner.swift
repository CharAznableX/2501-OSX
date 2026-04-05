//
//  RequestSigner.swift
//  project2501
//
//  Handles API request signing with master key and device attestation.
//

import CryptoKit
import Foundation
import LocalAuthentication

public struct RequestSigner: Sendable {

    // MARK: - Setup

    /// Full identity setup: generates Master Key, attests device, generates recovery code.
    public static func setup() async throws -> IdentityInfo {
        let project2501Id = try MasterKey.generate()
        let deviceId = try await DeviceKey.attest()
        let recovery = RecoveryManager.configure(address: project2501Id)

        return IdentityInfo(
            project2501Id: project2501Id,
            deviceId: deviceId,
            recovery: recovery
        )
    }

    /// Whether an identity already exists (no biometric prompt).
    public static func exists() -> Bool {
        MasterKey.exists()
    }

    // MARK: - Request Signing

    /// Sign an API request as the user identity.
    /// Returns a URLRequest with `Authorization: Bearer <token>`.
    public static func signRequest(
        method: String,
        path: String,
        audience: String
    ) async throws -> URLRequest {
        let context = Project2501IdentityContext.biometric()
        let project2501Id = try MasterKey.getProject2501Id(context: context)

        return try await buildSignedRequest(
            project2501Id: project2501Id,
            method: method,
            path: path,
            audience: audience,
            context: context
        )
    }

    // MARK: - Private

    private static func buildSignedRequest(
        project2501Id: Project2501ID,
        method: String,
        path: String,
        audience: String,
        context: LAContext
    ) async throws -> URLRequest {
        let deviceId = try DeviceKey.currentDeviceId()
        let counter = CounterStore.shared.next()
        let now = Int(Date().timeIntervalSince1970)

        let payload = TokenPayload(
            iss: project2501Id,
            dev: deviceId,
            cnt: counter,
            iat: now,
            exp: now + 60,
            aud: audience,
            act: "\(method) \(path)",
            par: nil,
            idx: nil
        )

        let payloadData = try JSONEncoder().encode(payload)

        // Layer 1: Identity signature (secp256k1)
        let identitySig = try MasterKey.sign(payload: payloadData, context: context)

        // Layer 2: Device assertion (App Attest)
        let payloadHash = Data(SHA256.hash(data: payloadData))
        let deviceAssertion = try await DeviceKey.assert(payloadHash: payloadHash)

        // Assemble 4-part token
        let headerData = try JSONEncoder().encode(TokenHeader.current)
        let token = [
            headerData.base64urlEncoded,
            payloadData.base64urlEncoded,
            identitySig.hexEncodedString,
            deviceAssertion.base64urlEncoded,
        ].joined(separator: ".")

        var request = URLRequest(url: URL(string: "https://\(audience)\(path)")!)
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return request
    }
}