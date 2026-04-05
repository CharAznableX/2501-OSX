// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Project2501CLI",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "project2501-cli", targets: ["Project2501CLI"]),
        .library(name: "Project2501CLICore", targets: ["Project2501CLICore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.12.0"),
        .package(path: "../Project2501Repository"),
    ],
    targets: [
        .executableTarget(
            name: "Project2501CLI",
            dependencies: [
                "Project2501CLICore"
            ]
        ),
        .target(
            name: "Project2501CLICore",
            dependencies: [
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "Project2501Repository", package: "Project2501Repository"),
            ]
        ),
        .testTarget(
            name: "Project2501CLITests",
            dependencies: ["Project2501CLICore"]
        ),
    ]
)
