// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Project2501Repository",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Project2501Repository", targets: ["Project2501Repository"])
    ],
    targets: [
        .target(
            name: "Project2501Repository",
            path: "."
        )
    ]
)
