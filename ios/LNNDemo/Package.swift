
// swift-tools-version: 5.7

import PackageDescription

let package = Package(
    name: "LNNDemo",
    platforms: [
        .iOS(.v16),
    ],
    products: [
        .executable(
            name: "LNNDemo",
            targets: ["LNNDemo"]
        ),
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "LNNDemo",
            dependencies: [],
            path: "LNNDemo"
        ),
    ]
)
