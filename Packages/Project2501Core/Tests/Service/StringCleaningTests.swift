import Foundation
import Testing

@testable import Project2501Core

struct StringCleaningTests {
    @Test
    func stripGeminiDisplayMetadata_removesGeminiSignatureMarkers() {
        let input = "\u{200B}ts:CiQabcDEF123+/=_\u{200B}Dependencies installed."
        let cleaned = StringCleaning.stripGeminiDisplayMetadata(input)

        #expect(cleaned == "Dependencies installed.")
    }

    @Test
    func stripGeminiDisplayMetadata_removesVisibleLeakedSignatureTokens() {
        let input = "ts:CiQbvj72+49RKk4lfHalZIoEXp8c2HsTTVB9c3ugC9IWty4E1FQKdAG+Pvb7T6Kk0wzT0GD Dependencies installed."
        let cleaned = StringCleaning.stripGeminiDisplayMetadata(input)

        #expect(cleaned == "Dependencies installed.")
    }
}
