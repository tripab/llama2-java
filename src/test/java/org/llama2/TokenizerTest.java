package org.llama2;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TokenizerTest {
    private static Tokenizer tokenizer;
    private static String tokenizerPath;
    private static int vocabSize;

    private static Logger logger = LoggerFactory.getLogger(TokenizerTest.class);

    @BeforeAll
    public static void setup() throws IOException {
        tokenizerPath = "tokenizer.bin";
        vocabSize = 32000;
        tokenizer = new Tokenizer(tokenizerPath, vocabSize);
    }

    private void testPromptEncoding(String prompt, int[] expectedTokens) {
        int[] promptTokens = new int[prompt.length() + 3];
        int numPromptTokens = tokenizer.encode(prompt, true, false, promptTokens);
        if (logger.isDebugEnabled()) {
            logger.debug("expected tokens:");
            for (int i = 0; i < expectedTokens.length; i++) {
                logger.debug("{}", expectedTokens[i]);
            }
            logger.debug("actual tokens:");
            for (int i = 0; i < expectedTokens.length; i++) {
                logger.debug("{}", promptTokens[i]);
            }
        }
        assertTrue(numPromptTokens == expectedTokens.length);
        assertTrue(IntStream.range(0, expectedTokens.length)
                .allMatch(i -> expectedTokens[i] == promptTokens[i]));
        logger.debug("OK");
    }

    /**
     * the tests below are taken from the Meta Llama 2 repo example code
     * https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
     * and the expected tokens come from me breaking in the debugger in Python
     */
    @Test
    public void testNonEmptyPromptEncodings() {
        String prompt = "I believe the meaning of life is";
        int[] expectedTokens = new int[]{1, 306, 4658, 278, 6593, 310, 2834, 338};
        testPromptEncoding(prompt, expectedTokens);

        prompt = "Simply put, the theory of relativity states that ";
        expectedTokens = new int[]{1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215,
                537, 5922, 393, 29871};
        testPromptEncoding(prompt, expectedTokens);

        prompt = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
        expectedTokens = new int[]{1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815,
                373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706,
                306, 925, 29871};
        testPromptEncoding(prompt, expectedTokens);

        prompt = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
        expectedTokens = new int[]{1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706,
                7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407,
                837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878,
                330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923,
                968, 1149};
        testPromptEncoding(prompt, expectedTokens);
    }
}
