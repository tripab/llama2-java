package org.llama2;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Comparator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
 */
public class Tokenizer {
    record TokenIndex(String str, int id) {
    }

    final String[] vocab;
    float[] vocabScores;
    TokenIndex[] sortedVocab;
    int vocabSize;
    int maxTokenLength;
    StringBuffer bytePieces; // stores all single-byte strings

    public Tokenizer(String tokenizerPath, int vocabSize)
            throws IOException {
        // TODO: write the vocabSize into the tokenizer file
        this.vocabSize = vocabSize;
        vocab = new String[vocabSize];
        vocabScores = new float[vocabSize];
        bytePieces = new StringBuffer();
        for (int i = 0; i < 256; i++) {
            bytePieces.append((char) i);
        }
        try (RandomAccessFile file = new RandomAccessFile(tokenizerPath, "r");
             FileChannel fileChannel = file.getChannel()) {
            ByteBuffer buf = ByteBuffer.allocate(4);
            int bytesRead = fileChannel.read(buf);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.flip();
            IntBuffer maxTokenLengthBytes = buf.asIntBuffer();
            if (bytesRead == 4) {
                maxTokenLength = maxTokenLengthBytes.get();
                ByteBuffer scoreBuf = ByteBuffer.allocate(4);
                ByteBuffer lenBuf = ByteBuffer.allocate(4);
                ByteBuffer vocabBuf;
                for (int i = 0; i < vocabSize; i++) {
                    scoreBuf.clear();
                    bytesRead = fileChannel.read(scoreBuf);
                    scoreBuf.order(ByteOrder.LITTLE_ENDIAN);
                    scoreBuf.flip();
                    if (bytesRead == 4) {
                        vocabScores[i] = scoreBuf.asFloatBuffer().get();
                    }
                    lenBuf.clear();
                    bytesRead = fileChannel.read(lenBuf);
                    lenBuf.order(ByteOrder.LITTLE_ENDIAN);
                    lenBuf.flip();
                    if (bytesRead == 4) {
                        int len = lenBuf.asIntBuffer().get();
                        vocabBuf = ByteBuffer.allocate(len);
                        bytesRead = fileChannel.read(vocabBuf);
                        vocabBuf.order(ByteOrder.LITTLE_ENDIAN);
                        vocabBuf.flip();
                        if (bytesRead == len) {
                            vocab[i] = StandardCharsets.UTF_8.decode(vocabBuf).toString();
                        }
                    }
                }
            }
        }
    }

    String decode(int prevToken, int token) {
        int vocabTokenPos = 0;
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prevToken == 1 && vocab[token].charAt(vocabTokenPos) == ' ') {
            vocabTokenPos++;
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        Pattern p = Pattern.compile("<0x([0-9A-Fa-f]{2})>");
        Matcher m = p.matcher(vocab[token].substring(vocabTokenPos));
        if (m.matches()) {
            int byteVal = Integer.parseInt(m.group(1), 16); // parse as hex
            return bytePieces.substring(byteVal, byteVal + 2);
        }

        return vocab[token].substring(vocabTokenPos);
    }

    void safePrint(String piece) {
        if (piece == null || piece.isEmpty()) {
            return;
        }
        // If the string has exactly one character, check if it's printable or whitespace
        if (piece.length() == 1) {
            char ch = piece.charAt(0);

            // Printable ASCII: 32–126 (inclusive)
            boolean isPrintable = (ch >= 32 && ch <= 126);
            boolean isWhitespace = Character.isWhitespace(ch);

            if (!(isPrintable || isWhitespace)) {
                return; // Non-printable single-byte, skip
            }
        }

        System.out.printf("%s", piece);
    }

    int encode(String text, boolean bos, boolean eos, int[] tokens) {
        // encode the string text (input) into an upper-bound preallocated tokens[] array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
        if (text == null) {
            System.err.println("cannot encode NULL text");
            System.exit(-1);
        }

        if (sortedVocab == null) {
            sortedVocab = new TokenIndex[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                sortedVocab[i] = new TokenIndex(vocab[i], i);
            }
            Arrays.sort(sortedVocab, Comparator.comparing(TokenIndex::str));
        }

        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        StringBuffer strBuffer = new StringBuffer(); // (t->max_token_length*2 +1 +2) * sizeof(char));
        int nTokens = 0;

        // add optional BOS (=1) token, if desired
        if (bos) {
            tokens[nTokens++] = 1;
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if (!text.isEmpty()) {
            final String SINGLE_SPACE = " ";
            int dummyPrefix = Arrays.binarySearch(sortedVocab, new TokenIndex(SINGLE_SPACE, -1),
                    Comparator.comparing(TokenIndex::str));
            tokens[nTokens++] = sortedVocab[dummyPrefix].id();
        }

        // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
        // Code point ↔ UTF-8 conversion
        // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
        // U+0000	U+007F	    0xxxxxxx
        // U+0080	U+07FF	    110xxxxx	10xxxxxx
        // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
        // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

        // process the raw (UTF-8) byte sequence of the input string
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            // reset buffer if the current byte is ASCII or a leading byte
            // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
            // 0x80 is 10000000
            // in UTF-8, all continuation bytes start with "10" in first two bits
            // so in English this is: "if this byte is not a continuation byte"
            if ((ch & 0xC0) != 0x80) {
                // this byte must be either a leading byte (11...) or an ASCII char (0x...)
                // => reset our location, as we're starting a new UTF-8 codepoint
                strBuffer.setLength(0);
            }

            // append the current byte to the buffer
            strBuffer.append(ch);

            // while the next character is a continuation byte, continue appending
            // but if there are too many of them, just stop to avoid overruning str_buffer size.
            if (((ch + 1) & 0xC0) == 0x80 && strBuffer.length() < 4) {
                continue;
            }

            // ok c+1 is not a continuation byte, so we've read in a full codepoint
            int index = Arrays.binarySearch(sortedVocab, new TokenIndex(strBuffer.toString(), -1),
                    Comparator.comparing(TokenIndex::str));
            int id = index >= 0 ? sortedVocab[index].id() : -1;
            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[nTokens++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (int j = 0; j < strBuffer.length(); j++) {
                    tokens[nTokens++] = strBuffer.charAt(j) + 3;
                }
            }
            strBuffer.setLength(0); // protect against a sequence of stray UTF8 continuation bytes
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float bestScore = (float) -1E10;
            int bestId = -1;
            int bestIdx = -1;

            for (int i = 0; i < nTokens - 1; i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.setLength(0);
                strBuffer.append(vocab[tokens[i]])
                        .append(vocab[tokens[i + 1]]);
                int index = Arrays.binarySearch(sortedVocab, new TokenIndex(strBuffer.toString(), -1),
                        Comparator.comparing(TokenIndex::str));
                int id = index >= 0 ? sortedVocab[index].id() : -1;
                if (id >= 0 && vocabScores[id] > bestScore) {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id];
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1)
                break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[bestIdx] = bestId;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = bestIdx + 1; i < nTokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            nTokens--;
        }

        // add optional EOS (=2) token, if desired
        if (eos) {
            tokens[nTokens++] = 2;
        }

        return nTokens;
    }
}
