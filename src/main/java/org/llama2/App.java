package org.llama2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class App {
    enum Mode {
        GENERATE, CHAT
    }

    private static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler,
                                 String prompt, int steps) {
        if (prompt == null) {
            prompt = "";
        }
        // encode the (string) prompt into tokens sequence
        int[] promptTokens = new int[prompt.length() + 3]; // +3 for '\0', ?BOS, ?EOS
        int numPromptTokens = tokenizer.encode(prompt, true, false, promptTokens);
        if (numPromptTokens < 1) {
            System.out.println("something is wrong, expected at least 1 prompt token.");
            System.exit(-1);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = promptTokens[0]; // kick off with the first token in the prompt
        int pos = 0;     // position in the sequence
        while (pos < steps) {
            // forward the transformer to get logits for the next token
            float[] logits = transformer.forward(token, pos);
            // advance the state machine
            if (pos < numPromptTokens - 1) {
                // if we are still processing the input prompt, force the next prompt token
                next = promptTokens[pos + 1];
            } else {
                // otherwise sample the next token from the logits
                next = sampler.sample(logits);
            }
            pos++;

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if (next == 1)
                break;
            tokenizer.safePrint(tokenizer.decode(token, next));
            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0) {
                start = System.currentTimeMillis();
            }
        }
        System.out.println();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1) {
            long end = System.currentTimeMillis();
            System.out.printf("achieved tok/s: %f\n", (pos - 1) / (double) (end - start) * 1000);
        }
    }

    private static void chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler,
                             String cliUserPrompt, String cliSystemPrompt, int steps) throws IOException {
        String systemPrompt = "", userPrompt, renderedPrompt;
        int numPromptTokens;
        int[] promptTokens = new int[1152];
        int userIndex;

        // start the main loop
        boolean userTurn = true;
        int next = 0; // will store the next token in the sequence
        int token; // stores the current token to feed into the transformer
        int pos = 0; // position in the sequence
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while (pos < steps) {
            // when it is the user's turn to contribute tokens to the dialog...
            if (userTurn) {
                // get the (optional) system prompt at position 0
                if (pos == 0) {
                    // at position 0, the user can also contribute a system prompt
                    if (cliSystemPrompt == null || cliSystemPrompt.isEmpty()) {
                        // system prompt was not passed in, attempt to get it from stdin
                        System.out.println("Enter system prompt (optional): ");
                        systemPrompt = br.readLine();
                    } else {
                        // system prompt was passed in, use it
                        systemPrompt = cliSystemPrompt;
                    }
                }
                // get the user prompt
                if (pos == 0 && (cliUserPrompt != null && !cliUserPrompt.isEmpty())) {
                    // user prompt for position 0 was passed in, use it
                    userPrompt = cliUserPrompt;
                } else {
                    // otherwise get user prompt from stdin
                    System.out.println("User: ");
                    userPrompt = br.readLine();
                }
                // render user/system prompts into the Llama 2 Chat schema
                if (pos == 0 && systemPrompt.isEmpty()) {
                    renderedPrompt = "[INST] <<SYS>>\n" +
                            systemPrompt +
                            "\n<</SYS>>\n\n" +
                            userPrompt +
                            " [/INST]";
                } else {
                    renderedPrompt = "[INST] " +
                            userPrompt +
                            " [/INST]";
                }
                // encode the rendered prompt into tokens
                numPromptTokens = tokenizer.encode(renderedPrompt, true, false, promptTokens);
                userIndex = 0;
                userTurn = false;
                System.out.println("Assistant: ");

                // determine the token to pass into the transformer next
                if (userIndex < numPromptTokens) {
                    // if we are still processing the input prompt, force the next prompt token
                    token = promptTokens[userIndex++];
                } else {
                    // otherwise use the next token sampled from previous turn
                    token = next;
                }
                // EOS (=2) token ends the Assistant turn
                if (token == 2) {
                    userTurn = true;
                }

                // forward the transformer to get logits for the next token
                float[] logits = transformer.forward(token, pos);
                next = sampler.sample(logits);
                pos++;

                if (userIndex >= numPromptTokens && next != 2) {
                    // the Assistant is responding, so print its output
                    tokenizer.safePrint(tokenizer.decode(token, next));
                }
                if (next == 2) {
                    System.out.println();
                }
            }
            System.out.println();
        }
    }

    static void main(String[] args) {
        String checkpointPath = "";// e.g. out/model.bin
        String tokenizerPath = "tokenizer.bin";
        float temperature = 1f;// 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topP = .9f;// top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        int steps = 256; // number of steps to run for
        String prompt = ""; // prompt string
        long rngSeed = 0;// seed rng with time by default
        Mode mode = Mode.GENERATE;
        String systemPrompt = ""; // the (optional) system prompt to use in chat mode

        if (args.length >= 1)
            checkpointPath = args[0];
        else {
            printUsage();
            System.exit(-1);
        }
        for (int i = 1; i < args.length; i += 2) {
            if (i >= args.length - 1) {
                printUsage();
            } // must have arg after flag
            if (args[i].charAt(0) != '-') {
                printUsage();
            } // must start with dash
            if (args[i].length() != 2) {
                printUsage();
            } // must be -x (one dash, one letter)
            if (args[i].charAt(1) == 't') {
                temperature = Float.parseFloat(args[i + 1]);
            } else if (args[i].charAt(1) == 'p') {
                topP = Float.parseFloat(args[i + 1]);
            } else if (args[i].charAt(1) == 's') {
                rngSeed = Integer.parseInt(args[i + 1]);
            } else if (args[i].charAt(1) == 'n') {
                steps = Integer.parseInt(args[i + 1]);
            } else if (args[i].charAt(1) == 'i') {
                prompt = args[i + 1];
            } else if (args[i].charAt(1) == 'z') {
                tokenizerPath = args[i + 1];
            } else if (args[i].charAt(1) == 'm') {
                mode = Mode.valueOf(args[i + 1]);
            } else if (args[i].charAt(1) == 'y') {
                systemPrompt = args[i + 1];
            } else {
                printUsage();
            }
        }

        // parameter validation/overrides
        if (rngSeed <= 0) {
            rngSeed = System.currentTimeMillis() / 1000;
        }
        if (temperature < 0f) {
            temperature = 0f;
        }
        if (topP < 0f || topP > 1f) {
            topP = .9f;
        }
        if (steps < 0) {
            steps = 0;
        }

        // build the Transformer via the model .bin file
        try {
            Transformer transformer = new Transformer(checkpointPath);
            if (steps == 0 || steps > transformer.config.seqLen()) {
                steps = transformer.config.seqLen(); // override to ~max length
            }
            try {
                // build the Tokenizer via the tokenizer .bin file
                Tokenizer tokenizer = new Tokenizer(tokenizerPath, transformer.config.vocabSize());
                // build the Sampler
                Sampler sampler = new Sampler(transformer.config.vocabSize(), temperature, topP, rngSeed);

                // run!
                if (mode == Mode.GENERATE) {
                    generate(transformer, tokenizer, sampler, prompt, steps);
                } else if (mode == Mode.CHAT) {
                    chat(transformer, tokenizer, sampler, prompt, systemPrompt, steps);
                } else {
                    System.err.println("unknown mode: " + mode);
                    printUsage();
                }
            } catch (IOException e) {
                throw new RuntimeException("Couldn't open file at " + tokenizerPath);
            }

            transformer.cleanup();
        } catch (IOException e) {
            throw new RuntimeException("Couldn't open file at " + checkpointPath);
        }
    }

    private static void printUsage() {
        System.out.print("Usage:   java -jar App.jar <checkpoint> [options]\n");
        System.out.print("Example: java -jar App.jar model.bin -n 256 -i \"Once upon a time\"\n");
        System.out.print("Options:\n");
        System.out.print("  -t <float>  temperature in [0,inf], default 1.0\n");
        System.out.print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
        System.out.print("  -s <int>    random seed, default time(NULL)\n");
        System.out.print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
        System.out.print("  -i <string> input prompt\n");
        System.out.print("  -z <string> optional path to custom tokenizer\n");
        System.out.print("  -m <string> mode: generate|chat, default: generate\n");
        System.out.print("  -y <string> (optional) system prompt in chat mode\n");
    }
}
