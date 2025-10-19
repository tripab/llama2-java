package org.llama2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * The Sampler, which takes logits and returns a sampled token
 * sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
 */
public class Sampler {
    // struct used when sorting probabilities during top-p sampling
    record ProbIndex(float prob, int index) {
    }

    int vocabSize;
    List<ProbIndex> probIndices; // buffer used in top-p sampling
    float temperature;
    float topP;
    long rngState;

    public Sampler(int vocabSize, float temperature, float topP, long rngSeed) {
        this.vocabSize = vocabSize;
        this.temperature = temperature;
        this.topP = topP;
        this.rngState = rngSeed;
        // buffer only used with nucleus sampling; may not need but it's relatively small
        this.probIndices = new ArrayList<>();
    }

    int sampleArgMax(float[] probabilities) {
        int maxInd = 0;
        float maxP = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxP) {
                maxP = probabilities[i];
                maxInd = i;
            }
        }

        return maxInd;
    }

    int sampleMult(float[] probabilities, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1)
        float cdf = 0f;
        for (int i = 0; i < probabilities.length; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }

        return probabilities.length - 1;
    }

    /**
     * top-p sampling (or "nucleus sampling") samples from the smallest set of
     * tokens that exceed probability top-p. This way we never sample tokens that
     * have very low probabilities and are less likely to go "off the rails".
     *
     * @param coin a random number in [0, 1)
     * @return the topP value
     */
    int sampleTopP(float[] probabilities, float coin) {
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        final float cutoff = (1.0f - topP) / (probabilities.length - 1);
        probIndices.clear();
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= cutoff) {
                probIndices.add(new ProbIndex(probabilities[i], i));
            }
        }
        probIndices.sort(Comparator.comparing(ProbIndex::prob).reversed());

        // truncate the list where cumulative probability exceeds topp
        float cumulativeProb = 0.0f;
        int lastIdx = probIndices.size() - 1; // in case of rounding errors consider all elements
        for (int i = 0; i < probIndices.size(); i++) {
            cumulativeProb += probIndices.get(i).prob();
            if (cumulativeProb > topP) {
                lastIdx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = coin * cumulativeProb;
        float cdf = 0f;
        for (int i = 0; i <= lastIdx; i++) {
            cdf += probIndices.get(i).prob();
            if (r < cdf) {
                return probIndices.get(i).index();
            }
        }

        return probIndices.get(lastIdx).index(); // in case of rounding errors
    }

    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    int randomU32() {
        rngState ^= (rngState >>> 12);
        rngState ^= (rngState << 25);
        rngState ^= (rngState >>> 27);
        long result = rngState * 0x2545F4914F6CDD1DL;  // 64-bit multiplier
        return (int) (result >>> 32);               // upper 32 bits
    }

    // Equivalent of: float random_f32(unsigned long long* state)
    float randomF32() {
        int r = randomU32();
        // shift right by 8 to get 24 random bits
        int value = (r >>> 8) & 0xFFFFFF;
        return value / 16777216.0f; // 2^24 = 16777216
    }

    int sample(float[] logits) {
        // sample the token given the logits and some hyperparameters
        int next;
        if (this.temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            next = sampleArgMax(logits);
        } else {
            // apply the temperature to the logits
            for (int q = 0; q < this.vocabSize; q++) {
                logits[q] /= this.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            NNUtils.softmax(logits, this.vocabSize);
            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = randomF32();
            // we sample from this distribution to get the next token
            if (this.topP <= 0 || this.topP >= 1) {
                // simply sample from the predicted probability distribution
                next = sampleMult(logits, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sampleTopP(logits, coin);
            }
        }

        return next;
    }
}
