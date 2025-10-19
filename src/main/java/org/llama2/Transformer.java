package org.llama2;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.IntStream;

class Transformer {
    record Config(int dim, // transformer dimension
                  int hiddenDim, // for ffn layers
                  int nLayers, // number of layers
                  int nHeads, // number of query heads
                  int nKVHeads, // number of key/value heads (can be < query heads because of multiquery)
                  int vocabSize, // vocabulary size, usually 256 (byte-level)
                  int seqLen // max sequence length
    ) {
    }

    class TransformerWeights {
        // token embedding table
        int tokenEmbeddingTableStartPos;
        float[] tokenEmbeddingTableRow = new float[config.dim()]; // float[][] tokenEmbeddingTable; // (vocabSize, dim)
        // weights for NNUtils.rmsnorms
        int rmsAttnWeightStartPos;
        float[] rmsAttnWeightRow = new float[config.dim()]; // float[][] rmsAttnWeight; // (layer, dim) NNUtils.rmsnorm weights
        int rmsFFNWeightStartPos;
        float[] rmsFFNWeightRow = new float[config.dim()]; // float[][] rmsFFNWeight; // (layer, dim)
        // weights for NNUtils.matmuls, note dim == nHeads * head_size
        int wqStartPos;
        float[][] wqMatrix = new float[config.dim()][config.dim()]; // float[][][] wq; // (layer, dim, nHeads*head_size)
        int wkStartPos;
        float[][] wkMatrix = new float[config.dim()][config.dim()]; // float[][][] wk; // (layer, dim, nKVHeads*head_size)
        int wvStartPos;
        float[][] wvMatrix = new float[config.dim()][config.dim()]; // float[][][] wv; // (layer, dim, nKVHeads*head_size)
        int woStartPos;
        float[][] woMatrix = new float[config.dim()][config.dim()]; // float[][][] wo; // (layer, nHeads*head_size, dim)
        // weights for ffn
        int w1StartPos;
        float[][] w1Matrix = new float[config.hiddenDim()][config.dim()]; // float[][][] w1; // (layer, hiddenDim, dim)
        int w2StartPos;
        float[][] w2Matrix = new float[config.dim()][config.hiddenDim()]; // float[][][] w2; // (layer, dim, hiddenDim)
        int w3StartPos;
        float[][] w3Matrix = new float[config.hiddenDim()][config.dim()]; // float[][][] w3; // (layer, hiddenDim, dim)
        // final NNUtils.rmsnorm
        int rmsFinalWeightStartPos;
        float[] rmsFinalWeightRow = new float[config.dim()]; // float[][] rmsFinalWeight; // (dim,)
        boolean rmsFinalWeightRowInitialized = false;
        // (optional) classifier weights for the logits, on the last layer
        int wclsStartPos;
        float[][] wclsMatrix = new float[config.vocabSize()][config.dim()]; // float[][] wcls; // (vocabSize, dim)
        boolean wclsMatrixInitialized = false;

        float[] tokenEmbeddingTableRow(int row) {
            int tokenEmbeddingTableRowPos = tokenEmbeddingTableStartPos + row * config.dim();
            for (int col = 0; col < config.dim(); col++) {
                tokenEmbeddingTableRow[col] = data.get(tokenEmbeddingTableRowPos + col);
            }

            return tokenEmbeddingTableRow;
        }

        float[] rmsAttnWeightRow(int row) {
            int rmsAttnWeightRowPos = rmsAttnWeightStartPos + row * config.dim();
            for (int col = 0; col < config.dim(); col++) {
                rmsAttnWeightRow[col] = data.get(rmsAttnWeightRowPos + col);
            }

            return rmsAttnWeightRow;
        }

        float[] rmsFFNWeightRow(int row) {
            int rmsAttnWeightRowPos = rmsFFNWeightStartPos + row * config.dim();
            for (int col = 0; col < config.dim(); col++) {
                rmsFFNWeightRow[col] = data.get(rmsAttnWeightRowPos + col);
            }

            return rmsFFNWeightRow;
        }

        float[][] wqMatrix(int matrixStartPos) {
            int wqRowPos = wqStartPos + (matrixStartPos * config.dim() * config.dim());
            for (int i = 0; i < config.dim(); i++) {
                for (int j = 0; j < config.dim(); j++) {
                    wqMatrix[i][j] = data.get(wqRowPos + i * config.dim() + j);
                }
            }

            return wqMatrix;
        }

        float[][] wkMatrix(int matrixStartPos) {
            int wkRowPos = wkStartPos + (matrixStartPos * config.dim() *
                    (config.nKVHeads() * config.dim() / config.nHeads()));
            for (int i = 0; i < config.dim(); i++) {
                for (int j = 0; j < (config.nKVHeads() * config.dim() / config.nHeads()); j++) {
                    wkMatrix[i][j] = data.get(wkRowPos + i * config.dim() + j);
                }
            }

            return wkMatrix;
        }

        float[][] wvMatrix(int matrixStartPos) {
            int wvRowPos = wvStartPos + (matrixStartPos * config.dim() *
                    (config.nKVHeads() * config.dim() / config.nHeads()));
            for (int i = 0; i < config.dim(); i++) {
                for (int j = 0; j < (config.nKVHeads() * config.dim() / config.nHeads()); j++) {
                    wvMatrix[i][j] = data.get(wvRowPos + i * config.dim() + j);
                }
            }

            return wvMatrix;
        }

        float[][] woMatrix(int matrixStartPos) {
            int woRowPos = woStartPos + (matrixStartPos * config.dim() * config.dim());
            for (int i = 0; i < config.dim(); i++) {
                for (int j = 0; j < config.dim(); j++) {
                    woMatrix[i][j] = data.get(woRowPos + i * config.dim() + j);
                }
            }

            return woMatrix;
        }

        float[][] w1Matrix(int matrixStartPos) {
            int w1RowPos = w1StartPos + (matrixStartPos * config.hiddenDim() * config.dim());
            for (int i = 0; i < config.hiddenDim(); i++) {
                for (int j = 0; j < config.dim(); j++) {
                    w1Matrix[i][j] = data.get(w1RowPos + i * config.dim() + j);
                }
            }

            return w1Matrix;
        }

        float[][] w2Matrix(int matrixStartPos) {
            int w2RowPos = w2StartPos + (matrixStartPos * config.dim() * config.hiddenDim());
            for (int i = 0; i < config.dim(); i++) {
                for (int j = 0; j < config.hiddenDim(); j++) {
                    w2Matrix[i][j] = data.get(w2RowPos + i * config.hiddenDim() + j);
                }
            }

            return w2Matrix;
        }

        float[][] w3Matrix(int matrixStartPos) {
            int w3RowPos = w3StartPos + (matrixStartPos * config.hiddenDim() * config.dim());
            for (int i = 0; i < config.hiddenDim(); i++) {
                for (int j = 0; j < config.dim(); j++) {
                    w3Matrix[i][j] = data.get(w3RowPos + i * config.dim() + j);
                }
            }

            return w3Matrix;
        }

        float[] rmsFinalWeightRow() {
            if (!rmsFinalWeightRowInitialized) {
                for (int col = 0; col < rmsFinalWeightRow.length; col++) {
                    rmsFinalWeightRow[col] = data.get(rmsFinalWeightStartPos + col);
                }
                rmsFinalWeightRowInitialized = true;
            }

            return rmsFinalWeightRow;
        }

        float[][] wclsMatrix() {
            if (!wclsMatrixInitialized) {
                for (int i = 0; i < config.vocabSize(); i++) {
                    for (int j = 0; j < config.dim(); j++) {
                        wclsMatrix[i][j] = data.get(wclsStartPos + i * config.dim() + j);
                    }
                }
                wclsMatrixInitialized = true;
            }

            return wclsMatrix;
        }
    }

    static class RunState {
        // current wave of activations
        float[] x; // activation at current time stamp (dim,)
        float[] xb; // same, but inside a residual branch (dim,)
        float[] xb2; // an additional buffer just for convenience (dim,)
        float[] hb; // buffer for hidden dimension in the ffn (hiddenDim,)
        float[] hb2; // buffer for hidden dimension in the ffn (hiddenDim,)
        float[] q; // query (dim,)
        float[] k; // key (dim,)
        float[] v; // value (dim,)
        float[][] attn; // buffer for scores/attention values (nHeads, seqLen)
        float[] logits; // output logits
        // kv cache
        float[][][] keyCache; // (layer, seqLen, dim)
        float[][][] valueCache; // (layer, seqLen, dim)

        public RunState(Config config) {
            // we calloc instead of malloc to keep valgrind happy
            int kvDim = (config.dim() * config.nKVHeads()) / config.nHeads();
            this.x = new float[config.dim()];
            this.xb = new float[config.dim()];
            this.xb2 = new float[config.dim()];
            this.hb = new float[config.hiddenDim];
            this.hb2 = new float[config.hiddenDim];
            this.q = new float[config.dim()];
            this.keyCache = new float[config.nLayers()][config.seqLen()][kvDim];
            this.valueCache = new float[config.nLayers()][config.seqLen()][kvDim];
            this.attn = new float[config.nHeads()][config.seqLen()];
            this.logits = new float[config.vocabSize()];
        }
    }

    final Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    final RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    final FloatBuffer data; // memory mapped data pointer

    final RandomAccessFile file;
    final FileChannel fileChannel;

    public Transformer(String checkpointPath) throws IOException {
        // read in the Config and the Weights from the checkpoint
        long fileSize = Files.size(Path.of(checkpointPath));
        try (RandomAccessFile file = new RandomAccessFile(checkpointPath, "r");
             FileChannel fileChannel = file.getChannel()) {
            // 28 bytes on the assumption that only 7 ints are written as is without padding
            ByteBuffer buffer = ByteBuffer.allocate(28);
            fileChannel.read(buffer);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.flip();
            IntBuffer intBuffer = buffer.asIntBuffer();
            config = new Config(
                    intBuffer.get(),
                    intBuffer.get(),
                    intBuffer.get(),
                    intBuffer.get(),
                    intBuffer.get(),
                    Math.abs(intBuffer.get()),
                    intBuffer.get()
            );
        }
        this.weights = new TransformerWeights();
        boolean sharedWeights = config.vocabSize() > 0;
        file = new RandomAccessFile(checkpointPath, "r");
        fileChannel = file.getChannel();
        MappedByteBuffer weightsBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize)
                .slice(28, (int) (fileSize - 28));
        weightsBuffer.order(ByteOrder.LITTLE_ENDIAN);
        data = weightsBuffer.asFloatBuffer();
        memoryMapWeights(sharedWeights);
        // allocate the RunState buffers
        state = new RunState(config);
    }

    private void memoryMapWeights(boolean sharedWeights) {
        int headSize = config.dim() / config.nHeads();
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        weights.tokenEmbeddingTableStartPos = 0;
        weights.rmsAttnWeightStartPos = weights.tokenEmbeddingTableStartPos
                + (config.vocabSize() * config.dim());

        weights.wqStartPos = weights.rmsAttnWeightStartPos + config.nLayers() * config.dim();
        weights.wkStartPos = weights.wqStartPos + config.nLayers() * config.dim() * config.dim();
        weights.wvStartPos = weights.wkStartPos +
                config.nLayers() * config.dim() * (config.nKVHeads() * headSize);
        weights.woStartPos = weights.wvStartPos +
                config.nLayers() * config.dim() * (config.nKVHeads() * headSize);

        weights.rmsFFNWeightStartPos = weights.woStartPos +
                config.nLayers() * config.dim() * config.dim();

        weights.w1StartPos = weights.rmsFFNWeightStartPos +
                config.nLayers() * config.dim();
        weights.w2StartPos = weights.w1StartPos +
                config.nLayers() * config.hiddenDim() * config.dim();
        weights.w3StartPos = weights.w2StartPos +
                config.nLayers() * config.dim() * config.hiddenDim();
        weights.rmsFinalWeightStartPos = weights.w3StartPos +
                config.nLayers() * config.hiddenDim() * config.dim();

        int wclsStartPos = weights.rmsFinalWeightStartPos + config.dim() +
                config.seqLen() * headSize / 2 + // skip what used to be freq_cis_real (for RoPE)
                config.seqLen() * headSize / 2; // skip what used to be freq_cis_imag (for RoPE)
        weights.wclsStartPos = sharedWeights ? weights.tokenEmbeddingTableStartPos : wclsStartPos;
    }

    public void cleanup() throws IOException {
        fileChannel.close();
        file.close();
    }

    public float[] forward(int token, int pos) {
        // a few convenience variables
        Config p = config;
        TransformerWeights w = weights;
        RunState s = state;
        float[] x = s.x;
        int dim = p.dim();
        int kvDim = (p.dim() * p.nKVHeads()) / p.nHeads();
        int kvMul = p.nHeads() / p.nKVHeads();
        int hiddenDim = p.hiddenDim();
        int headSize = dim / p.nHeads();

        // copy the token embedding into x
        float[] contentRow = w.tokenEmbeddingTableRow(token);
        System.arraycopy(contentRow, 0, x, 0, dim); // TODO: check for x's start position, and length

        // forward all the layers
        for (int l = 0; l < p.nLayers(); l++) {
            // attention NNUtils.rmsnorm
            NNUtils.rmsnorm(s.xb, x, w.rmsAttnWeightRow(l));

            // key and value point to the kv cache
            s.k = s.keyCache[l][pos];
            s.v = s.valueCache[l][pos];

            // qkv NNUtils.matmuls for this position
            NNUtils.matmul(s.q, s.xb, w.wqMatrix(l));
            NNUtils.matmul(s.k, s.xb, w.wkMatrix(l));
            NNUtils.matmul(s.v, s.xb, w.wvMatrix(l));

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int headDim = i % headSize;
                float freq = Double.valueOf(1D / Math.pow(10000f, headDim / (1f * headSize))).floatValue();
                float val = pos * freq;
                float fcr = Double.valueOf(Math.cos(val)).floatValue();
                float fci = Double.valueOf(Math.sin(val)).floatValue();
                int rotations = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotations; v++) {
                    float[] vec = v == 0 ? s.q : s.k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multihead attention. iterate over all heads
            final int layerIndex = l;
            IntStream.range(0, p.nHeads()).parallel()
                    .forEach(h -> attention(s, h, headSize, pos, layerIndex, kvDim, kvMul));

            // final NNUtils.matmul to get the output of the attention
            NNUtils.matmul(s.xb2, s.xb, w.woMatrix(l));

            // residual connection back into x
            for (int i = 0; i < dim; i++) {
                x[i] += s.xb2[i];
            }

            // ffn NNUtils.rmsnorm
            NNUtils.rmsnorm(s.xb, x, w.rmsFFNWeightRow(l));

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            NNUtils.matmul(s.hb, s.xb, w.w1Matrix(l));
            NNUtils.matmul(s.hb2, s.xb, w.w3Matrix(l));

            // SwiGLU non-linearity
            for (int i = 0; i < hiddenDim; i++) {
                float val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1f / (1f + (float) Math.exp(-1 * val)));
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // final NNUtils.matmul to get the output of the ffn
            NNUtils.matmul(s.xb, s.hb, w.w2Matrix(l));

            // residual connection
            for (int i = 0; i < dim; i++) {
                x[i] += s.xb[i];
            }
        }

        // final NNUtils.rmsnorm
        NNUtils.rmsnorm(x, x, w.rmsFinalWeightRow());

        // classifier into logits
        NNUtils.matmul(s.logits, x, w.wclsMatrix());
        return s.logits;
    }

    private void attention(RunState s, int h, int headSize, int pos,
                           int l, int kvDim, int kvMul) {
        // get the query vector for this head
        int qOffset = h * headSize; // in s.q
        float[] attn = s.attn[h];
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            int kOfsset = (h / kvMul) * headSize; // in s.keyCache[l][t*kvDim]
            // calculate the attention score as the dot product of q and k
            float score = 0f;
            for (int i = 0; i < headSize; i++) {
                score += s.q[qOffset + i] * s.keyCache[l][t][kOfsset + i];
            }
            score /= Double.valueOf(Math.sqrt(headSize)).floatValue();
            // save the score to the attention buffer
            attn[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        NNUtils.softmax(attn, pos + 1);

        // weighted sum of the values, store back into xb
        int xbOffset = h * headSize;
        for (int i = 0; i < headSize; i++) {
            s.xb[xbOffset + i] = 0;
        }
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            int vOfsset = (h / kvMul) * headSize; // in s.valueCache[l][t*kvDim]
            // get the attention weight for this timestep
            float a = attn[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < headSize; i++) {
                s.xb[xbOffset + i] += a * s.valueCache[l][t][vOfsset + i];
            }
        }
    }
}
