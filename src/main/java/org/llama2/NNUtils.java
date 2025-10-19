package org.llama2;

import java.util.stream.IntStream;

/**
 * Neural net blocks; the dynamics of the Transformer
 */
public class NNUtils {
    public static void rmsnorm(float[] o, float[] x, float[] weight) {
        float ss = 0f;
        for (float v : x) {
            ss += v * v;
        }
        ss /= x.length;
        ss += 1E-5f;
        ss = 1f / Double.valueOf(Math.sqrt(ss)).floatValue();
        for (int i = 0; i < x.length; i++) {
            o[i] = weight[i] * (ss * x[i]);
        }
    }

    public static void softmax(float[] x, int size) {
        // find max value (for numerical stability)
        float maxVal = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > maxVal) maxVal = x[i];
        }
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            x[i] = Double.valueOf(Math.exp(x[i] - maxVal)).floatValue();
            sum += x[i];
        }
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    public static void matmul(float[] xout, float[] x, float[][] w) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        var temp = IntStream.range(0, w.length).parallel().mapToObj(i -> {
            float val = 0f;
            for (int j = 0; j < x.length; j++) {
                val += w[i][j] * x[j];
            }
            return val;
        }).toList();
        for (int i = 0; i < xout.length; i++) {
            xout[i] = temp.get(i);
        }
    }
}
