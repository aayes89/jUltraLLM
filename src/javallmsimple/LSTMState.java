package javallmsimple;

import java.io.Serializable;

/**
 *
 * @author Slam
 */
public class LSTMState implements Serializable {

    private static final long serialVersionUID = 2L;

    final double[] h, c;
    // Compuertas y entrada cacheadas para el backward
    final double[] f, i, o, g;
    final double[] input;
    final double[] tanhC;  // tanh(c) cacheado, para evitar recalcular en backward

    public LSTMState(double[] h, double[] c) {
        this(h, c, null, null, null, null, null, null);
    }

    public LSTMState(double[] h, double[] c, double[] f, double[] i, double[] o, double[] g, double[] input, double[] tanhC) {
        this.h = h;
        this.c = c;
        this.f = f;
        this.i = i;
        this.o = o;
        this.g = g;
        this.input = input;
        this.tanhC = tanhC;
    }
}
