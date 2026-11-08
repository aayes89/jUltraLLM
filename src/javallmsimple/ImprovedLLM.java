package javallmsimple;

/**
 *
 * @author Slam
 */
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import java.util.concurrent.*;

public class ImprovedLLM implements Serializable {

    private static final long serialVersionUID = 5L;

    // ===================== Vocabulario =====================
    private final Map<String, Integer> wordToIndex = new HashMap<>();
    private final Map<Integer, String> indexToWord = new HashMap<>();
    private int vocabSize = 0;
    private int[] corpusIds;   // corpus ya pre-tokenizado

    // ===================== Arquitectura =====================
    private final int embeddingSize;
    private final int hiddenSize;
    private double[][] We;                                    // [vocab][emb]
    private double[][] Wf, Wi, Wo, Wg;                        // [hidden][emb]
    private double[][] Uf, Ui, Uo, Ug;                        // [hidden][hidden]
    private double[] bf, bi, bo, bg;                        // [hidden]
    private double[][] Why;                                   // [vocab][hidden]
    private double[] by;                                    // [vocab]

    // ===================== Adam (transient: no infla el .ser) ====
    private transient double[][] mWe, mWf, mWi, mWo, mWg, mUf, mUi, mUo, mUg, mWhy;
    private transient double[] mbf, mbi, mbo, mbg, mby;
    private transient double[][] vWe, vWf, vWi, vWo, vWg, vUf, vUi, vUo, vUg, vWhy;
    private transient double[] vbf, vbi, vbo, vbg, vby;

    // ===================== Hiperparámetros =====================
    private double baseLearningRate;
    private double learningRate;
    private final Random rand = new Random(42);

    private static final int MAX_VOCAB_SIZE = 5000;
    private static final int MAX_SEQUENCES = 30000;
    private static final double DROPOUT = 0.20;
    private static final double WEIGHT_DECAY = 0.01;
    private static final double GRAD_CLIP = 5.0;

    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int adamStep = 0;
    private int currentEpoch = 0;
    private double bestValLoss = Double.POSITIVE_INFINITY;

    // ===================== Entrenamiento =====================
    private int optimalBatchSize = 64;
    private int optimalThreads;
    private transient ExecutorService threadPool;

    // ===================== Constructores =====================
    public ImprovedLLM() {
        this(64, 256, 0.003);
    }

    public ImprovedLLM(int embeddingSize, int hiddenSize, double lr) {
        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;
        this.baseLearningRate = lr;
        this.learningRate = lr;
        this.optimalThreads = Math.max(2, Math.min(Runtime.getRuntime().availableProcessors() - 1, 16));
    }

    // ============================================================
    //                   CARGA DE CORPUS (word-level)
    // ============================================================
    public void loadAlpacaCorpus(String filePath, int maxConversations) throws IOException {
        Map<String, Integer> counts = new HashMap<>();
        int conv = 0;

        // Paso 1: frecuencias
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null && conv < maxConversations) {
                String q = extractJsonField(line, "question");
                String a = extractJsonField(line, "answer");
                if (q == null || a == null) {
                    continue;
                }
                String text = "Usuario: " + q + "\nAsistente: " + a;
                for (String tok : tokenize(text)) {
                    counts.merge(tok, 1, Integer::sum);
                }
                conv++;
            }
        }
        System.out.println("Conversaciones leídas: " + conv);

        buildVocab(counts);

        // Paso 2: tokenizar a IDs
        List<Integer> ids = new ArrayList<>();
        conv = 0;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null && conv < maxConversations) {
                String q = extractJsonField(line, "question");
                String a = extractJsonField(line, "answer");
                if (q == null || a == null) {
                    continue;
                }
                String text = "Usuario: " + q + "\nAsistente: " + a;
                ids.add(wordToIndex.get("<START>"));
                for (String tok : tokenize(text)) {
                    ids.add(wordToIndex.getOrDefault(tok, wordToIndex.get("<UNK>")));
                }
                ids.add(wordToIndex.get("<END>"));
                conv++;
            }
        }
        corpusIds = ids.stream().mapToInt(Integer::intValue).toArray();
        System.out.printf("Corpus tokenizado: %d tokens, vocab=%d%n", corpusIds.length, vocabSize);
        initializeWeights();
    }

    public void loadTextCorpus(String filePath) throws IOException {
        Map<String, Integer> counts = new HashMap<>();
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                lines.add(line);
                for (String tok : tokenize(line)) {
                    counts.merge(tok, 1, Integer::sum);
                }
            }
        }
        buildVocab(counts);
        List<Integer> ids = new ArrayList<>();
        for (String line : lines) {
            ids.add(wordToIndex.get("<START>"));
            for (String tok : tokenize(line)) {
                ids.add(wordToIndex.getOrDefault(tok, wordToIndex.get("<UNK>")));
            }
            ids.add(wordToIndex.get("<END>"));
        }
        corpusIds = ids.stream().mapToInt(Integer::intValue).toArray();
        System.out.printf("Corpus tokenizado: %d tokens, vocab=%d%n", corpusIds.length, vocabSize);
        initializeWeights();
    }

    private void buildVocab(Map<String, Integer> counts) {
        wordToIndex.put("<UNK>", vocabSize);
        indexToWord.put(vocabSize++, "<UNK>");
        wordToIndex.put("<START>", vocabSize);
        indexToWord.put(vocabSize++, "<START>");
        wordToIndex.put("<END>", vocabSize);
        indexToWord.put(vocabSize++, "<END>");

        counts.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .limit(MAX_VOCAB_SIZE - vocabSize)
                .forEach(e -> {
                    wordToIndex.put(e.getKey(), vocabSize);
                    indexToWord.put(vocabSize++, e.getKey());
                });
    }

    /**
     * Tokenizador por palabra + puntuación separada. Mucho más rápido que por
     * carácter.
     */
    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isWhitespace(c)) {
                if (cur.length() > 0) {
                    tokens.add(cur.toString());
                    cur.setLength(0);
                }
            } else if (isPunct(c)) {
                if (cur.length() > 0) {
                    tokens.add(cur.toString());
                    cur.setLength(0);
                }
                tokens.add(String.valueOf(c));
            } else if (Character.isLetterOrDigit(c)) {
                cur.append(c);
            }
            // Ignoramos emojis y símbolos raros → reducción de vocab
        }
        if (cur.length() > 0) {
            tokens.add(cur.toString());
        }
        return tokens;
    }

    private boolean isPunct(char c) {
        return ".,;:!?¿¡()[]{}<>\"'-".indexOf(c) >= 0;
    }

    private String detokenize(List<String> tokens) {
        StringBuilder sb = new StringBuilder();
        for (String t : tokens) {
            if ("<START>".equals(t) || "<END>".equals(t) || "<UNK>".equals(t)) {
                continue;
            }
            if (t.length() == 1 && isPunct(t.charAt(0))) {
                sb.append(t);
            } else {
                if (sb.length() > 0) {
                    sb.append(' ');
                }
                sb.append(t);
            }
        }
        return sb.toString();
    }

    private String extractJsonField(String json, String name) {
        String key = "\"" + name + "\": \"";
        int start = json.indexOf(key);
        if (start < 0) {
            return null;
        }
        start += key.length();
        int end = start;
        while (end < json.length()) {
            if (json.charAt(end) == '"' && json.charAt(end - 1) != '\\') {
                break;
            }
            end++;
        }
        return json.substring(start, end)
                .replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    // ============================================================
    //                   INICIALIZACIÓN
    // ============================================================
    private void initializeWeights() {
        double embScale = Math.sqrt(2.0 / embeddingSize);
        double hiddenScale = Math.sqrt(2.0 / hiddenSize);

        We = initMatrix(vocabSize, embeddingSize, embScale);
        Wf = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wi = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wo = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wg = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Uf = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Ui = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Uo = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Ug = initMatrix(hiddenSize, hiddenSize, hiddenScale);

        bf = new double[hiddenSize];
        Arrays.fill(bf, 1.0);
        bi = new double[hiddenSize];
        bo = new double[hiddenSize];
        bg = new double[hiddenSize];

        Why = initMatrix(vocabSize, hiddenSize, Math.sqrt(2.0 / hiddenSize));
        by = new double[vocabSize];

        initAdamStates();
    }

    private void initAdamStates() {
        mWe = new double[vocabSize][embeddingSize];
        vWe = new double[vocabSize][embeddingSize];
        mWhy = new double[vocabSize][hiddenSize];
        vWhy = new double[vocabSize][hiddenSize];
        mby = new double[vocabSize];
        vby = new double[vocabSize];

        mWf = new double[hiddenSize][embeddingSize];
        vWf = new double[hiddenSize][embeddingSize];
        mWi = new double[hiddenSize][embeddingSize];
        vWi = new double[hiddenSize][embeddingSize];
        mWo = new double[hiddenSize][embeddingSize];
        vWo = new double[hiddenSize][embeddingSize];
        mWg = new double[hiddenSize][embeddingSize];
        vWg = new double[hiddenSize][embeddingSize];

        mUf = new double[hiddenSize][hiddenSize];
        vUf = new double[hiddenSize][hiddenSize];
        mUi = new double[hiddenSize][hiddenSize];
        vUi = new double[hiddenSize][hiddenSize];
        mUo = new double[hiddenSize][hiddenSize];
        vUo = new double[hiddenSize][hiddenSize];
        mUg = new double[hiddenSize][hiddenSize];
        vUg = new double[hiddenSize][hiddenSize];

        mbf = new double[hiddenSize];
        vbf = new double[hiddenSize];
        mbi = new double[hiddenSize];
        vbi = new double[hiddenSize];
        mbo = new double[hiddenSize];
        vbo = new double[hiddenSize];
        mbg = new double[hiddenSize];
        vbg = new double[hiddenSize];
    }

    private double[][] initMatrix(int rows, int cols, double scale) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = rand.nextGaussian() * scale;
            }
        }
        return m;
    }

    // ============================================================
    //                   PREPARACIÓN DE DATOS
    // ============================================================
    private List<int[]> prepareTrainingData(int seqLength) {
        List<int[]> seqs = new ArrayList<>();
        int stride = seqLength;  // sin solapamiento
        for (int start = 0; start + seqLength + 1 <= corpusIds.length; start += stride) {
            int[] s = new int[seqLength + 1];
            System.arraycopy(corpusIds, start, s, 0, seqLength + 1);
            seqs.add(s);
            if (seqs.size() >= MAX_SEQUENCES) {
                break;
            }
        }
        return seqs;
    }

    // ============================================================
    //                   ENTRENAMIENTO
    // ============================================================
    public void train(int epochs, int seqLength) {
        System.out.println("=== ENTRENAMIENTO (CPU-only) ===");
        System.out.printf("Vocab=%d | Batch=%d | Threads=%d | Dropout=%.2f | WD=%.3f%n",
                vocabSize, optimalBatchSize, optimalThreads, DROPOUT, WEIGHT_DECAY);

        List<int[]> sequences = prepareTrainingData(seqLength);
        System.out.println("Secuencias totales: " + sequences.size());
        if (sequences.size() < 100) {
            System.out.println("⚠️  Muy pocas secuencias. Aumenta corpus o reduce seqLength.");
        }

        // Split 95/5
        Collections.shuffle(sequences, rand);
        int valSize = Math.max(1, sequences.size() / 20);
        List<int[]> valSeqs = sequences.subList(0, valSize);
        List<int[]> trainSeqs = sequences.subList(valSize, sequences.size());

        int batchesPerEpoch = (trainSeqs.size() + optimalBatchSize - 1) / optimalBatchSize;
        int totalSteps = epochs * batchesPerEpoch;
        int warmupSteps = Math.max(50, totalSteps / 20);
        int globalStep = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long t0 = System.currentTimeMillis();
            Collections.shuffle(trainSeqs, rand);
            double totalLoss = 0;
            int batches = 0;

            for (int start = 0; start < trainSeqs.size(); start += optimalBatchSize) {
                int end = Math.min(start + optimalBatchSize, trainSeqs.size());
                List<int[]> batch = trainSeqs.subList(start, end);

                learningRate = getLR(globalStep, totalSteps, baseLearningRate, warmupSteps);
                double loss = processBatch(batch);
                totalLoss += loss;
                batches++;
                globalStep++;

                if (batches % 20 == 0 || batches == batchesPerEpoch) {
                    System.out.printf("\r  batch %d/%d | loss=%.4f | lr=%.6f",
                            batches, batchesPerEpoch, totalLoss / batches, learningRate);
                }
            }
            System.out.println();

            double trainLoss = totalLoss / Math.max(1, batches);
            double valLoss = evaluate(valSeqs);
            long dt = System.currentTimeMillis() - t0;

            System.out.printf("✅ Epoch %d/%d | train=%.4f | val=%.4f | ppl=%.2f | %.1fs%n",
                    epoch + 1, epochs, trainLoss, valLoss, Math.exp(valLoss), dt / 1000.0);

            currentEpoch = epoch + 1;
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                saveModel("best_model.ser");
                System.out.println("   💾 Nuevo mejor modelo guardado (val=%.4f)".formatted(valLoss));
            }
        }
        System.out.println("\n🎉 Entrenamiento completado. Mejor val_loss = " + bestValLoss);
    }

    private double getLR(int step, int total, double base, int warmup) {
        if (step < warmup) {
            return base * (step + 1.0) / warmup;
        }
        double p = (double) (step - warmup) / Math.max(1, total - warmup);
        return base * 0.5 * (1.0 + Math.cos(Math.PI * p));
    }

    private double processBatch(List<int[]> batch) {
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }
        int chunkSize = Math.max(1, batch.size() / optimalThreads);
        List<Future<Gradients>> futures = new ArrayList<>();

        for (int i = 0; i < batch.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, batch.size());
            List<int[]> chunk = batch.subList(i, end);
            futures.add(threadPool.submit(() -> {
                Gradients g = new Gradients();
                for (int[] seq : chunk) {
                    g.loss += trainSequence(seq, g);
                }
                return g;
            }));
        }

        Gradients total = new Gradients();
        for (Future<Gradients> f : futures) {
            try {
                total.add(f.get());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        total.scale(1.0 / batch.size());
        applyGradientClip(total);
        adamStep++;
        applyAdamW(total);
        return total.loss / batch.size();
    }

    // ============================================================
    //                   FORWARD + BACKWARD LSTM (CPU, in-place)
    // ============================================================
    private double trainSequence(int[] seq, Gradients grad) {
        int T = seq.length - 1;
        double[][] h = new double[T][hiddenSize];
        double[][] c = new double[T][hiddenSize];
        double[][] fArr = new double[T][hiddenSize];
        double[][] iArr = new double[T][hiddenSize];
        double[][] oArr = new double[T][hiddenSize];
        double[][] gArr = new double[T][hiddenSize];
        double[][] tanhC = new double[T][hiddenSize];
        double[][] xArr = new double[T][embeddingSize];
        double[][] xScale = new double[T][embeddingSize];
        double[][] hScale = new double[T][hiddenSize];

        double[] prevH = new double[hiddenSize];
        double[] prevC = new double[hiddenSize];
        double loss = 0;

        // ---------- FORWARD ----------
        for (int t = 0; t < T; t++) {
            // Embedding + dropout
            double[] emb = We[seq[t]];
            for (int k = 0; k < embeddingSize; k++) {
                xScale[t][k] = (rand.nextDouble() >= DROPOUT) ? 1.0 / (1.0 - DROPOUT) : 0.0;
                xArr[t][k] = emb[k] * xScale[t][k];
            }

            matVecMul(fArr[t], Wf, xArr[t]);
            matVecMulAdd(fArr[t], Uf, prevH);
            addInPlace(fArr[t], bf);
            sigmoidInPlace(fArr[t]);

            matVecMul(iArr[t], Wi, xArr[t]);
            matVecMulAdd(iArr[t], Ui, prevH);
            addInPlace(iArr[t], bi);
            sigmoidInPlace(iArr[t]);

            matVecMul(oArr[t], Wo, xArr[t]);
            matVecMulAdd(oArr[t], Uo, prevH);
            addInPlace(oArr[t], bo);
            sigmoidInPlace(oArr[t]);

            matVecMul(gArr[t], Wg, xArr[t]);
            matVecMulAdd(gArr[t], Ug, prevH);
            addInPlace(gArr[t], bg);
            tanhInPlace(gArr[t]);

            for (int j = 0; j < hiddenSize; j++) {
                c[t][j] = fArr[t][j] * prevC[j] + iArr[t][j] * gArr[t][j];
                tanhC[t][j] = Math.tanh(c[t][j]);
                h[t][j] = oArr[t][j] * tanhC[t][j];
            }

            // Proyección a logits con dropout en h
            double[] hDrop = new double[hiddenSize];
            for (int k = 0; k < hiddenSize; k++) {
                hScale[t][k] = (rand.nextDouble() >= DROPOUT) ? 1.0 / (1.0 - DROPOUT) : 0.0;
                hDrop[k] = h[t][k] * hScale[t][k];
            }
            double[] logits = new double[vocabSize];
            matVecMul(logits, Why, hDrop);
            addInPlace(logits, by);
            softmaxInPlace(logits);

            int target = seq[t + 1];
            loss += -Math.log(Math.max(logits[target], 1e-10));

            prevH = h[t];
            prevC = c[t];
        }

        // ---------- BACKWARD ----------
        double[] dhNext = new double[hiddenSize];
        double[] dcNext = new double[hiddenSize];
        double[] dh = new double[hiddenSize];
        double[] dc = new double[hiddenSize];
        double[] df = new double[hiddenSize];
        double[] di = new double[hiddenSize];
        double[] dob = new double[hiddenSize];
        double[] dg = new double[hiddenSize];
        double[] dx = new double[embeddingSize];
        double[] dOutput = new double[vocabSize];
        double[] logitsCache = new double[vocabSize];
        double[] hDropCache = new double[hiddenSize];

        for (int t = T - 1; t >= 0; t--) {
            int target = seq[t + 1];
            int tokIdx = seq[t];

            // Recomputar logits (softmax) con hDrop
            for (int k = 0; k < hiddenSize; k++) {
                hDropCache[k] = h[t][k] * hScale[t][k];
            }
            matVecMul(logitsCache, Why, hDropCache);
            addInPlace(logitsCache, by);
            softmaxInPlace(logitsCache);

            Arrays.fill(dOutput, 0);
            dOutput[target] = -1.0;
            for (int k = 0; k < vocabSize; k++) {
                dOutput[k] += logitsCache[k];
            }

            for (int k = 0; k < vocabSize; k++) {
                double dk = dOutput[k];
                for (int j = 0; j < hiddenSize; j++) {
                    grad.dWhy[k][j] += dk * hDropCache[j];
                }
                grad.dby[k] += dk;
            }

            Arrays.fill(dh, 0);
            for (int k = 0; k < vocabSize; k++) {
                double dk = dOutput[k];
                for (int j = 0; j < hiddenSize; j++) {
                    dh[j] += dk * Why[k][j] * hScale[t][j];
                }
            }
            addInPlace(dh, dhNext);

            for (int j = 0; j < hiddenSize; j++) {
                double tc = tanhC[t][j];
                dc[j] = dh[j] * oArr[t][j] * (1.0 - tc * tc) + dcNext[j];
                df[j] = dc[j] * (t > 0 ? c[t - 1][j] : 0.0) * fArr[t][j] * (1.0 - fArr[t][j]);
                di[j] = dc[j] * gArr[t][j] * iArr[t][j] * (1.0 - iArr[t][j]);
                dob[j] = dh[j] * tc * oArr[t][j] * (1.0 - oArr[t][j]);
                dg[j] = dc[j] * iArr[t][j] * (1.0 - gArr[t][j] * gArr[t][j]);
            }

            Arrays.fill(dx, 0);
            for (int j = 0; j < hiddenSize; j++) {
                double dfj = df[j], dij = di[j], doj = dob[j], dgj = dg[j];
                for (int k = 0; k < embeddingSize; k++) {
                    grad.dWf[j][k] += dfj * xArr[t][k];
                    grad.dWi[j][k] += dij * xArr[t][k];
                    grad.dWo[j][k] += doj * xArr[t][k];
                    grad.dWg[j][k] += dgj * xArr[t][k];
                    dx[k] += dfj * Wf[j][k] + dij * Wi[j][k] + doj * Wo[j][k] + dgj * Wg[j][k];
                }
                grad.dbf[j] += dfj;
                grad.dbi[j] += dij;
                grad.dbo[j] += doj;
                grad.dbg[j] += dgj;
            }

            for (int j = 0; j < hiddenSize; j++) {
                double dfj = df[j], dij = di[j], doj = dob[j], dgj = dg[j];
                for (int k = 0; k < hiddenSize; k++) {
                    double prevHk = (t > 0) ? h[t - 1][k] : 0.0;
                    grad.dUf[j][k] += dfj * prevHk;
                    grad.dUi[j][k] += dij * prevHk;
                    grad.dUo[j][k] += doj * prevHk;
                    grad.dUg[j][k] += dgj * prevHk;
                }
            }

            // Gradiente al embedding (aplicando dropout scale)
            for (int k = 0; k < embeddingSize; k++) {
                grad.dWe[tokIdx][k] += dx[k] * xScale[t][k];
            }

            Arrays.fill(dhNext, 0);
            for (int j = 0; j < hiddenSize; j++) {
                double dfj = df[j], dij = di[j], doj = dob[j], dgj = dg[j];
                for (int k = 0; k < hiddenSize; k++) {
                    dhNext[k] += dfj * Uf[j][k] + dij * Ui[j][k] + doj * Uo[j][k] + dgj * Ug[j][k];
                }
            }
            for (int j = 0; j < hiddenSize; j++) {
                dcNext[j] = dc[j] * fArr[t][j];
            }
        }
        return loss / T;
    }

    private double evaluate(List<int[]> seqs) {
        // Evaluación sin dropout: guardamos DROPOUT no se aplica, pero como es final no podemos
        // mutarla. Hacemos una copia ligera: forward con dropout desactivado vía flag.
        // Simplificación: reutilizamos trainSequence pero forzando rand de dropout a 1.0.
        // Cambio rápido: hacer forward simple sin backward (más barato).
        double total = 0;
        double[] h = new double[hiddenSize], c = new double[hiddenSize];
        double[] f = new double[hiddenSize], i = new double[hiddenSize];
        double[] o = new double[hiddenSize], g = new double[hiddenSize];
        double[] tc = new double[hiddenSize];
        for (int[] seq : seqs) {
            Arrays.fill(h, 0);
            Arrays.fill(c, 0);
            double loss = 0;
            int T = seq.length - 1;
            for (int t = 0; t < T; t++) {
                double[] x = We[seq[t]];
                matVecMul(f, Wf, x);
                matVecMulAdd(f, Uf, h);
                addInPlace(f, bf);
                sigmoidInPlace(f);
                matVecMul(i, Wi, x);
                matVecMulAdd(i, Ui, h);
                addInPlace(i, bi);
                sigmoidInPlace(i);
                matVecMul(o, Wo, x);
                matVecMulAdd(o, Uo, h);
                addInPlace(o, bo);
                sigmoidInPlace(o);
                matVecMul(g, Wg, x);
                matVecMulAdd(g, Ug, h);
                addInPlace(g, bg);
                tanhInPlace(g);
                for (int j = 0; j < hiddenSize; j++) {
                    c[j] = f[j] * c[j] + i[j] * g[j];
                    tc[j] = Math.tanh(c[j]);
                    h[j] = o[j] * tc[j];
                }
                double[] logits = new double[vocabSize];
                matVecMul(logits, Why, h);
                addInPlace(logits, by);
                softmaxInPlace(logits);
                loss += -Math.log(Math.max(logits[seq[t + 1]], 1e-10));
            }
            total += loss / T;
        }
        return total / seqs.size();
    }

    // ============================================================
    //                   GRADIENTES + ADAMW
    // ============================================================
    class Gradients {

        double loss = 0;
        double[][] dWf = new double[hiddenSize][embeddingSize];
        double[][] dWi = new double[hiddenSize][embeddingSize];
        double[][] dWo = new double[hiddenSize][embeddingSize];
        double[][] dWg = new double[hiddenSize][embeddingSize];
        double[][] dUf = new double[hiddenSize][hiddenSize];
        double[][] dUi = new double[hiddenSize][hiddenSize];
        double[][] dUo = new double[hiddenSize][hiddenSize];
        double[][] dUg = new double[hiddenSize][hiddenSize];
        double[] dbf = new double[hiddenSize];
        double[] dbi = new double[hiddenSize];
        double[] dbo = new double[hiddenSize];
        double[] dbg = new double[hiddenSize];
        double[][] dWe = new double[vocabSize][embeddingSize];
        double[][] dWhy = new double[vocabSize][hiddenSize];
        double[] dby = new double[vocabSize];

        void add(Gradients o) {
            loss += o.loss;
            addArr(dWf, o.dWf);
            addArr(dWi, o.dWi);
            addArr(dWo, o.dWo);
            addArr(dWg, o.dWg);
            addArr(dUf, o.dUf);
            addArr(dUi, o.dUi);
            addArr(dUo, o.dUo);
            addArr(dUg, o.dUg);
            addVec(dbf, o.dbf);
            addVec(dbi, o.dbi);
            addVec(dbo, o.dbo);
            addVec(dbg, o.dbg);
            addArr(dWe, o.dWe);
            addArr(dWhy, o.dWhy);
            addVec(dby, o.dby);
        }

        void scale(double s) {
            scaleArr(dWf, s);
            scaleArr(dWi, s);
            scaleArr(dWo, s);
            scaleArr(dWg, s);
            scaleArr(dUf, s);
            scaleArr(dUi, s);
            scaleArr(dUo, s);
            scaleArr(dUg, s);
            scaleVec(dbf, s);
            scaleVec(dbi, s);
            scaleVec(dbo, s);
            scaleVec(dbg, s);
            scaleArr(dWe, s);
            scaleArr(dWhy, s);
            scaleVec(dby, s);
        }

        private void addArr(double[][] a, double[][] b) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[i].length; j++) {
                    a[i][j] += b[i][j];
                }
            }
        }

        private void addVec(double[] a, double[] b) {
            for (int i = 0; i < a.length; i++) {
                a[i] += b[i];
            }
        }

        private void scaleArr(double[][] a, double s) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[i].length; j++) {
                    a[i][j] *= s;
                }
            }
        }

        private void scaleVec(double[] a, double s) {
            for (int i = 0; i < a.length; i++) {
                a[i] *= s;
            }
        }
    }

    private void applyGradientClip(Gradients g) {
        double n = 0;
        n += sumSq(g.dWf);
        n += sumSq(g.dWi);
        n += sumSq(g.dWo);
        n += sumSq(g.dWg);
        n += sumSq(g.dUf);
        n += sumSq(g.dUi);
        n += sumSq(g.dUo);
        n += sumSq(g.dUg);
        n += sumSq(g.dWe);
        n += sumSq(g.dWhy);
        double norm = Math.sqrt(n);
        if (norm > GRAD_CLIP) {
            g.scale(GRAD_CLIP / norm);
        }
    }

    private double sumSq(double[][] a) {
        double s = 0;
        for (double[] r : a) {
            for (double v : r) {
                s += v * v;
            }
        }
        return s;
    }

    private void applyAdamW(Gradients g) {
        adamUpdate(Wf, g.dWf, mWf, vWf);
        adamUpdate(Wi, g.dWi, mWi, vWi);
        adamUpdate(Wo, g.dWo, mWo, vWo);
        adamUpdate(Wg, g.dWg, mWg, vWg);
        adamUpdate(Uf, g.dUf, mUf, vUf);
        adamUpdate(Ui, g.dUi, mUi, vUi);
        adamUpdate(Uo, g.dUo, mUo, vUo);
        adamUpdate(Ug, g.dUg, mUg, vUg);
        adamUpdate(bf, g.dbf, mbf, vbf);
        adamUpdate(bi, g.dbi, mbi, vbi);
        adamUpdate(bo, g.dbo, mbo, vbo);
        adamUpdate(bg, g.dbg, mbg, vbg);
        adamUpdate(We, g.dWe, mWe, vWe);
        adamUpdate(Why, g.dWhy, mWhy, vWhy);
        adamUpdate(by, g.dby, mby, vby);
    }

    private void adamUpdate(double[][] W, double[][] dW, double[][] m, double[][] v) {
        double c1 = 1.0 - Math.pow(beta1, adamStep);
        double c2 = 1.0 - Math.pow(beta2, adamStep);
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                double g = dW[i][j];
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;
                double adam = (m[i][j] / c1) / (Math.sqrt(v[i][j] / c2) + epsilon);
                W[i][j] -= learningRate * (adam + WEIGHT_DECAY * W[i][j]);
            }
        }
    }

    private void adamUpdate(double[] b, double[] db, double[] m, double[] v) {
        double c1 = 1.0 - Math.pow(beta1, adamStep);
        double c2 = 1.0 - Math.pow(beta2, adamStep);
        for (int i = 0; i < b.length; i++) {
            double g = db[i];
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;
            double adam = (m[i] / c1) / (Math.sqrt(v[i] / c2) + epsilon);
            b[i] -= learningRate * (adam + WEIGHT_DECAY * b[i]);
        }
    }

    // ============================================================
    //                   INFERENCIA
    // ============================================================
    public String generate(String seed, int maxLength, double temperature) {
        List<String> seedTokens = tokenize(seed);
        double[] h = new double[hiddenSize], c = new double[hiddenSize];
        double[] f = new double[hiddenSize], i = new double[hiddenSize];
        double[] o = new double[hiddenSize], g = new double[hiddenSize], tc = new double[hiddenSize];

        // Estado inicial con <START>
        int startId = wordToIndex.getOrDefault("<START>", 0);
        forwardStep(startId, h, c, f, i, o, g, tc);

        List<String> out = new ArrayList<>();
        for (String tok : seedTokens) {
            int id = wordToIndex.getOrDefault(tok, wordToIndex.get("<UNK>"));
            forwardStep(id, h, c, f, i, o, g, tc);
            out.add(tok);
        }

        for (int step = 0; step < maxLength; step++) {
            double[] logits = new double[vocabSize];
            matVecMul(logits, Why, h);
            addInPlace(logits, by);
            double temp = Math.max(1e-6, temperature);
            for (int j = 0; j < vocabSize; j++) {
                logits[j] /= temp;
            }
            softmaxInPlace(logits);

            int next = sampleFromDistribution(logits);
            String token = indexToWord.get(next);
            if ("<END>".equals(token)) {
                break;
            }
            if ("<START>".equals(token) || "<UNK>".equals(token)) {
                forwardStep(next, h, c, f, i, o, g, tc);
                continue;
            }
            out.add(token);
            forwardStep(next, h, c, f, i, o, g, tc);
        }
        return detokenize(out);
    }

    private void forwardStep(int id, double[] h, double[] c,
            double[] f, double[] i, double[] o, double[] g, double[] tc) {
        double[] x = We[id];
        matVecMul(f, Wf, x);
        matVecMulAdd(f, Uf, h);
        addInPlace(f, bf);
        sigmoidInPlace(f);
        matVecMul(i, Wi, x);
        matVecMulAdd(i, Ui, h);
        addInPlace(i, bi);
        sigmoidInPlace(i);
        matVecMul(o, Wo, x);
        matVecMulAdd(o, Uo, h);
        addInPlace(o, bo);
        sigmoidInPlace(o);
        matVecMul(g, Wg, x);
        matVecMulAdd(g, Ug, h);
        addInPlace(g, bg);
        tanhInPlace(g);
        for (int j = 0; j < hiddenSize; j++) {
            c[j] = f[j] * c[j] + i[j] * g[j];
            tc[j] = Math.tanh(c[j]);
            h[j] = o[j] * tc[j];
        }
    }

    private int sampleFromDistribution(double[] p) {
        double r = rand.nextDouble(), sum = 0;
        for (int j = 0; j < p.length; j++) {
            sum += p[j];
            if (r <= sum) {
                return j;
            }
        }
        return p.length - 1;
    }

    // ============================================================
    //                   UTILIDADES MATEMÁTICAS
    // ============================================================
    private void matVecMul(double[] y, double[][] M, double[] x) {
        for (int i = 0; i < M.length; i++) {
            double s = 0;
            double[] row = M[i];
            for (int j = 0; j < x.length; j++) {
                s += row[j] * x[j];
            }
            y[i] = s;
        }
    }

    private void matVecMulAdd(double[] y, double[][] M, double[] x) {
        for (int i = 0; i < M.length; i++) {
            double s = 0;
            double[] row = M[i];
            for (int j = 0; j < x.length; j++) {
                s += row[j] * x[j];
            }
            y[i] += s;
        }
    }

    private void addInPlace(double[] y, double[] x) {
        for (int k = 0; k < y.length; k++) {
            y[k] += x[k];
        }
    }

    private void sigmoidInPlace(double[] x) {
        for (int k = 0; k < x.length; k++) {
            x[k] = 1.0 / (1.0 + Math.exp(-x[k]));
        }
    }

    private void tanhInPlace(double[] x) {
        for (int k = 0; k < x.length; k++) {
            x[k] = Math.tanh(x[k]);
        }
    }

    private void softmaxInPlace(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (v > max) {
                max = v;
            }
        }
        double sum = 0;
        for (int k = 0; k < x.length; k++) {
            x[k] = Math.exp(x[k] - max);
            sum += x[k];
        }
        for (int k = 0; k < x.length; k++) {
            x[k] /= sum;
        }
    }

    // ============================================================
    //                   SERIALIZACIÓN (rápida)
    // ============================================================
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        initAdamStates();   // estados de Adam no se guardan → se reinician
        threadPool = Executors.newFixedThreadPool(optimalThreads);
    }

    public void saveModel(String path) {
        cleanup();
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(this);
            File f = new File(path);
            System.out.printf("💾 Modelo guardado: %s (%.1f MB)%n", path, f.length() / 1024.0 / 1024.0);
        } catch (IOException e) {
            System.err.println("Error al guardar: " + e.getMessage());
        }
    }

    public static ImprovedLLM loadModel(String path) {
        long t0 = System.currentTimeMillis();
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            ImprovedLLM m = (ImprovedLLM) in.readObject();
            System.out.printf("✅ Modelo cargado en %.2fs%n", (System.currentTimeMillis() - t0) / 1000.0);
            return m;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error al cargar: " + e.getMessage());
            return null;
        }
    }

    private void cleanup() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
            try {
                threadPool.awaitTermination(3, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {
                threadPool.shutdownNow();
            }
        }
    }

    // ============================================================
    //                   MENÚ INTERACTIVO
    // ============================================================
    public void startInteractiveMode() throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.println("=== LSTM optimizado ===");
        System.out.println("Comandos: LOAD_ALPACA, LOAD_TEXT, TRAIN, SAVE, LOAD, STATUS, EXIT");
        while (true) {
            System.out.print("\n> ");
            String input = sc.nextLine().trim();
            if (input.equalsIgnoreCase("EXIT")) {
                cleanup();
                break;
            } else if (input.equalsIgnoreCase("STATUS")) {
                System.out.printf("Vocab=%d | Corpus=%d | Batch=%d | Threads=%d | Epochs=%d%n",
                        vocabSize, corpusIds == null ? 0 : corpusIds.length,
                        optimalBatchSize, optimalThreads, currentEpoch);
                System.out.printf("Best val_loss=%.4f%n", bestValLoss);
            } else if (input.equalsIgnoreCase("LOAD_ALPACA")) {
                System.out.print("Archivo .jsonl: ");
                String p = sc.nextLine().trim();
                System.out.print("Máx conversaciones (recomendado 2000): ");
                int max = sc.nextLine().trim().isEmpty() ? 2000 : Integer.parseInt(sc.nextLine().trim().isEmpty() ? "2000" : "2000");
                try {
                    max = 2000;
                    System.out.print("(usando 2000 por defecto) ");
                    loadAlpacaCorpus(p, max);
                } catch (Exception e) {
                    System.out.println("Error: " + e.getMessage());
                }
            } else if (input.equalsIgnoreCase("LOAD_TEXT")) {
                System.out.print("Archivo .txt: ");
                String p = sc.nextLine().trim();
                try {
                    loadTextCorpus(p);
                } catch (IOException e) {
                    System.out.println("Error: " + e.getMessage());
                }
            } else if (input.equalsIgnoreCase("TRAIN")) {
                if (vocabSize == 0) {
                    System.out.println("Carga corpus primero.");
                    continue;
                }
                System.out.print("Épocas [5]: ");
                String s = sc.nextLine().trim();
                int ep = s.isEmpty() ? 5 : Integer.parseInt(s);
                System.out.print("Seq length [128]: ");
                s = sc.nextLine().trim();
                int sl = s.isEmpty() ? 128 : Integer.parseInt(s);
                train(ep, sl);
            } else if (input.equalsIgnoreCase("SAVE")) {
                System.out.print("Archivo [model.ser]: ");
                String p = sc.nextLine().trim();
                saveModel(p.isEmpty() ? "model.ser" : p);
            } else if (input.equalsIgnoreCase("LOAD")) {
                System.out.print("Archivo: ");
                String p = sc.nextLine().trim();
                ImprovedLLM m = loadModel(p);
                if (m != null) {
                    m.startInteractiveMode();
                    return;
                }
            } else if (!input.isEmpty()) {
                if (vocabSize == 0) {
                    System.out.println("Modelo no entrenado.");
                    continue;
                }
                System.out.print("Longitud [100]: ");
                String s = sc.nextLine().trim();
                int len = s.isEmpty() ? 100 : Integer.parseInt(s);
                System.out.print("Temperatura [0.7]: ");
                s = sc.nextLine().trim();
                double t = s.isEmpty() ? 0.7 : Double.parseDouble(s);
                System.out.println("🤖 " + generate(input, len, t));
            }
        }
        sc.close();
    }

    public static void main(String[] args) {
        ImprovedLLM m = new ImprovedLLM();
        Runtime.getRuntime().addShutdownHook(new Thread(m::cleanup));
        try {
            m.startInteractiveMode();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
