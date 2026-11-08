package javallmsimple;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

public class UltraLLM implements Serializable {

    private static final long serialVersionUID = 7L;

    // ===================== Vocabulario =====================
    private final Map<String, Integer> wordToIndex = new HashMap<>();
    private final Map<Integer, String> indexToWord = new HashMap<>();
    private int vocabSize = 0;
    private int[] corpusIds;

    // ===================== Arquitectura (1D Flat Arrays - floats) =====================
    private final int embeddingSize;
    private final int hiddenSize;

    private float[] We; // [vocabSize * embeddingSize]
    private float[] Wf, Wi, Wo, Wg; // [hiddenSize * embeddingSize]
    private float[] Uf, Ui, Uo, Ug; // [hiddenSize * hiddenSize]
    private float[] bf, bi, bo, bg; // [hiddenSize]
    private float[] Why; // [vocabSize * hiddenSize]
    private float[] by;  // [vocabSize]

    // ===================== AdamW (transient: no infla el .ser) =====================
    private transient float[] mWe, mWf, mWi, mWo, mWg, mUf, mUi, mUo, mUg, mWhy;
    private transient float[] mbf, mbi, mbo, mbg, mby;
    private transient float[] vWe, vWf, vWi, vWo, vWg, vUf, vUi, vUo, vUg, vWhy;
    private transient float[] vbf, vbi, vbo, vbg, vby;

    // ===================== Hiperparámetros =====================
    private float baseLearningRate;
    private transient float learningRate;
    private final Random rand = new Random(42);

    private static final int MAX_VOCAB_SIZE = 50000;
    private static final int MAX_SEQUENCES = 30000;
    private static final float DROPOUT = 0.15f;
    private static final float WEIGHT_DECAY = 0.01f;
    private static final float GRAD_CLIP = 5.0f;

    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float epsilon = 1e-8f;
    private int adamStep = 0;
    private int currentEpoch = 0;
    private float bestValLoss = Float.POSITIVE_INFINITY;

    // ===================== Entrenamiento y Sistema =====================
    private int optimalBatchSize = 1024;
    private int optimalThreads;
    private transient ExecutorService threadPool;

    // ===================== FastMath LUT =====================
    private static final int LUT_SIZE = 16384;
    private static final float MIN_X = -8.0f;
    private static final float MAX_X = 8.0f;
    private static final float SCALE = LUT_SIZE / (MAX_X - MIN_X);
    private static final float[] EXP_LUT = new float[LUT_SIZE];
    private static final float[] TANH_LUT = new float[LUT_SIZE];

    static {
        for (int i = 0; i < LUT_SIZE; i++) {
            float x = MIN_X + i / SCALE;
            EXP_LUT[i] = (float) Math.exp(x);
            TANH_LUT[i] = (float) Math.tanh(x);
        }
    }

    private static float fastExp(float x) {
        if (x <= MIN_X) {
            return 0.0f; // <-- FIX: Retornar 0.0f evita destruir el Softmax con ruido de fondo
        }
        if (x >= MAX_X) {
            return EXP_LUT[LUT_SIZE - 1];
        }
        int idx = (int) ((x - MIN_X) * SCALE);
        return EXP_LUT[Math.min(idx, LUT_SIZE - 1)]; // <-- FIX: Previene OutOfBounds
    }

    private static float fastTanh(float x) {
        if (x <= MIN_X) {
            return -1.0f; // <-- FIX: Límites asintóticos reales
        }
        if (x >= MAX_X) {
            return 1.0f;
        }
        int idx = (int) ((x - MIN_X) * SCALE);
        return TANH_LUT[Math.min(idx, LUT_SIZE - 1)]; // <-- FIX: Previene OutOfBounds
    }

    // ===================== Constructores =====================
    public UltraLLM() {
        this(1024, 512, 0.005f);
    }

    public UltraLLM(int embeddingSize, int hiddenSize, float lr) {
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

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
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

        List<Integer> ids = new ArrayList<>();
        conv = 0;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
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
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
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
        if (!wordToIndex.containsKey("<UNK>")) {
            wordToIndex.put("<UNK>", vocabSize);
            indexToWord.put(vocabSize++, "<UNK>");
            wordToIndex.put("<START>", vocabSize);
            indexToWord.put(vocabSize++, "<START>");
            wordToIndex.put("<END>", vocabSize);
            indexToWord.put(vocabSize++, "<END>");
        }
        counts.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .filter(e -> !wordToIndex.containsKey(e.getKey()))
                .limit(MAX_VOCAB_SIZE - vocabSize)
                .forEach(e -> {
                    wordToIndex.put(e.getKey(), vocabSize);
                    indexToWord.put(vocabSize++, e.getKey());
                });
    }

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
            } else if (".,;:!?¿¡()[]{}<>\"'-".indexOf(c) >= 0) {
                if (cur.length() > 0) {
                    tokens.add(cur.toString());
                    cur.setLength(0);
                }
                tokens.add(String.valueOf(c));
            } else if (Character.isLetterOrDigit(c)) {
                cur.append(c);
            }
        }
        if (cur.length() > 0) {
            tokens.add(cur.toString());
        }
        return tokens;
    }

    private String detokenize(List<String> tokens) {
        StringBuilder sb = new StringBuilder();
        for (String t : tokens) {
            if ("<START>".equals(t) || "<END>".equals(t) || "<UNK>".equals(t)) {
                continue;
            }
            if (t.length() == 1 && ".,;:!?¿¡()[]{}<>\"'-".indexOf(t.charAt(0)) >= 0) {
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
        return json.substring(start, end).replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    // ============================================================
    //                   INICIALIZACIÓN 1D
    // ============================================================
    private void initializeWeights() {
        if (We != null) {
            return;
        }
        float embScale = (float) Math.sqrt(2.0 / embeddingSize);
        float hiddenScale = (float) Math.sqrt(2.0 / hiddenSize);

        We = initArray(vocabSize * embeddingSize, embScale);
        Wf = initArray(hiddenSize * embeddingSize, hiddenScale);
        Wi = initArray(hiddenSize * embeddingSize, hiddenScale);
        Wo = initArray(hiddenSize * embeddingSize, hiddenScale);
        Wg = initArray(hiddenSize * embeddingSize, hiddenScale);

        Uf = initArray(hiddenSize * hiddenSize, hiddenScale);
        Ui = initArray(hiddenSize * hiddenSize, hiddenScale);
        Uo = initArray(hiddenSize * hiddenSize, hiddenScale);
        Ug = initArray(hiddenSize * hiddenSize, hiddenScale);

        bf = new float[hiddenSize];
        Arrays.fill(bf, 1.0f); // Forget bias = 1
        bi = new float[hiddenSize];
        bo = new float[hiddenSize];
        bg = new float[hiddenSize];

        Why = initArray(vocabSize * hiddenSize, (float) Math.sqrt(2.0 / hiddenSize));
        by = new float[vocabSize];

        initAdamStates();
    }

    private void initAdamStates() {
        mWe = new float[vocabSize * embeddingSize];
        vWe = new float[vocabSize * embeddingSize];
        mWf = new float[hiddenSize * embeddingSize];
        vWf = new float[hiddenSize * embeddingSize];
        mWi = new float[hiddenSize * embeddingSize];
        vWi = new float[hiddenSize * embeddingSize];
        mWo = new float[hiddenSize * embeddingSize];
        vWo = new float[hiddenSize * embeddingSize];
        mWg = new float[hiddenSize * embeddingSize];
        vWg = new float[hiddenSize * embeddingSize];
        mUf = new float[hiddenSize * hiddenSize];
        vUf = new float[hiddenSize * hiddenSize];
        mUi = new float[hiddenSize * hiddenSize];
        vUi = new float[hiddenSize * hiddenSize];
        mUo = new float[hiddenSize * hiddenSize];
        vUo = new float[hiddenSize * hiddenSize];
        mUg = new float[hiddenSize * hiddenSize];
        vUg = new float[hiddenSize * hiddenSize];
        mWhy = new float[vocabSize * hiddenSize];
        vWhy = new float[vocabSize * hiddenSize];

        mbf = new float[hiddenSize];
        vbf = new float[hiddenSize];
        mbi = new float[hiddenSize];
        vbi = new float[hiddenSize];
        mbo = new float[hiddenSize];
        vbo = new float[hiddenSize];
        mbg = new float[hiddenSize];
        vbg = new float[hiddenSize];
        mby = new float[vocabSize];
        vby = new float[vocabSize];
    }

    private float[] initArray(int size, float scale) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (float) (rand.nextGaussian() * scale);
        }
        return arr;
    }

    // ============================================================
    //         CONTEXTO DE HILO (ZERO-ALLOCATION)
    // ============================================================
    class ThreadContext {

        float[] h, c, fArr, iArr, oArr, gArr, tanhC, xArr, xScale, hScale;
        float[] dhNext, dcNext, dh, dc, df, di, dob, dg, dx, dOutput, logitsCache, hDropCache;
        Gradients grad;
        int maxSeq;

        ThreadContext(int maxSeqLen) {
            this.maxSeq = maxSeqLen;
            h = new float[maxSeqLen * hiddenSize];
            c = new float[maxSeqLen * hiddenSize];
            fArr = new float[maxSeqLen * hiddenSize];
            iArr = new float[maxSeqLen * hiddenSize];
            oArr = new float[maxSeqLen * hiddenSize];
            gArr = new float[maxSeqLen * hiddenSize];
            tanhC = new float[maxSeqLen * hiddenSize];
            xArr = new float[maxSeqLen * embeddingSize];
            xScale = new float[maxSeqLen * embeddingSize];
            hScale = new float[maxSeqLen * hiddenSize];

            dhNext = new float[hiddenSize];
            dcNext = new float[hiddenSize];
            dh = new float[hiddenSize];
            dc = new float[hiddenSize];
            df = new float[hiddenSize];
            di = new float[hiddenSize];
            dob = new float[hiddenSize];
            dg = new float[hiddenSize];
            dx = new float[embeddingSize];
            hDropCache = new float[hiddenSize];

            dOutput = new float[vocabSize];
            logitsCache = new float[vocabSize];
            grad = new Gradients();
        }

        void reset() {
            Arrays.fill(dhNext, 0f);
            Arrays.fill(dcNext, 0f);
            grad.reset();
        }
    }

    class Gradients {

        float loss = 0;
        float[] dWf, dWi, dWo, dWg, dUf, dUi, dUo, dUg, dbf, dbi, dbo, dbg, dWe, dWhy, dby;

        Gradients() {
            dWf = new float[hiddenSize * embeddingSize];
            dWi = new float[hiddenSize * embeddingSize];
            dWo = new float[hiddenSize * embeddingSize];
            dWg = new float[hiddenSize * embeddingSize];
            dUf = new float[hiddenSize * hiddenSize];
            dUi = new float[hiddenSize * hiddenSize];
            dUo = new float[hiddenSize * hiddenSize];
            dUg = new float[hiddenSize * hiddenSize];
            dbf = new float[hiddenSize];
            dbi = new float[hiddenSize];
            dbo = new float[hiddenSize];
            dbg = new float[hiddenSize];
            dWe = new float[vocabSize * embeddingSize];
            dWhy = new float[vocabSize * hiddenSize];
            dby = new float[vocabSize];
        }

        void reset() {
            loss = 0;
            Arrays.fill(dWf, 0f);
            Arrays.fill(dWi, 0f);
            Arrays.fill(dWo, 0f);
            Arrays.fill(dWg, 0f);
            Arrays.fill(dUf, 0f);
            Arrays.fill(dUi, 0f);
            Arrays.fill(dUo, 0f);
            Arrays.fill(dUg, 0f);
            Arrays.fill(dbf, 0f);
            Arrays.fill(dbi, 0f);
            Arrays.fill(dbo, 0f);
            Arrays.fill(dbg, 0f);
            Arrays.fill(dWe, 0f);
            Arrays.fill(dWhy, 0f);
            Arrays.fill(dby, 0f);
        }

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
            addArr(dbf, o.dbf);
            addArr(dbi, o.dbi);
            addArr(dbo, o.dbo);
            addArr(dbg, o.dbg);
            addArr(dWe, o.dWe);
            addArr(dWhy, o.dWhy);
            addArr(dby, o.dby);
        }

        void scale(float s) {
            scaleArr(dWf, s);
            scaleArr(dWi, s);
            scaleArr(dWo, s);
            scaleArr(dWg, s);
            scaleArr(dUf, s);
            scaleArr(dUi, s);
            scaleArr(dUo, s);
            scaleArr(dUg, s);
            scaleArr(dbf, s);
            scaleArr(dbi, s);
            scaleArr(dbo, s);
            scaleArr(dbg, s);
            scaleArr(dWe, s);
            scaleArr(dWhy, s);
            scaleArr(dby, s);
        }

        private void addArr(float[] a, float[] b) {
            for (int i = 0; i < a.length; i++) {
                a[i] += b[i];
            }
        }

        private void scaleArr(float[] a, float s) {
            for (int i = 0; i < a.length; i++) {
                a[i] *= s;
            }
        }
    }

    // ============================================================
    //                   PREPARACIÓN DE DATOS
    // ============================================================
    private List<int[]> prepareTrainingData(int seqLength) {
        List<int[]> seqs = new ArrayList<>();
        for (int start = 0; start + seqLength + 1 <= corpusIds.length; start += seqLength) {
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
        System.out.println("=== ENTRENAMIENTO LSTM OPTIMIZADO (Zero-Allocation & Flat Arrays) ===");
        System.out.printf("Vocab=%d | Batch=%d | Threads=%d | Dropout=%.2f | WD=%.3f%n",
                vocabSize, optimalBatchSize, optimalThreads, DROPOUT, WEIGHT_DECAY);

        List<int[]> sequences = prepareTrainingData(seqLength);
        if (sequences.isEmpty()) {
            System.out.println("⚠️ No hay secuencias suficientes para entrenar.");
            return;
        }

        Collections.shuffle(sequences, rand);
        int valSize = Math.max(1, sequences.size() / 20);
        List<int[]> valSeqs = sequences.subList(0, valSize);
        List<int[]> trainSeqs = sequences.subList(valSize, sequences.size());

        int batchesPerEpoch = (trainSeqs.size() + optimalBatchSize - 1) / optimalBatchSize;
        int totalSteps = epochs * batchesPerEpoch;
        int warmupSteps = Math.max(50, totalSteps / 20);
        int globalStep = 0;

        for (int epoch = currentEpoch; epoch < currentEpoch + epochs; epoch++) {
            long t0 = System.currentTimeMillis();
            Collections.shuffle(trainSeqs, rand);
            float totalLoss = 0;
            int batches = 0;

            for (int start = 0; start < trainSeqs.size(); start += optimalBatchSize) {
                int end = Math.min(start + optimalBatchSize, trainSeqs.size());
                List<int[]> batch = trainSeqs.subList(start, end);

                learningRate = getLR(globalStep, totalSteps, baseLearningRate, warmupSteps);
                float loss = processBatch(batch, seqLength);
                totalLoss += loss;
                batches++;
                globalStep++;

                if (batches % 10 == 0 || batches == batchesPerEpoch) {
                    System.out.printf("\r📊 Progreso: %d/%d | loss=%.4f | lr=%.6f",
                            batches, batchesPerEpoch, totalLoss / batches, learningRate);
                }
            }
            System.out.println();

            float trainLoss = totalLoss / Math.max(1, batches);
            float valLoss = evaluate(valSeqs, seqLength);
            long dt = System.currentTimeMillis() - t0;

            System.out.printf("✅ Epoca %d completada en %.1fs | train=%.4f | val=%.4f | ppl=%.2f%n",
                    epoch + 1, dt / 1000.0, trainLoss, valLoss, Math.exp(valLoss));

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                saveModel("best_model.ser");
                System.out.println("   💾 Nuevo mejor modelo guardado.");
            }
            saveModel("checkpoint_epoch_" + (epoch + 1) + ".ser");
            cleanOldCheckpoints(epoch + 1);
        }
        currentEpoch += epochs;
        System.out.println("\n🎉 Entrenamiento completado.");
    }

    private float getLR(int step, int total, float base, int warmup) {
        if (step < warmup) {
            return base * (step + 1.0f) / warmup;
        }
        float p = (float) (step - warmup) / Math.max(1, total - warmup);
        return (float) (base * 0.5 * (1.0 + Math.cos(Math.PI * p)));
    }

    private float processBatch(List<int[]> batch, int maxSeqLen) {
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }
        int chunkSize = Math.max(1, batch.size() / optimalThreads);
        List<Future<Gradients>> futures = new ArrayList<>();

        for (int i = 0; i < batch.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, batch.size());
            List<int[]> chunk = batch.subList(i, end);
            futures.add(threadPool.submit(() -> {
                ThreadContext ctx = new ThreadContext(maxSeqLen);
                for (int[] seq : chunk) {
                    ctx.grad.loss += trainSequence(seq, ctx);
                }
                return ctx.grad;
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
        total.scale(1.0f / batch.size());
        applyGradientClip(total);
        adamStep++;
        applyAdamW(total);
        return total.loss / batch.size();
    }

    // ============================================================
    //            FORWARD + BACKWARD (FLAT ARRAYS)
    // ============================================================
    private float trainSequence(int[] seq, ThreadContext ctx) {
        ctx.reset();
        int T = seq.length - 1;
        float loss = 0f;
        float[] prevH = new float[hiddenSize];
        float[] prevC = new float[hiddenSize];

        // ---------- FORWARD ----------
        for (int t = 0; t < T; t++) {
            int tokIdx = seq[t];
            int embOffset = tokIdx * embeddingSize;
            int xOffset = t * embeddingSize;
            int hOffset = t * hiddenSize;

            for (int k = 0; k < embeddingSize; k++) {
                ctx.xScale[xOffset + k] = (rand.nextFloat() >= DROPOUT) ? 1.0f / (1.0f - DROPOUT) : 0.0f;
                ctx.xArr[xOffset + k] = We[embOffset + k] * ctx.xScale[xOffset + k];
            }

            matVecMul(ctx.fArr, hOffset, Wf, ctx.xArr, xOffset, hiddenSize, embeddingSize);
            matVecMulAdd(ctx.fArr, hOffset, Uf, prevH, 0, hiddenSize, hiddenSize);
            addInPlace(ctx.fArr, hOffset, bf);
            sigmoidInPlace(ctx.fArr, hOffset, hiddenSize);

            matVecMul(ctx.iArr, hOffset, Wi, ctx.xArr, xOffset, hiddenSize, embeddingSize);
            matVecMulAdd(ctx.iArr, hOffset, Ui, prevH, 0, hiddenSize, hiddenSize);
            addInPlace(ctx.iArr, hOffset, bi);
            sigmoidInPlace(ctx.iArr, hOffset, hiddenSize);

            matVecMul(ctx.oArr, hOffset, Wo, ctx.xArr, xOffset, hiddenSize, embeddingSize);
            matVecMulAdd(ctx.oArr, hOffset, Uo, prevH, 0, hiddenSize, hiddenSize);
            addInPlace(ctx.oArr, hOffset, bo);
            sigmoidInPlace(ctx.oArr, hOffset, hiddenSize);

            matVecMul(ctx.gArr, hOffset, Wg, ctx.xArr, xOffset, hiddenSize, embeddingSize);
            matVecMulAdd(ctx.gArr, hOffset, Ug, prevH, 0, hiddenSize, hiddenSize);
            addInPlace(ctx.gArr, hOffset, bg);
            tanhInPlace(ctx.gArr, hOffset, hiddenSize);

            for (int j = 0; j < hiddenSize; j++) {
                ctx.c[hOffset + j] = ctx.fArr[hOffset + j] * prevC[j] + ctx.iArr[hOffset + j] * ctx.gArr[hOffset + j];
                ctx.tanhC[hOffset + j] = fastTanh(ctx.c[hOffset + j]);
                ctx.h[hOffset + j] = ctx.oArr[hOffset + j] * ctx.tanhC[hOffset + j];
            }

            for (int k = 0; k < hiddenSize; k++) {
                ctx.hScale[hOffset + k] = (rand.nextFloat() >= DROPOUT) ? 1.0f / (1.0f - DROPOUT) : 0.0f;
                ctx.hDropCache[k] = ctx.h[hOffset + k] * ctx.hScale[hOffset + k];
            }

            matVecMul(ctx.logitsCache, 0, Why, ctx.hDropCache, 0, vocabSize, hiddenSize);
            addInPlace(ctx.logitsCache, 0, by);
            softmaxInPlace(ctx.logitsCache);

            int target = seq[t + 1];
            loss += (float) -Math.log(Math.max(ctx.logitsCache[target], 1e-10f));

            System.arraycopy(ctx.h, hOffset, prevH, 0, hiddenSize);
            System.arraycopy(ctx.c, hOffset, prevC, 0, hiddenSize);
        }

        // ---------- BACKWARD ----------
        for (int t = T - 1; t >= 0; t--) {
            int target = seq[t + 1];
            int tokIdx = seq[t];
            int xOffset = t * embeddingSize;
            int hOffset = t * hiddenSize;

            for (int k = 0; k < hiddenSize; k++) {
                ctx.hDropCache[k] = ctx.h[hOffset + k] * ctx.hScale[hOffset + k];
            }
            matVecMul(ctx.logitsCache, 0, Why, ctx.hDropCache, 0, vocabSize, hiddenSize);
            addInPlace(ctx.logitsCache, 0, by);
            softmaxInPlace(ctx.logitsCache);

            Arrays.fill(ctx.dOutput, 0f);
            ctx.dOutput[target] = -1.0f;
            for (int k = 0; k < vocabSize; k++) {
                ctx.dOutput[k] += ctx.logitsCache[k];
            }

            for (int k = 0; k < vocabSize; k++) {
                float dk = ctx.dOutput[k];
                int whyRow = k * hiddenSize;
                for (int j = 0; j < hiddenSize; j++) {
                    ctx.grad.dWhy[whyRow + j] += dk * ctx.hDropCache[j];
                }
                ctx.grad.dby[k] += dk;
            }

            Arrays.fill(ctx.dh, 0f);
            for (int k = 0; k < vocabSize; k++) {
                float dk = ctx.dOutput[k];
                int whyRow = k * hiddenSize;
                for (int j = 0; j < hiddenSize; j++) {
                    ctx.dh[j] += dk * Why[whyRow + j] * ctx.hScale[hOffset + j];
                }
            }
            for (int j = 0; j < hiddenSize; j++) {
                ctx.dh[j] += ctx.dhNext[j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                float tc = ctx.tanhC[hOffset + j];
                ctx.dc[j] = ctx.dh[j] * ctx.oArr[hOffset + j] * (1.0f - tc * tc) + ctx.dcNext[j];
                ctx.df[j] = ctx.dc[j] * (t > 0 ? ctx.c[(t - 1) * hiddenSize + j] : 0.0f) * ctx.fArr[hOffset + j] * (1.0f - ctx.fArr[hOffset + j]);
                ctx.di[j] = ctx.dc[j] * ctx.gArr[hOffset + j] * ctx.iArr[hOffset + j] * (1.0f - ctx.iArr[hOffset + j]);
                ctx.dob[j] = ctx.dh[j] * tc * ctx.oArr[hOffset + j] * (1.0f - ctx.oArr[hOffset + j]);
                ctx.dg[j] = ctx.dc[j] * ctx.iArr[hOffset + j] * (1.0f - ctx.gArr[hOffset + j] * ctx.gArr[hOffset + j]);
            }

            Arrays.fill(ctx.dx, 0f);
            for (int j = 0; j < hiddenSize; j++) {
                float dfj = ctx.df[j], dij = ctx.di[j], doj = ctx.dob[j], dgj = ctx.dg[j];
                int wRow = j * embeddingSize;
                int uRow = j * hiddenSize;

                for (int k = 0; k < embeddingSize; k++) {
                    ctx.grad.dWf[wRow + k] += dfj * ctx.xArr[xOffset + k];
                    ctx.grad.dWi[wRow + k] += dij * ctx.xArr[xOffset + k];
                    ctx.grad.dWo[wRow + k] += doj * ctx.xArr[xOffset + k];
                    ctx.grad.dWg[wRow + k] += dgj * ctx.xArr[xOffset + k];
                    ctx.dx[k] += dfj * Wf[wRow + k] + dij * Wi[wRow + k] + doj * Wo[wRow + k] + dgj * Wg[wRow + k];
                }
                ctx.grad.dbf[j] += dfj;
                ctx.grad.dbi[j] += dij;
                ctx.grad.dbo[j] += doj;
                ctx.grad.dbg[j] += dgj;

                for (int k = 0; k < hiddenSize; k++) {
                    float prevHk = (t > 0) ? ctx.h[(t - 1) * hiddenSize + k] : 0.0f;
                    ctx.grad.dUf[uRow + k] += dfj * prevHk;
                    ctx.grad.dUi[uRow + k] += dij * prevHk;
                    ctx.grad.dUo[uRow + k] += doj * prevHk;
                    ctx.grad.dUg[uRow + k] += dgj * prevHk;
                }
            }

            int embOffset = tokIdx * embeddingSize;
            for (int k = 0; k < embeddingSize; k++) {
                ctx.grad.dWe[embOffset + k] += ctx.dx[k] * ctx.xScale[xOffset + k];
            }

            Arrays.fill(ctx.dhNext, 0f);
            for (int j = 0; j < hiddenSize; j++) {
                float dfj = ctx.df[j], dij = ctx.di[j], doj = ctx.dob[j], dgj = ctx.dg[j];
                int uRow = j * hiddenSize;
                for (int k = 0; k < hiddenSize; k++) {
                    ctx.dhNext[k] += dfj * Uf[uRow + k] + dij * Ui[uRow + k] + doj * Uo[uRow + k] + dgj * Ug[uRow + k];
                }
            }
            for (int j = 0; j < hiddenSize; j++) {
                ctx.dcNext[j] = ctx.dc[j] * ctx.fArr[hOffset + j];
            }
        }
        return loss / T;
    }

    private float evaluate(List<int[]> seqs, int maxSeqLen) {
        float total = 0;
        ThreadContext ctx = new ThreadContext(maxSeqLen);
        for (int[] seq : seqs) {
            Arrays.fill(ctx.h, 0f);
            Arrays.fill(ctx.c, 0f);
            float loss = 0;
            int T = seq.length - 1;
            for (int t = 0; t < T; t++) {
                int tokIdx = seq[t];
                int embOffset = tokIdx * embeddingSize;

                System.arraycopy(We, embOffset, ctx.xArr, 0, embeddingSize);

                matVecMul(ctx.fArr, 0, Wf, ctx.xArr, 0, hiddenSize, embeddingSize);
                matVecMulAdd(ctx.fArr, 0, Uf, ctx.h, 0, hiddenSize, hiddenSize);
                addInPlace(ctx.fArr, 0, bf);
                sigmoidInPlace(ctx.fArr, 0, hiddenSize);

                matVecMul(ctx.iArr, 0, Wi, ctx.xArr, 0, hiddenSize, embeddingSize);
                matVecMulAdd(ctx.iArr, 0, Ui, ctx.h, 0, hiddenSize, hiddenSize);
                addInPlace(ctx.iArr, 0, bi);
                sigmoidInPlace(ctx.iArr, 0, hiddenSize);

                matVecMul(ctx.oArr, 0, Wo, ctx.xArr, 0, hiddenSize, embeddingSize);
                matVecMulAdd(ctx.oArr, 0, Uo, ctx.h, 0, hiddenSize, hiddenSize);
                addInPlace(ctx.oArr, 0, bo);
                sigmoidInPlace(ctx.oArr, 0, hiddenSize);

                matVecMul(ctx.gArr, 0, Wg, ctx.xArr, 0, hiddenSize, embeddingSize);
                matVecMulAdd(ctx.gArr, 0, Ug, ctx.h, 0, hiddenSize, hiddenSize);
                addInPlace(ctx.gArr, 0, bg);
                tanhInPlace(ctx.gArr, 0, hiddenSize);

                for (int j = 0; j < hiddenSize; j++) {
                    ctx.c[j] = ctx.fArr[j] * ctx.c[j] + ctx.iArr[j] * ctx.gArr[j];
                    ctx.tanhC[j] = fastTanh(ctx.c[j]);
                    ctx.h[j] = ctx.oArr[j] * ctx.tanhC[j];
                }

                matVecMul(ctx.logitsCache, 0, Why, ctx.h, 0, vocabSize, hiddenSize);
                addInPlace(ctx.logitsCache, 0, by);
                softmaxInPlace(ctx.logitsCache);

                loss += (float) -Math.log(Math.max(ctx.logitsCache[seq[t + 1]], 1e-10f));
            }
            total += loss / T;
        }
        return total / seqs.size();
    }

    private void applyGradientClip(Gradients g) {
        float n = 0;
        for (float v : g.dWf) {
            n += v * v;
        }
        for (float v : g.dWi) {
            n += v * v;
        }
        for (float v : g.dWo) {
            n += v * v;
        }
        for (float v : g.dWg) {
            n += v * v;
        }
        for (float v : g.dUf) {
            n += v * v;
        }
        for (float v : g.dUi) {
            n += v * v;
        }
        for (float v : g.dUo) {
            n += v * v;
        }
        for (float v : g.dUg) {
            n += v * v;
        }
        for (float v : g.dWe) {
            n += v * v;
        }
        for (float v : g.dWhy) {
            n += v * v;
        }
        float norm = (float) Math.sqrt(n);
        if (norm > GRAD_CLIP) {
            g.scale(GRAD_CLIP / norm);
        }
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

    private void adamUpdate(float[] W, float[] dW, float[] m, float[] v) {
        float c1 = 1.0f - (float) Math.pow(beta1, adamStep);
        float c2 = 1.0f - (float) Math.pow(beta2, adamStep);
        for (int i = 0; i < W.length; i++) {
            float g = dW[i];
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;
            float adam = (m[i] / c1) / ((float) Math.sqrt(v[i] / c2) + epsilon);
            W[i] -= learningRate * (adam + WEIGHT_DECAY * W[i]);
        }
    }

    // ============================================================
    //       GENERACIÓN DE TEXTO COHERENTE (TOP-P NUCLEUS)
    // ============================================================
    public String generate(String seed, int maxLength, float temperature, float topP) {
        List<String> seedTokens = tokenize(seed);
        ThreadContext ctx = new ThreadContext(1);

        int startId = wordToIndex.getOrDefault("<START>", 0);
        forwardStepGen(startId, ctx);

        List<String> out = new ArrayList<>();
        for (String tok : seedTokens) {
            int id = wordToIndex.getOrDefault(tok, wordToIndex.get("<UNK>"));
            forwardStepGen(id, ctx);
            out.add(tok);
        }

        for (int step = 0; step < maxLength; step++) {
            matVecMul(ctx.logitsCache, 0, Why, ctx.h, 0, vocabSize, hiddenSize);
            addInPlace(ctx.logitsCache, 0, by);

            float temp = Math.max(1e-6f, temperature);
            for (int j = 0; j < vocabSize; j++) {
                ctx.logitsCache[j] /= temp;
            }

            int next = sampleTopP(ctx.logitsCache, topP);
            String token = indexToWord.get(next);
            if ("<END>".equals(token)) {
                break;
            }

            if ("<START>".equals(token) || "<UNK>".equals(token)) {
                forwardStepGen(next, ctx);
                continue;
            }
            out.add(token);
            forwardStepGen(next, ctx);
        }
        return detokenize(out);
    }

    private void forwardStepGen(int id, ThreadContext ctx) {
        int embOffset = id * embeddingSize;
        System.arraycopy(We, embOffset, ctx.xArr, 0, embeddingSize);

        matVecMul(ctx.fArr, 0, Wf, ctx.xArr, 0, hiddenSize, embeddingSize);
        matVecMulAdd(ctx.fArr, 0, Uf, ctx.h, 0, hiddenSize, hiddenSize);
        addInPlace(ctx.fArr, 0, bf);
        sigmoidInPlace(ctx.fArr, 0, hiddenSize);

        matVecMul(ctx.iArr, 0, Wi, ctx.xArr, 0, hiddenSize, embeddingSize);
        matVecMulAdd(ctx.iArr, 0, Ui, ctx.h, 0, hiddenSize, hiddenSize);
        addInPlace(ctx.iArr, 0, bi);
        sigmoidInPlace(ctx.iArr, 0, hiddenSize);

        matVecMul(ctx.oArr, 0, Wo, ctx.xArr, 0, hiddenSize, embeddingSize);
        matVecMulAdd(ctx.oArr, 0, Uo, ctx.h, 0, hiddenSize, hiddenSize);
        addInPlace(ctx.oArr, 0, bo);
        sigmoidInPlace(ctx.oArr, 0, hiddenSize);

        matVecMul(ctx.gArr, 0, Wg, ctx.xArr, 0, hiddenSize, embeddingSize);
        matVecMulAdd(ctx.gArr, 0, Ug, ctx.h, 0, hiddenSize, hiddenSize);
        addInPlace(ctx.gArr, 0, bg);
        tanhInPlace(ctx.gArr, 0, hiddenSize);

        for (int j = 0; j < hiddenSize; j++) {
            ctx.c[j] = ctx.fArr[j] * ctx.c[j] + ctx.iArr[j] * ctx.gArr[j];
            ctx.tanhC[j] = fastTanh(ctx.c[j]);
            ctx.h[j] = ctx.oArr[j] * ctx.tanhC[j];
        }
    }

    private int sampleTopP(float[] logits, float topP) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }

        float sum = 0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = fastExp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        Integer[] indices = new Integer[probs.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

        float cumulative = 0.0f;
        int lastIndex = 0;
        for (int i = 0; i < indices.length; i++) {
            cumulative += probs[indices[i]];
            lastIndex = i;
            if (cumulative >= topP) {
                break;
            }
        }

        float r = rand.nextFloat() * cumulative;
        float currSum = 0;
        for (int i = 0; i <= lastIndex; i++) {
            currSum += probs[indices[i]];
            if (r <= currSum) {
                return indices[i];
            }
        }
        return indices[lastIndex];
    }

    // ============================================================
    //             MATEMÁTICAS (OPTIMIZADAS PARA CACHÉ)
    // ============================================================
    private void matVecMul(float[] y, int yOffset, float[] M, float[] x, int xOffset, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            float s = 0;
            int mRow = i * cols;
            for (int j = 0; j < cols; j++) {
                s += M[mRow + j] * x[xOffset + j];
            }
            y[yOffset + i] = s;
        }
    }

    private void matVecMulAdd(float[] y, int yOffset, float[] M, float[] x, int xOffset, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            float s = 0;
            int mRow = i * cols;
            for (int j = 0; j < cols; j++) {
                s += M[mRow + j] * x[xOffset + j];
            }
            y[yOffset + i] += s;
        }
    }

    private void addInPlace(float[] y, int yOffset, float[] x) {
        for (int k = 0; k < x.length; k++) {
            y[yOffset + k] += x[k];
        }
    }

    private void sigmoidInPlace(float[] x, int offset, int length) {
        for (int k = 0; k < length; k++) {
            x[offset + k] = 1.0f / (1.0f + fastExp(-x[offset + k]));
        }
    }

    private void tanhInPlace(float[] x, int offset, int length) {
        for (int k = 0; k < length; k++) {
            x[offset + k] = fastTanh(x[offset + k]);
        }
    }

    private void softmaxInPlace(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) {
                max = v;
            }
        }
        float sum = 0;
        for (int k = 0; k < x.length; k++) {
            x[k] = fastExp(x[k] - max);
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
        initAdamStates();
        threadPool = Executors.newFixedThreadPool(optimalThreads);
    }

    public void saveModel(String path) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(this);
            System.out.printf("💾 Modelo guardado: %s (%.1f MB)%n", path, new File(path).length() / 1024.0 / 1024.0);
        } catch (IOException e) {
            System.err.println("Error al guardar: " + e.getMessage());
        }
    }

    public static UltraLLM loadModel(String path) {
        long t0 = System.currentTimeMillis();
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            UltraLLM m = (UltraLLM) in.readObject();
            System.out.printf("✅ Modelo cargado en %.2fs%n", (System.currentTimeMillis() - t0) / 1000.0);
            return m;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error al cargar: " + e.getMessage());
            return null;
        }
    }

    private void cleanOldCheckpoints(int epoch) {
        int minKeep = Math.max(1, epoch - 3);
        for (int i = 1; i < minKeep; i++) {
            File f = new File("checkpoint_epoch_" + i + ".ser");
            if (f.exists()) {
                f.delete();
            }
        }
    }

    private void cleanup() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
            try {
                if (!threadPool.awaitTermination(3, TimeUnit.SECONDS)) {
                    threadPool.shutdownNow();
                }
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
        System.out.println("=== UltraLLM (1D Arrays, Zero-Allocation, Top-P) ===");
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
                System.out.print("Máx conversaciones [2000]: ");
                String maxStr = sc.nextLine().trim();
                int max = maxStr.isEmpty() ? 2000 : Integer.parseInt(maxStr);
                try {
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
                    System.out.println("⚠️ Carga corpus primero.");
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
                UltraLLM m = loadModel(p);
                if (m != null) {
                    m.startInteractiveMode();
                    return;
                }
            } else if (!input.isEmpty()) {
                if (vocabSize == 0) {
                    System.out.println("⚠️ Modelo no entrenado.");
                    continue;
                }
                System.out.print("Longitud [100]: ");
                String s = sc.nextLine().trim();
                int len = s.isEmpty() ? 100 : Integer.parseInt(s);
                System.out.print("Temperatura [0.7]: ");
                s = sc.nextLine().trim();
                float t = s.isEmpty() ? 0.7f : Float.parseFloat(s);
                System.out.print("Top-P [0.9]: ");
                s = sc.nextLine().trim();
                float topP = s.isEmpty() ? 0.9f : Float.parseFloat(s);

                System.out.println("🤖 Generando texto...");
                long startT = System.currentTimeMillis();
                String res = generate(input, len, t, topP);
                System.out.printf("\n📜 Generado en %dms:\n%s\n", (System.currentTimeMillis() - startT), res);
            }
        }
        sc.close();
    }

    public static void main(String[] args) {
        UltraLLM m = new UltraLLM();
        Runtime.getRuntime().addShutdownHook(new Thread(m::cleanup));
        try {
            m.startInteractiveMode();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
