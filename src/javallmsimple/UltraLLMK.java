package javallmsimple;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class UltraLLMK implements Serializable {

    private static final long serialVersionUID = 11L;

    // ===================== Vocabulario Char-Level =====================
    private final Map<Character, Integer> charToIndex = new HashMap<>();
    private final Map<Integer, Character> indexToChar = new HashMap<>();
    private int vocabSize = 0;
    private int[] corpusIds;

    // ===================== Arquitectura (1D Flat Arrays) =====================
    private final int embeddingSize;
    private final int hiddenSize;

    private float[] We;
    private float[] Wf, Wi, Wo, Wg;
    private float[] Uf, Ui, Uo, Ug;
    private float[] bf, bi, bo, bg;
    private float[] Why;
    private float[] by;

    // ===================== AdamW (transient) =====================
    private transient float[] mWe, mWf, mWi, mWo, mWg, mUf, mUi, mUo, mUg, mWhy;
    private transient float[] mbf, mbi, mbo, mbg, mby;
    private transient float[] vWe, vWf, vWi, vWo, vWg, vUf, vUi, vUo, vUg, vWhy;
    private transient float[] vbf, vbi, vbo, vbg, vby;

    // ===================== Hiperparámetros =====================
    private float learningRate = 0.001f;
    private final Random rand = new Random(42);

    private static final float DROPOUT = 0.10f;
    private static final float WEIGHT_DECAY = 1e-4f;
    private static final float GRAD_CLIP = 1.0f;

    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float epsilon = 1e-8f;
    private int adamStep = 0;
    private int currentEpoch = 0;
    private float bestValLoss = Float.POSITIVE_INFINITY;
    private Map<String, String> knowledgeBase = new HashMap<>();
    private Map<String, List<Integer>> invertedIndex = new HashMap<>();
    private List<String> dbQuestions = new ArrayList<>();
    private List<String> dbAnswers = new ArrayList<>();
    private static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "cual", "cuales", "quien", "quienes", "cuyo", "cuya", "cuyos", "cuyas", "cuanto", "cuanta", "cuantos", "cuantas", "ser", "es", "son", "fue", "era", "eran", "fui", "fue", "fueron", "era", "éramos", "seas", "sea", "seamos", "sean", "tenga", "tengan", "tiene", "tienen", "hay", "había", "hubo", "haya", "hayan", "dime", "dame", "nombra", "nombre", "escribe", "genera", "prepara", "proporcione", "haz", "crea", "explica", "cuentame"
    ));

    // ===================== Sistema (ForkJoin optimizado) =====================
    private int batchSize = 128; // Ideal para paralelismo en CPU (un hilo por sample)

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
            return 0.0f;
        }
        if (x >= MAX_X) {
            return EXP_LUT[LUT_SIZE - 1];
        }
        int idx = (int) ((x - MIN_X) * SCALE);
        return EXP_LUT[Math.min(idx, LUT_SIZE - 1)];
    }

    private static float fastTanh(float x) {
        if (x <= MIN_X) {
            return -1.0f;
        }
        if (x >= MAX_X) {
            return 1.0f;
        }
        int idx = (int) ((x - MIN_X) * SCALE);
        return TANH_LUT[Math.min(idx, LUT_SIZE - 1)];
    }

    public UltraLLMK() {
        //this(64, 256, 0.002f);
        this(128, 512, 0.001f);
    }

    public UltraLLMK(int embeddingSize, int hiddenSize, float lr) {
        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = lr;
    }

    // ============================================================
    //                   CARGA DE CORPUS (Char-level)
    // ============================================================
    public void loadAlpacaCorpus(String filePath) throws IOException {
        System.out.println("⏳ Cargando corpus Alpaca completo y construyendo Índice Invertido...");

        // PROTECCIÓN CONTRA NULLS 
        if (dbQuestions == null) {
            dbQuestions = new ArrayList<>();
        }
        if (dbAnswers == null) {
            dbAnswers = new ArrayList<>();
        }
        if (invertedIndex == null) {
            invertedIndex = new HashMap<>();
        }

        StringBuilder sb = new StringBuilder();
        dbQuestions.clear();
        dbAnswers.clear();
        invertedIndex.clear();

        int conv = 0;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                String q = extractJsonField(line, "question");
                String a = extractJsonField(line, "answer");
                if (q == null || a == null) {
                    continue;
                }

                // Limpiar texto para la base de datos
                String cleanQ = cleanText(q);

                dbQuestions.add(cleanQ);
                dbAnswers.add(a.trim());
                sb.append("Usuario: ").append(q).append("\nAsistente: ").append(a).append("\n\n");

                // Construir índice invertido (Palabra -> ID de Preguntas)
                String[] words = cleanQ.split(" ");
                int qId = dbQuestions.size() - 1;
                for (String w : words) {
                    if (w.length() < 3) {
                        continue;
                    }
                    invertedIndex.computeIfAbsent(w, k -> new ArrayList<>()).add(qId);
                }
                conv++;
            }
        }

        String fullText = sb.toString();
        System.out.println("Conversaciones leídas: " + conv);

        // Vocabulario
        Set<Character> uniqueChars = new TreeSet<>();
        for (int i = 0; i < fullText.length(); i++) {
            uniqueChars.add(fullText.charAt(i));
        }
        for (Character ch : uniqueChars) {
            if (!charToIndex.containsKey(ch)) {
                charToIndex.put(ch, vocabSize);
                indexToChar.put(vocabSize, ch);
                vocabSize++;
            }
        }

        System.out.printf("✅ Corpus Alpaca: %d caracteres. Vocabulario: %d%n", fullText.length(), vocabSize);

        corpusIds = new int[fullText.length()];
        for (int i = 0; i < fullText.length(); i++) {
            corpusIds[i] = charToIndex.get(fullText.charAt(i));
        }

        initializeWeights();
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
        // Reemplazar correctamente las comillas escapadas y los saltos de línea
        return json.substring(start, end)
                .replace("\\n", "\n")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
    }

    public void loadTextCorpus(String filePath) throws IOException {
        System.out.println("⏳ Cargando corpus en modo Char-Level...");
        File file = new File(filePath);
        long fileLen = file.length();
        char[] textBuffer = new char[(int) fileLen];
        int len = 0;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            int c;
            while ((c = br.read()) != -1) {
                textBuffer[len++] = (char) c;
            }
        }

        // Construir vocabulario único
        Set<Character> uniqueChars = new TreeSet<>();
        for (int i = 0; i < len; i++) {
            uniqueChars.add(textBuffer[i]);
        }

        for (Character ch : uniqueChars) {
            charToIndex.put(ch, vocabSize);
            indexToChar.put(vocabSize, ch);
            vocabSize++;
        }

        System.out.printf("✅ Corpus cargado: %d caracteres totales. Vocabulario único: %d%n", len, vocabSize);

        // Convertir a IDs
        corpusIds = new int[len];
        for (int i = 0; i < len; i++) {
            corpusIds[i] = charToIndex.get(textBuffer[i]);
        }

        initializeWeights();
    }

    // ============================================================
    //         INICIALIZACIÓN XAVIER/GLOROT UNIFORM (Karpathy style)
    // ============================================================
    private void initializeWeights() {
        int currentVocab = (We == null) ? 0 : We.length / embeddingSize;

        if (We != null && vocabSize == currentVocab) {
            return; // Ya está actualizado
        }
        if (We == null) {
            // Inicialización desde cero (Primera vez)
            We = xavierUniform(vocabSize, embeddingSize);
            Why = xavierUniform(vocabSize, hiddenSize);
            by = new float[vocabSize];
        } else {
            // Expansión: Mantener pesos viejos y añadir filas nuevas
            System.out.println("⚠️ Expandiendo matrices para nuevos caracteres (Conservando aprendizaje)...");

            float[] newWe = xavierUniform(vocabSize, embeddingSize);
            System.arraycopy(We, 0, newWe, 0, We.length);
            We = newWe;

            float[] newWhy = xavierUniform(vocabSize, hiddenSize);
            System.arraycopy(Why, 0, newWhy, 0, Why.length);
            Why = newWhy;

            float[] newBy = new float[vocabSize];
            System.arraycopy(by, 0, newBy, 0, by.length);
            by = newBy;
        }

        // Inicializar matrices internas (solo la primera vez)
        if (Wf == null) {
            Wf = xavierUniform(hiddenSize, embeddingSize);
            Wi = xavierUniform(hiddenSize, embeddingSize);
            Wo = xavierUniform(hiddenSize, embeddingSize);
            Wg = xavierUniform(hiddenSize, embeddingSize);
            Uf = xavierUniform(hiddenSize, hiddenSize);
            Ui = xavierUniform(hiddenSize, hiddenSize);
            Uo = xavierUniform(hiddenSize, hiddenSize);
            Ug = xavierUniform(hiddenSize, hiddenSize);
            bf = new float[hiddenSize];
            Arrays.fill(bf, 1.0f);
            bi = new float[hiddenSize];
            bo = new float[hiddenSize];
            bg = new float[hiddenSize];
        }

        initAdamStates();
    }

    private float[] xavierUniform(int fanIn, int fanOut) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[] arr = new float[fanIn * fanOut];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (rand.nextFloat() * 2 * limit) - limit;
        }
        return arr;
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

    // ============================================================
    //         CONTEXTO DE HILO (ZERO-ALLOCATION)
    // ============================================================
    class ThreadContext {

        float[] h, c, fArr, iArr, oArr, gArr, tanhC, xArr, xScale, hScale;
        float[] dhNext, dcNext, dh, dc, df, di, dob, dg, dx, dOutput, logitsCache, hDropCache;
        Gradients grad;

        ThreadContext(int maxSeqLen) {
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

        synchronized void add(Gradients o) {
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

        private void addArr(float[] a, float[] b) {
            for (int i = 0; i < a.length; i++) {
                a[i] += b[i];
            }
        }

        void scale(float s) {
            for (int i = 0; i < dWf.length; i++) {
                dWf[i] *= s;
                dWi[i] *= s;
                dWo[i] *= s;
                dWg[i] *= s;
            }
            for (int i = 0; i < dUf.length; i++) {
                dUf[i] *= s;
                dUi[i] *= s;
                dUo[i] *= s;
                dUg[i] *= s;
            }
            for (int i = 0; i < dbf.length; i++) {
                dbf[i] *= s;
                dbi[i] *= s;
                dbo[i] *= s;
                dbg[i] *= s;
            }
            for (int i = 0; i < dWe.length; i++) {
                dWe[i] *= s;
            }
            for (int i = 0; i < dWhy.length; i++) {
                dWhy[i] *= s;
            }
            for (int i = 0; i < dby.length; i++) {
                dby[i] *= s;
            }
        }
    }

    // ============================================================
    //                   PREPARACIÓN DE DATOS (Mini-batches)
    // ============================================================
    private List<int[]> prepareTrainingData(int seqLength) {
        List<int[]> seqs = new ArrayList<>();
        int stride = seqLength; // Sin traslape para ir mas rapido
        for (int start = 0; start + seqLength + 1 <= corpusIds.length; start += stride) {
            int[] s = new int[seqLength + 1];
            System.arraycopy(corpusIds, start, s, 0, seqLength + 1);
            seqs.add(s);
        }
        return seqs;
    }

    // ============================================================
    //                   ENTRENAMIENTO BATCHED PARALELO
    // ============================================================
    public void train(int epochs, int seqLength) {
        System.out.println("=== ENTRENAMIENTO CHAR-RNN (Optimizado Estilo Karpathy) ===");
        System.out.printf("Vocab=%d | Batch=%d | LR=%.5f | SeqLen=%d%n", vocabSize, batchSize, learningRate, seqLength);

        List<int[]> sequences = prepareTrainingData(seqLength);
        if (sequences.isEmpty()) {
            System.out.println("⚠️ No hay secuencias.");
            return;
        }

        Collections.shuffle(sequences, rand);
        int valSize = Math.max(1, sequences.size() / 20);
        List<int[]> valSeqs = sequences.subList(0, valSize);
        List<int[]> trainSeqs = sequences.subList(valSize, sequences.size());

        int batchesPerEpoch = (trainSeqs.size() + batchSize - 1) / batchSize;

        // Asignamos un ThreadContext por hilo para evitar GC
        ThreadLocal<ThreadContext> localCtx = ThreadLocal.withInitial(() -> new ThreadContext(seqLength));

        for (int epoch = currentEpoch; epoch < currentEpoch + epochs; epoch++) {
            long t0 = System.currentTimeMillis();
            Collections.shuffle(trainSeqs, rand);

            float totalLoss = 0;
            int batches = 0;

            for (int start = 0; start < trainSeqs.size(); start += batchSize) {
                int end = Math.min(start + batchSize, trainSeqs.size());
                List<int[]> batch = trainSeqs.subList(start, end);

                Gradients batchGrads = new Gradients();

                // Paralelismo de Grano Fino por Secuencia (ForkJoin nativo de Java)
                batch.parallelStream().forEach(seq -> {
                    ThreadContext ctx = localCtx.get();
                    ctx.reset();
                    float loss = trainSequence(seq, ctx);
                    ctx.grad.loss = loss;
                    batchGrads.add(ctx.grad);
                });

                batchGrads.scale(1.0f / batch.size());
                applyGradientClip(batchGrads);
                adamStep++;
                applyAdamW(batchGrads);

                totalLoss += batchGrads.loss;
                batches++;

                if (batches % 10 == 0 || batches == batchesPerEpoch) {
                    System.out.printf("\r📊 Progreso: %d/%d | loss=%.4f ", batches, batchesPerEpoch, totalLoss / batches);
                }
            }
            System.out.println();

            float trainLoss = totalLoss / Math.max(1, batches);

            // Validacion rápida
            double valLoss = valSeqs.parallelStream().mapToDouble(seq -> evaluate(seq, localCtx.get())).average().orElse(0);

            long dt = System.currentTimeMillis() - t0;
            System.out.printf("✅ Epoca %d en %.1fs | train=%.4f | val=%.4f%n", epoch + 1, dt / 1000.0, trainLoss, valLoss);

            if (valLoss < bestValLoss) {
                bestValLoss = (float) valLoss;
                saveModel("best_model_char.ser");
            }
        }
        currentEpoch += epochs;
        System.out.println("\n🎉 Entrenamiento completado.");
    }

    private void applyGradientClip(Gradients g) {
        float n = 0;
        for (float v : g.dWf) {
            n += v * v;
        }
        for (float v : g.dUf) {
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
        float c1 = 1.0f - (float) Math.pow(beta1, adamStep);
        float c2 = 1.0f - (float) Math.pow(beta2, adamStep);

        adamUpdate(Wf, g.dWf, mWf, vWf, c1, c2);
        adamUpdate(Wi, g.dWi, mWi, vWi, c1, c2);
        adamUpdate(Wo, g.dWo, mWo, vWo, c1, c2);
        adamUpdate(Wg, g.dWg, mWg, vWg, c1, c2);
        adamUpdate(Uf, g.dUf, mUf, vUf, c1, c2);
        adamUpdate(Ui, g.dUi, mUi, vUi, c1, c2);
        adamUpdate(Uo, g.dUo, mUo, vUo, c1, c2);
        adamUpdate(Ug, g.dUg, mUg, vUg, c1, c2);
        adamUpdate(bf, g.dbf, mbf, vbf, c1, c2);
        adamUpdate(bi, g.dbi, mbi, vbi, c1, c2);
        adamUpdate(bo, g.dbo, mbo, vbo, c1, c2);
        adamUpdate(bg, g.dbg, mbg, vbg, c1, c2);
        adamUpdate(We, g.dWe, mWe, vWe, c1, c2);
        adamUpdate(Why, g.dWhy, mWhy, vWhy, c1, c2);
        adamUpdate(by, g.dby, mby, vby, c1, c2);
    }

    private void adamUpdate(float[] W, float[] dW, float[] m, float[] v, float c1, float c2) {
        for (int i = 0; i < W.length; i++) {
            float grad = dW[i];
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            float adam = (m[i] / c1) / ((float) Math.sqrt(v[i] / c2) + epsilon);
            W[i] -= learningRate * (adam + WEIGHT_DECAY * W[i]);
        }
    }

    // ============================================================
    //            FORWARD + BACKWARD LSTM
    // ============================================================
    private float trainSequence(int[] seq, ThreadContext ctx) {
        int T = seq.length - 1;
        float loss = 0f;
        float[] prevH = new float[hiddenSize];
        float[] prevC = new float[hiddenSize];

        // FORWARD
        for (int t = 0; t < T; t++) {
            int tokIdx = seq[t];
            int embOffset = tokIdx * embeddingSize;
            int xOffset = t * embeddingSize;
            int hOffset = t * hiddenSize;

            for (int k = 0; k < embeddingSize; k++) {
                // Char level no necesita Dropout pesado, lo bajamos o ignoramos para mayor precision
                ctx.xScale[xOffset + k] = 1.0f;
                ctx.xArr[xOffset + k] = We[embOffset + k];
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
                ctx.hScale[hOffset + k] = 1.0f;
                ctx.hDropCache[k] = ctx.h[hOffset + k];
            }

            matVecMul(ctx.logitsCache, 0, Why, ctx.hDropCache, 0, vocabSize, hiddenSize);
            addInPlace(ctx.logitsCache, 0, by);
            softmaxInPlace(ctx.logitsCache);

            int target = seq[t + 1];
            loss += (float) -Math.log(Math.max(ctx.logitsCache[target], 1e-10f));

            System.arraycopy(ctx.h, hOffset, prevH, 0, hiddenSize);
            System.arraycopy(ctx.c, hOffset, prevC, 0, hiddenSize);
        }

        // BACKWARD
        for (int t = T - 1; t >= 0; t--) {
            int target = seq[t + 1];
            int tokIdx = seq[t];
            int xOffset = t * embeddingSize;
            int hOffset = t * hiddenSize;

            for (int k = 0; k < hiddenSize; k++) {
                ctx.hDropCache[k] = ctx.h[hOffset + k];
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
                    ctx.dh[j] += dk * Why[whyRow + j];
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
                ctx.grad.dWe[embOffset + k] += ctx.dx[k];
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

    private float evaluate(int[] seq, ThreadContext ctx) {
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
        return loss / T;
    }

    // ============================================================
    //       GENERACIÓN DE TEXTO COHERENTE (Char-by-Char)
    // ============================================================
    public String generate(String seed, int maxLength, float temperature, float topP) {
        ThreadContext ctx = new ThreadContext(1);
        StringBuilder out = new StringBuilder();

        // Procesar el seed
        for (int i = 0; i < seed.length(); i++) {
            char ch = seed.charAt(i);
            int id = charToIndex.getOrDefault(ch, 0);
            forwardStepGen(id, ctx);
            out.append(ch);
        }

        // Ventana deslizante para evitar repeticiones
        List<Integer> recentTokens = new ArrayList<>();

        for (int step = 0; step < maxLength; step++) {
            matVecMul(ctx.logitsCache, 0, Why, ctx.h, 0, vocabSize, hiddenSize);
            addInPlace(ctx.logitsCache, 0, by);

            float temp = Math.max(1e-6f, temperature);
            for (int j = 0; j < vocabSize; j++) {
                ctx.logitsCache[j] /= temp;
            }

            // Penalización por repetición (Últimas 20 letras)
            for (int tokenId : recentTokens) {
                ctx.logitsCache[tokenId] -= 1.5f;
            }

            int next = sampleTopP(ctx.logitsCache, topP);
            char token = indexToChar.get(next);

            out.append(token);
            forwardStepGen(next, ctx);

            //Filtro anti-rols
            // Si el modelo intenta empezar otra conversación, la corto
            if (out.toString().endsWith("\nUsuario:") || out.toString().endsWith("\nAsistente:")) {
                break;
            }

            // Actualizar la ventana
            recentTokens.add(next);
            if (recentTokens.size() > 20) {
                recentTokens.remove(0); // Elimina la más antigua
            }
        }
        return out.toString();
    }

    private String cleanText(String text) {
        if (text == null) {
            return "";
        }
        String normalized = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFD);
        return normalized.replaceAll("\\p{M}", "")
                .toLowerCase()
                .replace("usuario:", "")
                .replace("asistente:", "")
                .replaceAll("[^a-z0-9 ]", " ")
                .replaceAll("\\s+", " ")
                .trim();
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
    //             MATEMÁTICAS (Caché Optimizada)
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
    //                   SERIALIZACIÓN Y UTILIDADES
    // ============================================================
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        initAdamStates();
        if (knowledgeBase == null) {
            knowledgeBase = new HashMap<>();
        }
        if (dbQuestions == null) {
            dbQuestions = new ArrayList<>();
        }
        if (dbAnswers == null) {
            dbAnswers = new ArrayList<>();
        }
        if (invertedIndex == null) {
            invertedIndex = new HashMap<>();
        }

    }

    public void saveModel(String path) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(this);
            System.out.printf("💾 Modelo guardado: %s (%.1f MB)%n", path, new File(path).length() / 1024.0 / 1024.0);
        } catch (IOException e) {
            System.err.println("Error al guardar: " + e.getMessage());
        }
    }

    public static UltraLLMK loadModel(String path) {
        long t0 = System.currentTimeMillis();
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            UltraLLMK m = (UltraLLMK) in.readObject();
            System.out.printf("✅ Modelo cargado en %.2fs%n", (System.currentTimeMillis() - t0) / 1000.0);
            return m;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error al cargar: " + e.getMessage());
            return null;
        }
    }

    // ============================================================
    //                   MENÚ INTERACTIVO
    // ============================================================
    public void startInteractiveMode() throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.println("=== UltraLLMK (Karpathy Char-RNN Style) ===");
        System.out.println("Comandos: LOAD_ALPACA, LOAD_TEXT, TRAIN, SAVE, LOAD, STATUS, EXIT");
        while (true) {
            System.out.print("\n> ");
            String input = sc.nextLine().trim();
            if (input.equalsIgnoreCase("EXIT")) {
                break;
            } else if (input.equalsIgnoreCase("STATUS")) {
                System.out.printf("Vocab=%d | Corpus=%d | Batch=%d | Epochs=%d%n",
                        vocabSize, corpusIds == null ? 0 : corpusIds.length,
                        batchSize, currentEpoch);
                System.out.printf("Best val_loss=%.4f%n", bestValLoss);
            } else if (input.equalsIgnoreCase("LOAD_ALPACA")) {
                System.out.print("Archivo .jsonl: ");
                String p = sc.nextLine().trim();
                try {
                    loadAlpacaCorpus(p);
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
                UltraLLMK m = loadModel(p);
                if (m != null) {
                    m.startInteractiveMode();
                    return;
                }
            } else if (!input.isEmpty()) {
                if (vocabSize == 0) {
                    System.out.println("⚠️ Modelo no entrenado.");
                    continue;
                }

                String cleanQuery = cleanText(input);
                String[] queryWords = cleanQuery.split(" ");

                // BÚSQUEDA INTELIGENTE (TF-IDF Simplificado con Stopwords) 
                int bestId = -1;
                double bestScore = 0.0;

                // Calcular la rareza (IDF) de las palabras de la consulta
                Map<String, Double> queryWeights = new HashMap<>();
                for (String qw : queryWords) {
                    if (qw.length() < 3 || STOPWORDS.contains(qw)) {
                        continue;
                    }
                    // Una regla de "stemming" muy simple para español
                    if (qw.endsWith("es") && qw.length() > 4) {
                        qw = qw.substring(0, qw.length() - 2);
                    } else if (qw.endsWith("s") && qw.length() > 3) {
                        qw = qw.substring(0, qw.length() - 1);
                    }
                    // IDF: Cuanto menos aparece una palabra, más importante es.
                    // Usamos el tamaño del índice invertido como medida de frecuencia.
                    int docFreq = invertedIndex.containsKey(qw) ? invertedIndex.get(qw).size() : 1;
                    double idf = Math.log((double) dbQuestions.size() / (docFreq + 1)) + 1.0;
                    queryWeights.put(qw, idf);
                }

                // Puntuar cada pregunta de la base de datos
                Map<Integer, Double> docScores = new HashMap<>();
                for (Map.Entry<String, Double> qEntry : queryWeights.entrySet()) {
                    String qw = qEntry.getKey();
                    double weight = qEntry.getValue();

                    List<Integer> qIds = invertedIndex.get(qw);
                    if (qIds != null) {
                        for (int id : qIds) {
                            docScores.put(id, docScores.getOrDefault(id, 0.0) + weight);
                        }
                    }
                }

                // Encontrar la mejor coincidencia
                for (Map.Entry<Integer, Double> entry : docScores.entrySet()) {
                    int id = entry.getKey();
                    double score = entry.getValue();

                    // Penalización por longitud: si la pregunta de la BD es muy larga, su puntaje baja.
                    // Esto evita que la "carta de recomendación" gane siempre.
                    String dbQ = dbQuestions.get(id);
                    int dbWordCount = dbQ.split(" ").length;
                    score = score / Math.sqrt(dbWordCount);

                    if (score > bestScore) {
                        bestScore = score;
                        bestId = id;
                    }
                }

                // Umbral de confianza (ajustado para TF-IDF)
                if (bestId != -1 && bestScore > 1.5) {
                    System.out.println("\n🧠 [Respuesta recuperada de la BD (Score: " + String.format("%.2f", bestScore) + ")]:");
                    System.out.println("📌 Pregunta en BD: \"" + dbQuestions.get(bestId) + "\"");
                    System.out.println("💡 Respuesta: " + dbAnswers.get(bestId));
                    continue;
                }

                // --- FALLBACK AL LSTM ---
                System.out.println("⚠️ Pregunta no encontrada exactamente. Improvisando con LSTM...");
                System.out.print("Longitud [300]: ");
                String s = sc.nextLine().trim();
                int len = s.isEmpty() ? 300 : Integer.parseInt(s);
                System.out.print("Temperatura [0.3]: ");
                s = sc.nextLine().trim();
                float t = s.isEmpty() ? 0.3f : Float.parseFloat(s);
                System.out.print("Top-P [0.5]: ");
                s = sc.nextLine().trim();
                float topP = s.isEmpty() ? 0.5f : Float.parseFloat(s);

                System.out.println("🤖 Generando texto...");
                long startT = System.currentTimeMillis();

                String prompt = "Usuario: " + input + "\nAsistente: ";
                String res = generate(prompt, len, t, topP);

                res = res.replace(prompt, "").trim();
                if (res.contains("Usuario:")) {
                    res = res.substring(0, res.indexOf("Usuario:")).trim();
                }
                if (res.contains("Asistente:")) {
                    res = res.substring(0, res.indexOf("Asistente:")).trim();
                }

                System.out.printf("\n📜 Generado en %dms:\n%s\n", (System.currentTimeMillis() - startT), res);
            }
        }
        sc.close();
    }

    public static void main(String[] args) {
        UltraLLMK m = new UltraLLMK();
        try {
            m.startInteractiveMode();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
