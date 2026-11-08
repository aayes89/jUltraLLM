/*
 * The MIT License
 *
 * Copyright 2025 Slam.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package javallmsimple;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

// Importaciones de TornadoVM
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class ImprovedLanguageModel implements Serializable {

    private static final long serialVersionUID = 4L;

    // --- Modos de ejecución ---
    public enum ComputeMode {
        CPU, GPU, HYBRID
    }
    private ComputeMode currentMode = ComputeMode.CPU;

    // Cachés para TornadoVM
    private transient TornadoExecutionPlan gpuPlan;
    private transient FloatArray gpuMatrixCache;
    private transient FloatArray gpuVectorCache;
    private transient FloatArray gpuResultCache;
    private transient int cachedRows = -1;
    private transient int cachedCols = -1;

    // Arquitectura
    private final Map<String, Integer> wordToIndex;
    private final Map<Integer, String> indexToWord;
    private int vocabSize;

    private double[][] We;
    private double[][] Wf, Wi, Wo, Wg;
    private double[][] Uf, Ui, Uo, Ug;
    private double[] bf, bi, bo, bg;
    private double[][] Why;
    private double[] by;

    // ADAM
    private double[][] mWf, mWi, mWo, mWg, mUf, mUi, mUo, mUg, mWe, mWhy;
    private double[] mbf, mbi, mbo, mbg, mby;
    private double[][] vWf, vWi, vWo, vWg, vUf, vUi, vUo, vUg, vWe, vWhy;
    private double[] vbf, vbi, vbo, vbg, vby;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int adamStep = 0;
    private int currentEpoch = 0;

    // Hiperparámetros
    private final int embeddingSize;
    private final int hiddenSize;
    private double learningRate;
    private final Random rand;
    private static final int MAX_VOCAB_SIZE = 50000;
    private List<String> corpusWords;

    private final double gradientClipNorm = 5.0;

    // Sistema y Hardware
    private SystemInfo systemInfo;
    private transient ExecutorService threadPool;
    private int optimalBatchSize = 128;
    private int optimalThreads = 8;

    // Candado estático para evitar que los 22 hilos choquen en la GPU
    private static final Object gpuLock = new Object();

    public ImprovedLanguageModel() {
        this(64, 256, 0.002);
    }

    public ImprovedLanguageModel(int embeddingSize, int hiddenSize, double learningRate) {
        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
        this.wordToIndex = new HashMap<>();
        this.indexToWord = new HashMap<>();
        this.vocabSize = 0;
        this.rand = new Random(42);
        this.corpusWords = new ArrayList<>();
        this.systemInfo = new SystemInfo();
        optimizeForHardware();
    }

    // ============ CARGA DE CORPUS ============
    public void loadCorpus(String filePath) throws IOException {
        Map<String, Integer> charCounts = new HashMap<>();
        List<String> newCorpusChars = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] sentences = line.split("(?<=[.!?])\\s+");
                for (String sentence : sentences) {
                    sentence = sentence.trim();
                    if (sentence.isEmpty()) {
                        continue;
                    }

                    newCorpusChars.add("<START>");
                    for (int i = 0; i < sentence.length();) {
                        int cp = sentence.codePointAt(i);
                        String s = new String(Character.toChars(cp));
                        newCorpusChars.add(s);

                        if (!s.matches("[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9 .,;:!?¿¡\\n\\-]")) {
                            continue; // Ignora emojis, comillas raras, corchetes, etc.
                        }

                        charCounts.put(s, charCounts.getOrDefault(s, 0) + 1);
                        i += Character.charCount(cp);
                    }
                    newCorpusChars.add("<END>");
                }
            }
        }

        charCounts.putIfAbsent("<START>", 1000);
        charCounts.putIfAbsent("<END>", 1000);
        charCounts.putIfAbsent("<UNK>", 100);

        updateVocabulary(charCounts);
        corpusWords.addAll(newCorpusChars);
        initializeNewWeights();
    }

    private void updateVocabulary(Map<String, Integer> wordCounts) {
        List<Map.Entry<String, Integer>> sortedWords = wordCounts.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .filter(entry -> !wordToIndex.containsKey(entry.getKey()))
                .limit(MAX_VOCAB_SIZE - vocabSize)
                .collect(Collectors.toList());

        for (Map.Entry<String, Integer> entry : sortedWords) {
            wordToIndex.put(entry.getKey(), vocabSize);
            indexToWord.put(vocabSize, entry.getKey());
            vocabSize++;
        }
    }

    private void initializeNewWeights() {
        if (We == null) {
            initializeWeights();
            return;
        }
        int oldVocabSize = We.length;
        if (vocabSize > oldVocabSize) {
            double[][] newWe = new double[vocabSize][embeddingSize];
            double[][] newWhy = new double[vocabSize][hiddenSize];
            double[][] newMWe = new double[vocabSize][embeddingSize];
            double[][] newVWe = new double[vocabSize][embeddingSize];
            double[][] newMWhy = new double[vocabSize][hiddenSize];
            double[][] newVWhy = new double[vocabSize][hiddenSize];
            double[] newBy = new double[vocabSize];
            double[] newMby = new double[vocabSize];
            double[] newVby = new double[vocabSize];

            for (int i = 0; i < oldVocabSize; i++) {
                newWe[i] = We[i];
                newWhy[i] = Why[i];
                newMWe[i] = mWe[i];
                newVWe[i] = vWe[i];
                newMWhy[i] = mWhy[i];
                newVWhy[i] = vWhy[i];
            }
            System.arraycopy(by, 0, newBy, 0, by.length);
            System.arraycopy(mby, 0, newMby, 0, mby.length);
            System.arraycopy(vby, 0, newVby, 0, vby.length);

            double embScale = Math.sqrt(2.0 / embeddingSize);
            double hiddenScale = Math.sqrt(2.0 / hiddenSize);
            for (int i = oldVocabSize; i < vocabSize; i++) {
                newWe[i] = new double[embeddingSize];
                newWhy[i] = new double[hiddenSize];
                newMWe[i] = new double[embeddingSize];
                newVWe[i] = new double[embeddingSize];
                newMWhy[i] = new double[hiddenSize];
                newVWhy[i] = new double[hiddenSize];
                for (int j = 0; j < embeddingSize; j++) {
                    newWe[i][j] = rand.nextGaussian() * embScale;
                }
                for (int j = 0; j < hiddenSize; j++) {
                    newWhy[i][j] = rand.nextGaussian() * hiddenScale;
                }
            }

            We = newWe;
            Why = newWhy;
            mWe = newMWe;
            vWe = newVWe;
            mWhy = newMWhy;
            vWhy = newVWhy;
            by = newBy;
            mby = newMby;
            vby = newVby;
        }
    }

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
        mWe = new double[vocabSize][embeddingSize];
        vWe = new double[vocabSize][embeddingSize];
        mWhy = new double[vocabSize][hiddenSize];
        vWhy = new double[vocabSize][hiddenSize];
        mbf = new double[hiddenSize];
        vbf = new double[hiddenSize];
        mbi = new double[hiddenSize];
        vbi = new double[hiddenSize];
        mbo = new double[hiddenSize];
        vbo = new double[hiddenSize];
        mbg = new double[hiddenSize];
        vbg = new double[hiddenSize];
        mby = new double[vocabSize];
        vby = new double[vocabSize];
    }

    private double[][] initMatrix(int rows, int cols, double scale) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextGaussian() * scale;
            }
        }
        return matrix;
    }

    // ============ ENTRENAMIENTO ============
    public void train(int epochsToRun, int seqLength) {
        System.out.println("=== CONFIGURACION DE ENTRENAMIENTO ===");
        systemInfo.printSystemInfo();
        System.out.println("Modo de cómputo: " + currentMode);
        System.out.println("Batch size optimo: " + optimalBatchSize);
        System.out.println("Threads: " + optimalThreads);
        System.out.println("Vocab size: " + vocabSize);
        if (currentEpoch > 0) {
            System.out.println("🔄 Reanudando desde la época: " + currentEpoch);
        }
        System.out.println("==========================================\n");

        List<List<Integer>> sequences = prepareTrainingData(seqLength);
        System.out.println("Secuencias validas preparadas: " + sequences.size());

        int startEpoch = currentEpoch;
        int endEpoch = currentEpoch + epochsToRun;

        for (int epoch = startEpoch; epoch < endEpoch; epoch++) {
            long epochStart = System.currentTimeMillis();
            System.out.printf("\n🚀 EPOCA %d/%d (Total entrenadas: %d)\n", (epoch + 1), endEpoch, epoch + 1);

            double totalLoss = 0;
            int processedBatches = 0;
            Collections.shuffle(sequences, rand);

            for (int start = 0; start < sequences.size(); start += optimalBatchSize) {
                int end = Math.min(start + optimalBatchSize, sequences.size());
                List<List<Integer>> batch = sequences.subList(start, end);

                double batchLoss = processBatchParallel(batch);
                totalLoss += batchLoss;
                processedBatches++;

                if (processedBatches % 10 == 0 || processedBatches == (sequences.size() + optimalBatchSize - 1) / optimalBatchSize) {
                    double progress = (double) (start + batch.size()) / sequences.size() * 100;
                    double currentLoss = totalLoss / processedBatches;
                    double currentPerplexity = Math.exp(currentLoss);
                    System.out.printf("\r📊 Progreso: %.1f%% | Batches: %d/%d | Loss: %.4f | PPL: %.2f | Mem: %dGB",
                            progress, processedBatches, (sequences.size() + optimalBatchSize - 1) / optimalBatchSize,
                            currentLoss, currentPerplexity, systemInfo.getUsedMemoryGB());
                }
            }

            long epochTime = System.currentTimeMillis() - epochStart;
            double epochLoss = totalLoss / Math.max(1, processedBatches);
            System.out.printf("\n✅ Epoca completada en %.2fs | Loss: %.4f | Perplexity: %.2f\n",
                    epochTime / 1000.0, epochLoss, Math.exp(epochLoss));

            currentEpoch = epoch + 1; // Actualizar contador global
            saveCheckpoint("checkpoint_epoch_" + currentEpoch + ".ser");

            if (currentEpoch > 0 && currentEpoch % 3 == 0) {
                learningRate *= 0.90;
                System.out.printf("📉 Learning rate ajustado a: %.6f\n", learningRate);
            }
        }
        System.out.println("\n🎉 ¡Entrenamiento completado!");
        clean_old_checkpoints();
    }

    private double processBatchParallel(List<List<Integer>> batch) {
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }

        int numThreads = Math.min(optimalThreads, batch.size());
        int chunkSize = Math.max(1, batch.size() / numThreads);
        List<Future<Gradients>> futures = new ArrayList<>();

        for (int i = 0; i < batch.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, batch.size());
            List<List<Integer>> chunk = batch.subList(i, end);
            futures.add(threadPool.submit(() -> {
                Gradients g = new Gradients();
                for (List<Integer> seq : chunk) {
                    g.loss += trainSequence(seq, g); // Llamada al método restaurado
                }
                return g;
            }));
        }

        Gradients totalGrad = new Gradients();
        for (Future<Gradients> f : futures) {
            try {
                totalGrad.add(f.get());
            } catch (Exception e) {
                System.err.println("Error en hilo: " + e.getMessage());
            }
        }

        totalGrad.scale(1.0 / batch.size());
        applyGradientClip(totalGrad);
        adamStep++;
        applyAdam(totalGrad);

        return totalGrad.loss / batch.size();
    }

    // ============ LÓGICA DEL LSTM (FORWARD Y BACKWARD) ============
    private double trainSequence(List<Integer> sequence, Gradients grad) {
        int seqLen = sequence.size() - 1;
        double[][] h = new double[seqLen][hiddenSize];
        double[][] c = new double[seqLen][hiddenSize];
        double[][] f = new double[seqLen][hiddenSize];
        double[][] i = new double[seqLen][hiddenSize];
        double[][] o = new double[seqLen][hiddenSize];
        double[][] g = new double[seqLen][hiddenSize];
        double[][] tanhC = new double[seqLen][hiddenSize];
        double[][] x = new double[seqLen][embeddingSize];

        double[] prevH = new double[hiddenSize];
        double[] prevC = new double[hiddenSize];
        double loss = 0;

        // Forward
        for (int t = 0; t < seqLen; t++) {
            x[t] = getEmbedding(sequence.get(t));

            matVecMul(f[t], Wf, x[t]);
            matVecMulAdd(f[t], Uf, prevH);
            addInPlace(f[t], bf);
            sigmoidInPlace(f[t]);
            matVecMul(i[t], Wi, x[t]);
            matVecMulAdd(i[t], Ui, prevH);
            addInPlace(i[t], bi);
            sigmoidInPlace(i[t]);
            matVecMul(o[t], Wo, x[t]);
            matVecMulAdd(o[t], Uo, prevH);
            addInPlace(o[t], bo);
            sigmoidInPlace(o[t]);
            matVecMul(g[t], Wg, x[t]);
            matVecMulAdd(g[t], Ug, prevH);
            addInPlace(g[t], bg);
            tanhInPlace(g[t]);

            for (int j = 0; j < hiddenSize; j++) {
                c[t][j] = f[t][j] * prevC[j] + i[t][j] * g[t][j];
                tanhC[t][j] = Math.tanh(c[t][j]);
                h[t][j] = o[t][j] * tanhC[t][j];
            }

            double[] logits = new double[vocabSize];
            matVecMul(logits, Why, h[t]);
            addInPlace(logits, by);
            softmaxInPlace(logits);

            int target = sequence.get(t + 1);
            loss += -Math.log(Math.max(logits[target], 1e-10));

            prevH = h[t];
            prevC = c[t];
        }

        // Backward
        double[] dhNext = new double[hiddenSize], dcNext = new double[hiddenSize];
        double[] dh = new double[hiddenSize], dc = new double[hiddenSize];
        double[] df = new double[hiddenSize], di = new double[hiddenSize];
        double[] dob = new double[hiddenSize], dg = new double[hiddenSize];
        double[] dx = new double[embeddingSize];
        double[] dOutput = new double[vocabSize], logitsCache = new double[vocabSize];

        for (int t = seqLen - 1; t >= 0; t--) {
            int target = sequence.get(t + 1);
            int tokIdx = sequence.get(t);

            matVecMul(logitsCache, Why, h[t]);
            addInPlace(logitsCache, by);
            softmaxInPlace(logitsCache);

            Arrays.fill(dOutput, 0);
            dOutput[target] = -1.0;
            for (int k = 0; k < vocabSize; k++) {
                dOutput[k] += logitsCache[k];
            }

            for (int k = 0; k < vocabSize; k++) {
                for (int j = 0; j < hiddenSize; j++) {
                    grad.dWhy[k][j] += dOutput[k] * h[t][j];
                }
                grad.dby[k] += dOutput[k];
            }

            Arrays.fill(dh, 0);
            for (int k = 0; k < vocabSize; k++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dh[j] += dOutput[k] * Why[k][j];
                }
            }
            addInPlace(dh, dhNext);

            for (int j = 0; j < hiddenSize; j++) {
                dc[j] = dh[j] * o[t][j] * (1.0 - tanhC[t][j] * tanhC[t][j]) + dcNext[j];
                df[j] = dc[j] * (t > 0 ? c[t - 1][j] : 0) * f[t][j] * (1.0 - f[t][j]);
                di[j] = dc[j] * g[t][j] * i[t][j] * (1.0 - i[t][j]);
                dob[j] = dh[j] * tanhC[t][j] * o[t][j] * (1.0 - o[t][j]);
                dg[j] = dc[j] * i[t][j] * (1.0 - g[t][j] * g[t][j]);
            }

            Arrays.fill(dx, 0);
            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < embeddingSize; k++) {
                    grad.dWf[j][k] += df[j] * x[t][k];
                    grad.dWi[j][k] += di[j] * x[t][k];
                    grad.dWo[j][k] += dob[j] * x[t][k];
                    grad.dWg[j][k] += dg[j] * x[t][k];
                    dx[k] += df[j] * Wf[j][k] + di[j] * Wi[j][k] + dob[j] * Wo[j][k] + dg[j] * Wg[j][k];
                }
                grad.dbf[j] += df[j];
                grad.dbi[j] += di[j];
                grad.dbo[j] += dob[j];
                grad.dbg[j] += dg[j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < hiddenSize; k++) {
                    double prevH_k = (t > 0) ? h[t - 1][k] : 0.0;
                    grad.dUf[j][k] += df[j] * prevH_k;
                    grad.dUi[j][k] += di[j] * prevH_k;
                    grad.dUo[j][k] += dob[j] * prevH_k;
                    grad.dUg[j][k] += dg[j] * prevH_k;
                }
            }

            for (int k = 0; k < embeddingSize; k++) {
                grad.dWe[tokIdx][k] += dx[k];
            }

            Arrays.fill(dhNext, 0);
            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < hiddenSize; k++) {
                    dhNext[k] += df[j] * Uf[j][k] + di[j] * Ui[j][k] + dob[j] * Uo[j][k] + dg[j] * Ug[j][k];
                }
            }
            for (int j = 0; j < hiddenSize; j++) {
                dcNext[j] = dc[j] * f[t][j];
            }
        }
        return loss / seqLen;
    }

    // ============ KERNEL PARA GPU ============
    public static void gpuMatVecKernel(FloatArray matrix, FloatArray vector, FloatArray result, int rows, int cols) {
        for (@Parallel int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int rowOffset = i * cols;
            for (int j = 0; j < cols; j++) {
                sum += matrix.get(rowOffset + j) * vector.get(j);
            }
            result.set(i, sum);
        }
    }

    // ============ MATEMÁTICAS IN-PLACE ============
    private void matVecMul(double[] y, double[][] M, double[] x) {
        if (currentMode != ComputeMode.CPU) {
            dispatchGPU(y, M, x);
        } else {
            matVecMul_CPU(y, M, x);
        }
    }

    private void matVecMul_CPU(double[] y, double[][] M, double[] x) {
        for (int i = 0; i < M.length; i++) {
            double sum = 0;
            double[] row = M[i];
            for (int j = 0; j < x.length; j++) {
                sum += row[j] * x[j];
            }
            y[i] = sum;
        }
    }

    private void matVecMulAdd(double[] y, double[][] M, double[] x) {
        if (currentMode != ComputeMode.CPU) {
            double[] temp = new double[y.length];
            dispatchGPU(temp, M, x);
            for (int i = 0; i < y.length; i++) {
                y[i] += temp[i];
            }
        } else {
            for (int i = 0; i < M.length; i++) {
                double sum = 0;
                double[] row = M[i];
                for (int j = 0; j < x.length; j++) {
                    sum += row[j] * x[j];
                }
                y[i] += sum;
            }
        }
    }

    private void dispatchGPU(double[] y, double[][] M, double[] x) {
        synchronized (gpuLock) {
            int rows = M.length, cols = x.length;

            // Si cambia el tamaño, cerramos el plan anterior para liberar VRAM
            if (gpuPlan == null || cachedRows != rows || cachedCols != cols) {
                if (gpuPlan != null) {
                    try {
                        gpuPlan.close();
                    } catch (Exception e) {
                    }
                }

                gpuMatrixCache = new FloatArray(rows * cols);
                gpuVectorCache = new FloatArray(cols);
                gpuResultCache = new FloatArray(rows);

                TaskGraph taskGraph = new TaskGraph("matVec")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, gpuMatrixCache, gpuVectorCache)
                        .task("multiply", ImprovedLanguageModel::gpuMatVecKernel, gpuMatrixCache, gpuVectorCache, gpuResultCache, rows, cols)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, gpuResultCache);

                gpuPlan = new TornadoExecutionPlan(taskGraph.snapshot());
                cachedRows = rows;
                cachedCols = cols;
            }

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    gpuMatrixCache.set(i * cols + j, (float) M[i][j]);
                }
            }
            for (int j = 0; j < cols; j++) {
                gpuVectorCache.set(j, (float) x[j]);
            }

            gpuPlan.execute();

            for (int i = 0; i < rows; i++) {
                y[i] = gpuResultCache.get(i);
            }
        }
    }

    private void addInPlace(double[] y, double[] x) {
        for (int i = 0; i < y.length; i++) {
            y[i] += x[i];
        }
    }

    private void sigmoidInPlace(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        }
    }

    private void tanhInPlace(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.tanh(x[i]);
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
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++) {
            x[i] /= sum;
        }
    }

    private double[] getEmbedding(int idx) {
        if (idx >= 0 && idx < vocabSize) {
            return We[idx];
        }
        Integer unk = wordToIndex.get("<UNK>");
        return We[unk != null ? unk : 0];
    }

    // ============ GRADIENTES Y ADAM ============
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
        for (double[] r : g.dWf) {
            for (double v : r) {
                n += v * v;
            }
        }
        for (double[] r : g.dUf) {
            for (double v : r) {
                n += v * v;
            }
        }
        for (double[] r : g.dWe) {
            for (double v : r) {
                n += v * v;
            }
        }
        for (double[] r : g.dWhy) {
            for (double v : r) {
                n += v * v;
            }
        }
        double norm = Math.sqrt(n);
        if (norm > gradientClipNorm) {
            g.scale(gradientClipNorm / norm);
        }
    }

    private void applyAdam(Gradients g) {
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
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * dW[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
                W[i][j] -= learningRate * (m[i][j] / c1) / (Math.sqrt(v[i][j] / c2) + epsilon);
            }
        }
    }

    private void adamUpdate(double[] b, double[] db, double[] m, double[] v) {
        double c1 = 1.0 - Math.pow(beta1, adamStep);
        double c2 = 1.0 - Math.pow(beta2, adamStep);
        for (int i = 0; i < b.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * db[i];
            v[i] = beta2 * v[i] + (1 - beta2) * db[i] * db[i];
            b[i] -= learningRate * (m[i] / c1) / (Math.sqrt(v[i] / c2) + epsilon);
        }
    }

    // ============ GENERACIÓN ============
    public String generateText(String seed, int maxLength, double temperature) {
        ComputeMode originalMode = currentMode;
        currentMode = ComputeMode.CPU; // Forzar CPU para evitar el cuello de botella de PCIe

        StringBuilder result = new StringBuilder();
        double[] h = new double[hiddenSize], c = new double[hiddenSize];
        double[] f = new double[hiddenSize], i = new double[hiddenSize], o = new double[hiddenSize], g = new double[hiddenSize], tanhC = new double[hiddenSize];

        // Inyectar <START> para inicializar el contexto del modelo
        String startToken = "<START>";
        if (wordToIndex.containsKey(startToken)) {
            double[] x = getEmbedding(wordToIndex.get(startToken));
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }
        }

        // Procesar el seed (prompt) letra por letra
        for (int idx = 0; idx < seed.length();) {
            int cp = seed.codePointAt(idx);
            String s = new String(Character.toChars(cp));
            idx += Character.charCount(cp);
            if (!wordToIndex.containsKey(s)) {
                if (!wordToIndex.containsKey("<UNK>")) {
                    continue;
                }
                s = "<UNK>";
            }
            double[] x = getEmbedding(wordToIndex.get(s));
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }
            result.append(s);
        }

        // Generar el texto nuevo - uso codePointBefore para evitar errores con emojis o tildes compuestas
        String lastChar = seed.isEmpty() ? "<START>" : new String(Character.toChars(seed.codePointBefore(seed.length())));

        for (int step = 0; step < maxLength; step++) {
            double[] logits = new double[vocabSize];
            matVecMul(logits, Why, h);
            addInPlace(logits, by);

            for (int j = 0; j < logits.length; j++) {
                logits[j] /= Math.max(1e-6, temperature);
            }
            softmaxInPlace(logits);

            int nextIdx = 0;
            double r = rand.nextDouble(), sum = 0;
            for (int j = 0; j < logits.length; j++) {
                sum += logits[j];
                if (r <= sum) {
                    nextIdx = j;
                    break;
                }
            }
            String next = indexToWord.get(nextIdx);

            // ACTUALIZAR EL ESTADO DEL LSTM 
            // (Incluso si es <UNK>, para que la red no se quede atascada en el mismo estado)
            double[] x = getEmbedding(nextIdx);
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }

            // FILTRAR TOKENS ESPECIALES ANTES DE IMPRIMIR
            if ("<START>".equals(next) || "<UNK>".equals(next)) {
                continue; // Ignorar e ir al siguiente paso, pero el LSTM ya avanzó
            }

            if ("<END>".equals(next)) {
                // Si el modelo predice <END> tras un signo de puntuación, forzamos un espacio
                if (step < 5 && (lastChar.equals("?") || lastChar.equals("!") || lastChar.equals("."))) {
                    next = " ";
                } else {
                    break; // Terminar normalmente
                }
            }

            result.append(next);
            lastChar = next;
        }

        currentMode = originalMode;
        return result.toString();
    }

    public String generateText_old(String seed, int maxLength, double temperature) {
        ComputeMode originalMode = currentMode;
        currentMode = ComputeMode.CPU;
        // -----------------------------

        StringBuilder result = new StringBuilder();
        double[] h = new double[hiddenSize], c = new double[hiddenSize];
        double[] f = new double[hiddenSize], i = new double[hiddenSize], o = new double[hiddenSize], g = new double[hiddenSize], tanhC = new double[hiddenSize];
        String lastChar = null;

        for (int idx = 0; idx < seed.length();) {
            int cp = seed.codePointAt(idx);
            String s = new String(Character.toChars(cp));
            idx += Character.charCount(cp);
            if (!wordToIndex.containsKey(s)) {
                if (!wordToIndex.containsKey("<UNK>")) {
                    continue;
                }
                s = "<UNK>";
            }
            double[] x = getEmbedding(wordToIndex.get(s));
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }
            result.append(s);
            lastChar = s;
        }

        if (lastChar == null) {
            lastChar = "<START>";
            double[] x = getEmbedding(wordToIndex.get(lastChar));
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }
        }

        for (int step = 0; step < maxLength; step++) {
            double[] logits = new double[vocabSize];
            matVecMul(logits, Why, h);
            addInPlace(logits, by);
            for (int j = 0; j < logits.length; j++) {
                logits[j] /= Math.max(1e-6, temperature);
            }
            softmaxInPlace(logits);

            int nextIdx = 0;
            double r = rand.nextDouble(), sum = 0;
            for (int j = 0; j < logits.length; j++) {
                sum += logits[j];
                if (r <= sum) {
                    nextIdx = j;
                    break;
                }
            }
            String next = indexToWord.get(nextIdx);
            if ("<END>".equals(next)) {
                break;
            }
            result.append(next);

            double[] x = getEmbedding(nextIdx);
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
                tanhC[j] = Math.tanh(c[j]);
                h[j] = o[j] * tanhC[j];
            }
        }

        currentMode = originalMode; // Restaurar el modo que el usuario había elegido
        return result.toString();
    }

    private List<List<Integer>> prepareTrainingData(int seqLength) {
        List<List<Integer>> seqs = new ArrayList<>();
        int stride = Math.max(1, seqLength / 2);
        for (int start = 0; start + seqLength + 1 <= corpusWords.size(); start += stride) {
            List<Integer> s = new ArrayList<>(seqLength + 1);
            for (int i = 0; i <= seqLength; i++) {
                String w = corpusWords.get(start + i);
                s.add(wordToIndex.getOrDefault(w, wordToIndex.get("<UNK>")));
            }
            seqs.add(s);
        }
        return seqs;
    }

    private void clean_old_checkpoints() {
        // Mantener solo los últimos 3 checkpoints para ahorrar espacio
        int minKeep = Math.max(1, currentEpoch - 3);
        for (int i = 1; i < minKeep; i++) {
            File ftmp = new File("checkpoint_epoch_" + i + ".ser");
            if (ftmp.exists()) {
                ftmp.delete();
            }
        }
    }

    private void saveCheckpoint(String filePath) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
            System.out.println("✅ Checkpoint guardado en: " + filePath);
        } catch (IOException e) {
            System.err.println("Error al guardar el checkpoint: " + e.getMessage());
        }
    }

    private void optimizeForHardware() {
        long availableMemoryMB = systemInfo.getAvailableMemoryMB();
        int cores = systemInfo.getAvailableCores();
        System.out.println("🔧 OPTIMIZACION AUTOMATICA DE HARDWARE:");
        systemInfo.printSystemInfo();
        optimalThreads = Math.min(cores - 2, 24);
        optimalThreads = Math.max(optimalThreads, 2);
        int maxBatchByMemory = (int) Math.min(availableMemoryMB / 2, 512);
        optimalBatchSize = Math.min(maxBatchByMemory, 128);
        optimalBatchSize = Math.max(optimalBatchSize, 16);
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }
        System.out.printf("✅ Configuracion optimizada - Threads: %d, Batch Size: %d\n\n", optimalThreads, optimalBatchSize);
    }

    private void cleanup() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
            try {
                if (!threadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                    threadPool.shutdownNow();
                }
            } catch (InterruptedException e) {
                threadPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        if (gpuPlan != null) {
            try {
                gpuPlan.close();
            } catch (Exception e) {
            }
            gpuPlan = null;
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        threadPool = Executors.newFixedThreadPool(optimalThreads);
    }

    // ============ CARGA DE CORPUS ALPACA ============
    public void loadAlpacaCorpus(String filePath) throws IOException {
        Map<String, Integer> charCounts = new HashMap<>();
        List<String> newCorpusChars = new ArrayList<>();

        System.out.println("⏳ Parseando archivo Alpaca...");

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            int count = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                // Parseo manual de JSON (asumiendo el formato exacto de tu archivo)
                String question = extractJsonField(line, "question");
                String answer = extractJsonField(line, "answer");

                if (question != null && answer != null) {
                    // Limpiamos el texto para quitar el "Usuario: ... \n\nAsistente:" redundante
                    question = question.replace("Usuario:", "").replace("Asistente:", "").trim();

                    // Creamos la secuencia de entrenamiento
                    String textBlock = "Usuario: " + question + "\nAsistente: " + answer;

                    newCorpusChars.add("<START>");
                    for (int i = 0; i < textBlock.length();) {
                        int cp = textBlock.codePointAt(i);
                        String s = new String(Character.toChars(cp));
                        newCorpusChars.add(s);
                        charCounts.put(s, charCounts.getOrDefault(s, 0) + 1);
                        i += Character.charCount(cp);
                    }
                    newCorpusChars.add("<END>");
                    count++;
                }
            }
            System.out.println("✅ Parseadas " + count + " conversaciones.");
        }

        charCounts.putIfAbsent("<START>", 1000);
        charCounts.putIfAbsent("<END>", 1000);
        charCounts.putIfAbsent("<UNK>", 100);

        updateVocabulary(charCounts);
        corpusWords.addAll(newCorpusChars);
        initializeNewWeights();
    }

    // Método auxiliar para extraer campos de un JSON simple sin librerías
    private String extractJsonField(String json, String fieldName) {
        String key = "\"" + fieldName + "\": \"";
        int start = json.indexOf(key);
        if (start == -1) {
            return null;
        }
        start += key.length();

        // Buscar el cierre de la comilla, manejando comillas escapadas
        int end = start;
        while (end < json.length()) {
            if (json.charAt(end) == '\"' && json.charAt(end - 1) != '\\') {
                break;
            }
            end++;
        }

        String raw = json.substring(start, end);
        // Limpiar caracteres de escape comunes
        return raw.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    public void saveModel(String filePath) {
        cleanup();
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
            System.out.println("Modelo guardado en: " + filePath);
        } catch (IOException e) {
            System.err.println("Error al guardar el modelo: " + e.getMessage());
        }
    }

    public static ImprovedLanguageModel loadModel(String filePath) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            return (ImprovedLanguageModel) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error al cargar el modelo: " + e.getMessage());
            return null;
        }
    }

    // ============ MENÚ INTERACTIVO ============
    public void startInteractiveMode() throws IOException {
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== MODELO DE LENGUAJE MEJORADO CON OPTIMIZACION DE HARDWARE ===");
        System.out.println("Comandos disponibles:");
        System.out.println("  EXIT - Salir");
        System.out.println("  LOAD_TEXT - Cargar corpus de texto");
        System.out.println("  LOAD_ALPACA - Cargar corpus conversacional (.jsonl)");
        System.out.println("  MODE - Cambiar modo de cómputo (CPU / GPU / HYBRID)");
        System.out.println("  TRAIN - Entrenar modelo");
        System.out.println("  SAVE - Guardar modelo");
        System.out.println("  LOAD - Cargar modelo");
        System.out.println("  STATUS - Ver estado del sistema");
        System.out.println("  OPTIMIZE - Reoptimizar para hardware");

        while (true) {
            System.out.print("\n> ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("EXIT")) {
                cleanup();
                break;
            } else if (input.equalsIgnoreCase("STATUS")) {
                systemInfo.printSystemInfo();
                System.out.printf("Configuracion actual - Threads: %d, Batch Size: %d, Modo: %s\n",
                        optimalThreads, optimalBatchSize, currentMode);
                if (vocabSize > 0) {
                    System.out.printf("Modelo - Vocabulario: %d terminos, Corpus: %d caracteres\n", vocabSize, corpusWords.size());
                }
            } else if (input.equalsIgnoreCase("OPTIMIZE")) {
                optimizeForHardware();
            } else if (input.equalsIgnoreCase("MODE")) {
                System.out.println("Seleccione el modo de cómputo:");
                System.out.println("  1. CPU (solo procesador, como antes)");
                System.out.println("  2. GPU (acelerado por CUDA/OpenCL)");
                System.out.println("  3. HYBRID (CPU para lógica, GPU para multiplicaciones)");
                System.out.print("Opción [1]: ");
                String opt = scanner.nextLine().trim();
                if (opt.isEmpty() || opt.equals("1")) {
                    currentMode = ComputeMode.CPU;
                } else if (opt.equals("2")) {
                    currentMode = ComputeMode.GPU;
                } else if (opt.equals("3")) {
                    currentMode = ComputeMode.HYBRID;
                }
                System.out.println("✅ Modo cambiado a: " + currentMode);
            } else if (input.equalsIgnoreCase("LOAD_TEXT")) {
                System.out.print("📁 Archivo: (por defecto: src/javallmsimple/corpus.txt) ");
                String path = scanner.nextLine().trim();
                if (path.isEmpty()) {
                    path = "src/javallmsimple/corpus.txt";
                }
                try {
                    System.out.println("🔄 Cargando corpus...");
                    long startTime = System.currentTimeMillis();
                    loadCorpus(path);
                    System.out.printf("✅ Corpus cargado en %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0);
                    System.out.printf("📊 Estadisticas: %d caracteres, vocabulario de %d terminos\n", corpusWords.size(), vocabSize);
                } catch (IOException e) {
                    System.out.println("❌ Error: " + e.getMessage());
                }
            } else if (input.equalsIgnoreCase("LOAD_ALPACA")) {
                System.out.print("📁 Archivo Alpaca (.jsonl): ");
                String path = scanner.nextLine().trim();
                try {
                    System.out.println("🔄 Cargando corpus Alpaca...");
                    long startTime = System.currentTimeMillis();
                    loadAlpacaCorpus(path);
                    System.out.printf("✅ Corpus cargado en %.2fs\n", (System.currentTimeMillis() - startTime) / 1000.0);
                    System.out.printf("📊 Estadisticas: %d caracteres, vocabulario de %d terminos\n", corpusWords.size(), vocabSize);
                } catch (IOException e) {
                    System.out.println("❌ Error: " + e.getMessage());
                }
            } else if (input.equalsIgnoreCase("TRAIN")) {
                if (vocabSize == 0) {
                    System.out.println("⚠️  Primero carga un corpus con LOAD_TEXT");
                    continue;
                }

                if (currentEpoch > 0) {
                    System.out.println("ℹ️  Este modelo ya tiene " + currentEpoch + " épocas entrenadas.");
                }

                System.out.print("🎯 Epocas ADICIONALES a entrenar (recomendado 5-10): ");
                String epochStr = scanner.nextLine().trim();
                int epochs = epochStr.isEmpty() ? 5 : Integer.parseInt(epochStr);

                System.out.print("📏 Longitud de secuencia (recomendado 128): ");
                String seqStr = scanner.nextLine().trim();
                int seqLength = seqStr.isEmpty() ? 128 : Integer.parseInt(seqStr);

                long trainStart = System.currentTimeMillis();
                train(epochs, seqLength);
                System.out.printf("\n🏁 Entrenamiento completado en %.2f minutos\n", (System.currentTimeMillis() - trainStart) / 60000.0);
            } else if (input.equalsIgnoreCase("SAVE")) {
                System.out.print("💾 Nombre del archivo (default: train_model.ser): ");
                String filename = scanner.nextLine().trim();
                if (filename.isEmpty()) {
                    filename = "train_model.ser";
                }
                saveModel(filename);
            } else if (input.equalsIgnoreCase("LOAD")) {
                System.out.print("📂 Archivo a cargar: ");
                String filename = scanner.nextLine().trim();
                ImprovedLanguageModel loaded = loadModel(filename);
                if (loaded != null) {
                    System.out.println("✅ Modelo cargado exitosamente");
                    loaded.startInteractiveMode();
                    return;
                }
            } else if (!input.isEmpty()) {
                if (vocabSize == 0) {
                    System.out.println("⚠️  Modelo no entrenado. Usa LOAD_TEXT y TRAIN primero.");
                    continue;
                }
                System.out.print("📝 Longitud del texto (default 200): ");
                String lenStr = scanner.nextLine().trim();
                int length = lenStr.isEmpty() ? 200 : Integer.parseInt(lenStr);
                System.out.print("🌡️  Temperatura (0.1=conservador, 1.5=creativo, default 0.7): ");
                String tempStr = scanner.nextLine().trim();
                double temp = tempStr.isEmpty() ? 0.7 : Double.parseDouble(tempStr);
                System.out.println("🤖 Generando texto...");
                long genStart = System.currentTimeMillis();
                String response = generateText(input, length, temp);
                System.out.printf("\n📜 Generado en %dms:\n", System.currentTimeMillis() - genStart);
                System.out.println("\"" + response + "\"");
            }
        }
        scanner.close();
    }

    public static void main(String[] args) {
        System.out.println("🚀 Iniciando Modelo de Lenguaje Optimizado...\n");
        ImprovedLanguageModel model = new ImprovedLanguageModel();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n🔄 Cerrando modelo y liberando recursos...");
            model.cleanup();
        }));
        try {
            model.startInteractiveMode();
        } catch (IOException e) {
            System.err.println("❌ Error: " + e.getMessage());
        } finally {
            model.cleanup();
        }
    }

    private static class SystemInfo implements Serializable {

        private static final long serialVersionUID = 1L;

        public SystemInfo() {
        }

        private void printSystemInfo() {
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory() / (1024 * 1024);
            long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
            Properties p = System.getProperties();
            System.out.println("OS: " + p.getProperty("os.name"));
            System.out.println("OS Arch: " + p.getProperty("os.arch"));
            System.out.println("RAM Max: " + maxMemory + "MB");
            System.out.println("RAM Used: " + usedMemory + "MB");
            System.out.println("Available Cores: " + runtime.availableProcessors());
        }

        public int getUsedMemoryGB() {
            Runtime runtime = Runtime.getRuntime();
            long usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory();
            return (int) (usedMemoryBytes / (1024L * 1024L * 1024L));
        }

        private long getAvailableMemoryMB() {
            return Runtime.getRuntime().maxMemory() / (1024 * 1024);
        }

        private int getAvailableCores() {
            return Runtime.getRuntime().availableProcessors();
        }
    }
}
