# jUltraLLM
LLM escrito en Java usando alpaca_chat.jsonl como corpus de entrenamiento.

# Como usar
* LOAD_ALPACA - cargar el modelo en formato jsonl que se indique (alpaca_es_chat.jsonl incluido en el repositorio)
* LOAD_TEXT - cargar un corpus en formato de texto (.txt) de su preferencia
* TRAIN - inicia el entrenamiento del modelo
* SAVE - guarda el modelo entrenado (función implementada automáticamente en cada época)
* LOAD - cargar un modelo entrenado
* STATUS - muestra información actual del modelo cargado
* EXIT - Salir de la ejecución

# Demostración
```
=== UltraLLMK (Karpathy Char-RNN Style) ===
Comandos: LOAD_ALPACA, LOAD_TEXT, TRAIN, SAVE, LOAD, STATUS, EXIT

> load
Archivo: model.ser
? Modelo cargado en 0.18s
=== UltraLLMK (Karpathy Char-RNN Style) ===
Comandos: LOAD_ALPACA, LOAD_TEXT, TRAIN, SAVE, LOAD, STATUS, EXIT

> load_alpaca
Archivo .jsonl: alpaca_es_chat.jsonl
? Cargando corpus Alpaca completo y construyendo �ndice Invertido...
Conversaciones le�das: 49969
? Corpus Alpaca: 44549043 caracteres. Vocabulario: 531
?? Expandiendo matrices para nuevos caracteres (Conservando aprendizaje)...

> Nombre tres países de europa

? [Respuesta recuperada de la BD (Score: 4.57)]:
? Pregunta en BD: "que pais tiene la cima mas alta de europa"
? Respuesta: La cima m�s alta de Europa es el Monte Elbrus, que se encuentra en Rusia.

> Nombre tres países europeos

? [Respuesta recuperada de la BD (Score: 7.74)]:
? Pregunta en BD: "nombre un pais europeo"
? Respuesta: Un pa�s europeo es Francia.

> Sugiere un peinado adecuado para una cara ovalada

? [Respuesta recuperada de la BD (Score: 14.57)]:
? Pregunta en BD: "sugiere un peinado adecuado para una cara ovalada"
? Respuesta: Para personas con una forma de cara ovalada, muchos peinados pueden funcionar muy bien. Un flequillo de lado o en capas son ideales ya que pueden suavizar la forma de la cara mientras agregan dimensi�n. Los cortes de pelo bob, cortes de pelo a la altura del hombro y cortes de pelo largos en capas con ondas de aspecto natural tambi�n son excelentes opciones. Si quieres llevarlo recogido, prueba una cola de caballo baja o alta, un mo�o suelto y otros peinados que acent�en los p�mulos. Una raya lateral profunda o una raya en el medio tambi�n pueden acentuar tus caracter�sticas. En general, la clave es mantener las capas suaves y evitar agregar cualquier volumen a los lados que pueda hacer que la cara parezca m�s ancha.

> status
Vocab=531 | Corpus=44549043 | Batch=128 | Epochs=15
Best val_loss=1.1063

> 
```

# Licencia
MIT
