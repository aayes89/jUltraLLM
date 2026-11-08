# jUltraLLM
LLM escrito en Java usando alpaca_chat.jsonl como corpus de entrenamiento.

# Motivación
Siempre se comunica que sólo en Python es posible generar un modelo de lenguaje eficiente y que sus librerías son las mejores para este tipo de actividades. Con este proyecto pretendo demostrar que es posible en otros lenguajes y que los tiempos de entrenamiento usando sólo CPU no son descomunales como se cree. El resultado de este en particular, es un modelo de 8MB entrenado con un corpus de alpaca_chat en español de 50MB, aplicando algunas optimizaciones que facilitan la velocidad de respuesta con una alta tasa de efectividad. Ciertamente no es un modelo impresionante o para uso doméstico intensivo y aun queda mucho trabajo por hacer.

# Como usar
* LOAD_ALPACA - cargar el modelo en formato jsonl que se indique (alpaca_es_chat.jsonl incluido en el repositorio)
* LOAD_TEXT - cargar un corpus en formato de texto (.txt) de su preferencia
* TRAIN - inicia el entrenamiento del modelo
* SAVE - guarda el modelo entrenado (función implementada automáticamente en cada época)
* LOAD - cargar un modelo entrenado
* STATUS - muestra información actual del modelo cargado
* EXIT - Salir de la ejecución

# Pendientes
* añadir soporte para razonamiento (cadenas de pensamiento)
* optimizar código para aumentar velocidad de entrenamiento sin disminuir calidad
* corregir el soporte de UTF-8 para admitir caracteres especiales en español
* calcular tiempo estimado de entrenamiento antes de iniciarlo según selección de modo (CPU, GPU, híbrido)
* implementar uso inteligente de librería Tornado para ejecución en GPU NVIDIA sin crear cuello de botella

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
```
MIT License

Copyright (c) 2026 Allan (Slam)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
