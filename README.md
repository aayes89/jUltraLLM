# jUltraLLM
An LLM written in Java, using `alpaca_chat.jsonl` as a training corpus. (Spanish version on master branch)

# Motivation
It is often claimed that an efficient language model can only be built using Python, and that Python’s libraries are the best for such tasks. With this project, I aim to demonstrate that this is possible in other languages ​​as well, and that training times even when using only a CPU are not as prohibitively long as commonly believed. The result, in this specific case, is an 8MB model trained on a 50MB Spanish language `alpaca_chat` corpus. By applying certain optimizations, the model achieves fast response times with a high degree of effectiveness. It is certainly not a groundbreaking model, nor is it intended for intensive, everyday use; there is still much work to be done.

# How to use
* LOAD_ALPACA – Load a model in JSONL format (the repository includes `alpaca_es_chat.jsonl`).
* LOAD_TEXT – Load a text-based corpus (`.txt` file) of your choice.
* TRAIN – Begin training the model.
* SAVE – Save the trained model (this function is triggered automatically at the end of each epoch).
* LOAD – Load a previously trained model.
* STATUS – Display current information about the loaded model.
* EXIT – Exit the application.

# To-do list
* Add support for reasoning (chains of thought).
* Optimize the code to increase training speed without compromising quality.
* Improve UTF-8 support to correctly handle special characters used in Spanish.
* Calculate the estimated training time before starting, based on the selected mode (CPU, GPU, or hybrid).
* Implement an intelligent approach to using the Tornado library for execution on NVIDIA GPUs, ensuring no bottlenecks are created.
