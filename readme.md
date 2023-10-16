# Summary

This project was created to keep the progress of using huggingface transformers course

The [question_answering.ipynb](https://github.com/daniakaguict/transformers-course/blob/main/question_answering.ipynb) contains the tutorial from this [link](https://huggingface.co/learn/nlp-course/chapter7/7?fw=tf)

Reference: https://huggingface.co/learn/nlp-course/chapter1/1

# Setup
Create Python Environment of type conda

# Transformers

## Pros:

* Easy to use: The Hugging Face Transformers API is easy to use, even for beginners. It provides a simple and intuitive interface for loading and using pre-trained models.

* Comprehensive: The Hugging Face Transformers API provides a wide range of pre-trained models for a variety of tasks, including natural language processing, computer vision, and machine translation.

* Flexible: The Hugging Face Transformers API is flexible and can be used to fine-tune pre-trained models on new tasks.

* Well-maintained: The Hugging Face Transformers API is well-maintained and actively developed.
Community support: The Hugging Face Transformers API has a large and active community of users and contributors.

## Cons:

* Large models: The pre-trained models available in the Hugging Face Transformers API are often very large, which can make them slow to load and train.

* Computational resources: Training and using pre-trained models can require a lot of computational resources, such as GPUs.

* Complexity: The Hugging Face Transformers API can be complex, especially for beginners. You get to learn Keras as well as transformers.

## Glossary

* **Architecture:** The skeleton of the model, defining each layer and operation.
* **Checkpoints:** The weights that will be loaded in a given architecture.
* **Model:** An umbrella term for both architecture and checkpoints. This course will specify architecture or checkpoint when it matters to reduce ambiguity.
* **Logits:** The unnormalized outputs of a neural network. They are typically in the form of a vector of numbers, where each number represents the probability of the network predicting a particular class. Logits are often used in classification tasks, where the goal is to train the network to predict one of a number of possible classes. For example, a neural network might be trained to predict the type of animal in a picture, or the sentiment of a tweet. To convert logits into probabilities, the logits are typically passed through a softmax function. The softmax function takes the logits as input and outputs a vector of probabilities, where each probability represents the probability of the network predicting a particular class.

## Aditional research

### Responsable IA
I learned that there are great tools to study fairness in models, like data_validation and what-if-tool
https://www.tensorflow.org/tfx/data_validation/get_started
https://pair-code.github.io/what-if-tool/

### RAG vs Fine-tuing

**Retrieval-augmented generation (RAG)** and **fine-tuning** are two different approaches to improving the performance of large language models (LLMs).

**RAG** models work by combining retrieval and generation. First, the model retrieves relevant documents from a database of external knowledge. Then, the model uses the retrieved documents to generate a response to the query.

**Fine-tuning** involves training the LLM on a new dataset of text and code. This dataset is typically specific to the task that the model is being used for.

**RAG** models have several advantages over fine-tuning. First, RAG models are more efficient. This is because RAG models do not require the LLM to be retrained on a large dataset of labeled data. Second, RAG models are more adaptable. This is because RAG models can be used to solve a variety of different tasks by simply changing the retrieval database.

**Fine-tuning** models also have several advantages over RAG models. First, fine-tuned models are typically more accurate. This is because fine-tuned models are trained on a dataset of text and code that is specific to the task that the model is being used for. Second, fine-tuned models are typically faster. This is because fine-tuned models do not need to retrieve documents from a database before generating a response.

**Which approach is better for you will depend on your specific needs and requirements.** If you need a model that is efficient and adaptable, then a RAG model may be a good choice. If you need a model that is accurate and fast, then a fine-tuned model may be a better choice.

Here is a table that summarizes the key differences between RAG and fine-tuning:

| Characteristic | RAG | Fine-tuning |
|---|---|---|
| Efficiency | More efficient | Less efficient |
| Adaptability | More adaptable | Less adaptable |
| Accuracy | Less accurate | More accurate |
| Speed | Slower | Faster |

Here are some examples of tasks where RAG models are often used:

* Question answering
* Summarization
* Translation

Here are some examples of tasks where fine-tuned models are often used:

* Text classification
* Code generation
* Natural language inference

Overall, both RAG and fine-tuning are powerful approaches to improving the performance of LLMs. The best approach for you will depend on your specific needs and requirements.