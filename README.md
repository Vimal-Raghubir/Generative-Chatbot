# Generative-Chatbot

This repository contains a Jupyter notebook designed to train a generative chatbot using the Ubuntu Dialogue Corpus. The dataset, available at: https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus, consists of conversations extracted from Ubuntu chat logs, typically used for Ubuntu technical support.

## Notebook Overview

### Key Features:


1. Model Development:
- Implementation of two sequence-to-sequence models:
    - One with an attention mechanism (Luong or Bahdanau).
    - One without attention.
- Use of PyTorch for model training and optimization.
2. Data Preprocessing:
- Cleaning and tokenizing conversation text.
- Create dialogue pairs to mimic conversations.
- Splitting the data into training and testing sets.
3. Evaluation Metrics:
- BLEU and METEOR scores for measuring the quality of generated responses.
- Cosine similarity for response evaluation.
4. Libraries Used:
- PyTorch
- TensorFlow (Tokenizer)
- NLTK (BLEU and METEOR scores)
- Scikit-learn
- Pandas
- NumPy

## Requirements
To run the notebook, ensure the following Python libraries are installed:

- PyTorch
- TensorFlow
- NLTK
- Scikit-learn
- Pandas
- NumPy
- tqdm

Install the required libraries using:

`pip install torch tensorflow nltk scikit-learn pandas numpy tqdm`

### How to Use

1. Clone the repository:

`git clone https://github.com/Vimal-Raghubir/Generative-Chatbot.git`

2. Open the Jupyter notebook:

`jupyter notebook Generative_Chatbot.ipynb`

3. Follow the structured steps in the notebook to:

- Preprocess the dataset.
- Train and evaluate the models.
- Generate chatbot responses.

## Performance Evaluation

The notebook evaluates model performance using:

- BLEU scores for n-gram overlap between generated and reference responses.
- METEOR scores for semantic similarity.
- Cosine similarity for vector-based evaluation.

## Future Enhancements

Potential improvements include:

- Expanding training with larger models or additional datasets.
- Implementing Transformer-based architectures like GPT or BERT for enhanced response quality.
- Fine-tuning the attention mechanism for better context understanding.

## Acknowledgments
This project is inspired by the Ubuntu Dialogue Corpus and aims to demonstrate the development of generative chatbots for technical support. Special thanks to Kaggle for providing the dataset.