# Sentiment Analysis Using BERT

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers). The model is fine-tuned on the IMDB dataset to classify movie reviews as either positive or negative. Additionally, it includes functionality to fetch and analyze user reviews from IMDb using their API.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Using the IMDb API](#using-the-imdb-api)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to perform sentiment analysis on movie reviews using a pre-trained BERT model. The model is fine-tuned on the IMDB dataset, and it includes additional functionality to fetch real-time user reviews from IMDb using the RapidAPI.

## Installation

To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have access to Google Colab, as the project is designed to run there.

4. If using the IMDb API, set up your RapidAPI key.

## Dataset

The IMDB dataset is used for training and validation. It consists of 50,000 movie reviews, labeled as either positive or negative.

- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Model Architecture

This project uses the BERT model for sequence classification. The model architecture includes:

- A pre-trained BERT base model.
- A classification head for binary sentiment prediction.

### Key Features

- Fine-tuning on the IMDB dataset.
- Usage of attention masks to handle padding.
- Tokenization using `bert-base-uncased`.
- Gradient updates only on the classification head and the last two encoder layers.

## Training the Model

To train the model:

1. Load the dataset and preprocess it using the `pipeline` function.
2. Split the dataset into training and validation sets.
3. Fine-tune the BERT model on the training data.
4. Evaluate the model on the validation set.

Training and evaluation are performed on GPU if available.

## Evaluation

The model is evaluated using metrics such as:

- Accuracy
- Confusion Matrix

The project includes a function to visualize the confusion matrix and compare it to a perfect prediction scenario.

## Using the IMDb API

The project also includes scripts to fetch user reviews for movies from IMDb using the RapidAPI. The following steps outline the process:

1. Fetch the IMDb ID for a movie title.
2. Retrieve a specified number of user reviews.
3. Analyze the sentiment of these reviews using the trained BERT model.

## Results

The model achieves satisfactory performance in sentiment classification. The confusion matrix and accuracy scores are visualized to provide insights into the model's predictions.

## Contributing

If you wish to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. Please ensure your contributions are well-documented.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
