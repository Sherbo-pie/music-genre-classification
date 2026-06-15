# Music Genre Classification Using Frequency Domain Analysis

## Overview

This project explores automatic music genre classification by combining signal processing techniques with machine learning models. Audio signals are transformed into informative frequency-domain features, which are then used to train classifiers capable of distinguishing between musical genres.

## Dataset

- **Dataset:** GTZAN Genre Collection
- **Genres:** 10 music genres
- **Samples:** 1000 audio clips
- **Clip Length:** 30 seconds each

## Methodology

### Feature Extraction

Audio signals were processed using frequency-domain analysis techniques to extract:

- **MFCC (Mel-Frequency Cepstral Coefficients)** – captures timbre and characteristics of human auditory perception
- **Zero Crossing Rate (ZCR)** – measures signal sign changes and noisiness
- **Spectral Centroid** – indicates the perceived brightness of a sound

These features were extracted using signal processing concepts including the Discrete Fourier Transform (DFT), Mel-scale filtering, and Discrete Cosine Transform (DCT).

## Models Evaluated

The extracted features were used to train and compare:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN)
- Gradient Boosting

## Results

| Model | Accuracy |
|---------|---------|
| KNN | 64.0% |
| SVM | 67.5% |
| ANN | **68.5%** |
| Gradient Boosting | 61.0% |

### Key Observations

- ANN achieved the best overall performance.
- Classical and Metal genres showed strong classification performance.
- Rock exhibited lower performance due to spectral similarity with neighboring genres.
- Results highlight the importance of feature representation in audio classification tasks.

## Key Takeaways

- Demonstrates the connection between signal processing and machine learning.
- Shows how frequency-domain features can effectively represent audio signals.
- Highlights the strengths and limitations of different classification models for music data.

## Tools & Libraries

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## Report

A detailed report containing feature extraction methodology, experiments, model comparisons, and analysis is included in this repository.
