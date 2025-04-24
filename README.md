# Personalized Recommendation Algorithm Based On Deep Learning

## Overview

Personalized recommendations are a key component of modern digital platforms, and understanding the performance trade-offs between different algorithmic approaches is critical for both research and practical implementation. This article uses the MovieLens dataset to make personalized recommendations for users based on movie characteristics and user characteristics. We focus on combining traditional collaborative filtering methods with neural networks to discover potential nonlinear relationships and bring better recommendations to users, and compare them with traditional collaborative filtering methods to judge the quality of each model based on RMSE, precision, recall, and F1 score.

## Project Structure

```
├── __pycache__/      # a Python cache directory
├── data/                 # Directory for dataset files
│   ├── ratings.csv       # User ratings data
│   ├── movies.csv        # Movie information 
│   ├── tags.csv          # User-generated tags
│   └── links.csv         # Links to external movie databases
├── results/              # Directory for results and visualizations
├── main.py               # Main entry point for the program
├── attngraphrec.py       # Core recommendation model implementation
├── data_analysis.py      # Data exploration and visualization 
├── NCF.py                # Collaborative filtering algorithms with neural networks added
├── CF.py                 # Collaborative filtering algorithms
├── recommendation_visualization.py  # Visualization of recommendation results
└── LSTM_recommender.py   # Alternative LSTM-based recommendation model
```

## Dataset

This project uses the MovieLens small dataset, which includes:
- 100,836 ratings from 610 users across 9,742 movies
- Ratings on a 5-star scale with half-star increments
- User-generated tags
- Movie metadata (titles, genres)
- Links to IMDb and TMDb 

## Features

### Core Features
- **Hybrid Architecture**: Combines graph neural networks with attention mechanisms for enhanced recommendation quality
- **Time-Aware Splitting**: Uses temporal information to create realistic train/validation/test splits
- **Advanced Metrics**: Evaluates recommendations using traditional accuracy metrics and novel diversity measures
- **Visualization Tools**: Comprehensive visualizations for data exploration and recommendation analysis
- **Integration with PyTorch**: Leverages PyTorch and PyTorch Geometric for efficient model implementation

### Technical Features
- Graph-based user-item interaction modeling
- Attention mechanism to highlight relevant features
- Time-aware training to handle temporal dynamics
- Automatic dependency checking and installation
- GPU acceleration when available

## Models

The project implements two recommendation models:

1. **AttnGraphRec **: A hybrid model that combines:
   - Graph Convolutional Networks (GCN) to model user-item interaction graph
   - Attention mechanism to focus on relevant features
   - Multi-layer perceptron for final prediction

2. **LSTMRecommender **: Sequence-based recommendation using:
   - LSTM to capture sequential patterns in user behavior
   - User and item embeddings
   - Dense layers for prediction

3. **NCF **: Collaborative filtering algorithms with neural networks added:
   - User and item embeddings
   - Add the MLP to capture non-linear relationship

## Metrics

The recommendation quality is evaluated using:

- **Accuracy Metrics**:
  - Root Mean Square Error (RMSE)
  - Precision, Recall, and F1 Score


## Usage

### Prerequisites
- Python 3.7+
- PyTorch
- PyTorch Geometric
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn

### Running the Project

1. Place the MovieLens dataset files in the `data` directory.

2. Run the  `main.py` 
```
