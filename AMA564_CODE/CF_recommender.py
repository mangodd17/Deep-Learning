import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class MatrixFactorizationCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, dropout=0.1):
        super(MatrixFactorizationCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Global bias term
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # User and item bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store dimensions for safer index handling
        self.num_users = num_users
        self.num_items = num_items
        
        # Initialize embeddings
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Normal distribution"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_indices, item_indices):
        user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
        item_indices = torch.clamp(item_indices, 0, self.num_items - 1)
        
        # Get embeddings and apply dropout
        user_emb = self.dropout(self.user_embedding(user_indices))
        item_emb = self.dropout(self.item_embedding(item_indices))
        
        # Get bias terms
        user_b = self.user_bias(user_indices).squeeze()
        item_b = self.item_bias(item_indices).squeeze()
        
        # Calculate dot product
        dot_product = torch.sum(user_emb * item_emb, dim=1)
        
        # Combine with biases
        prediction = self.global_bias + user_b + item_b + dot_product
        
        return torch.sigmoid(prediction)  # Scale to 0-1

def load_and_preprocess_data(ratings_path, movies_path, min_ratings=20):
    # Load data
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # Filter movies with at least min_ratings ratings
    if min_ratings > 0:
        movie_counts = ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings].index.tolist()
        print(f"Filtering to {len(valid_movies)} movies with at least {min_ratings} ratings")
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movies)]
        print(f"Filtered to {len(ratings_df)} ratings")

    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    
    user_id_map = {id: i for i, id in enumerate(user_ids)}
    movie_id_map = {id: i for i, id in enumerate(movie_ids)}
    
    # Map user and item IDs
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
    ratings_df['item_idx'] = ratings_df['movieId'].map(movie_id_map)
    
    # Normalize ratings
    ratings_df['rating'] = ratings_df['rating'] / 5.0
    
    return ratings_df, movies_df, user_id_map, movie_id_map

def train_model(model, optimizer, train_loader, device, criterion):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        user_indices, item_indices, ratings = batch
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        ratings = ratings.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device, criterion):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_indices, item_indices, ratings = batch
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            ratings = ratings.to(device)
            
            # Get predictions
            pred = model(user_indices, item_indices)
            loss = criterion(pred, ratings)
            total_loss += loss.item()
            
            predictions.append(pred.cpu())
            actuals.append(ratings.cpu())

    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()

    rmse = np.sqrt(mean_squared_error(actuals, predictions * 5.0))  # Scale back to 5-star rating
    
    # Calculate classification metrics for "liked" items (rating >= 0.7 which is 3.5/5)
    binary_preds = (predictions >= 0.7).astype(int)
    binary_actuals = (actuals >= 0.7).astype(int)
    
    # Calculate precision and recall
    precision = np.sum((binary_preds == 1) & (binary_actuals == 1)) / (np.sum(binary_preds == 1) + 1e-10)
    recall = np.sum((binary_preds == 1) & (binary_actuals == 1)) / (np.sum(binary_actuals == 1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return total_loss / len(test_loader), rmse, precision, recall, f1

def generate_recommendations(model, user_id_map, movie_id_map, ratings_df, device, top_k=10):
    """Generate movie recommendations for all users"""
    model.eval()
    recommendations = {}

    user_indices = list(range(len(user_id_map)))
    num_items = len(movie_id_map)
    reverse_user_map = {idx: user_id for user_id, idx in user_id_map.items()}
    reverse_movie_map = {idx: movie_id for movie_id, idx in movie_id_map.items()}

    user_history = {}
    for _, row in ratings_df.iterrows():
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        if user_idx not in user_history:
            user_history[user_idx] = set()
        user_history[user_idx].add(item_idx)
    
    for user_idx in tqdm(user_indices, desc="Generating recommendations"):
        # Skip users with no history
        if user_idx not in user_history:
            continue
        unrated_items = list(set(range(num_items)) - user_history[user_idx])
        if not unrated_items:
            continue

        user_tensor = torch.tensor([user_idx] * len(unrated_items), dtype=torch.long, device=device)
        item_tensor = torch.tensor(unrated_items, dtype=torch.long, device=device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(user_tensor, item_tensor).cpu().numpy()
        
        # Create (item, score) pairs
        item_scores = list(zip(unrated_items, predictions))

        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k items
        top_items = [item for item, _ in item_scores[:top_k]]
        recommendations[user_idx] = np.array(top_items)
    
    return recommendations

def calculate_metrics(recommendations, user_history, movie_genres):
    """Calculate recommendation metrics: novelty and diversity"""
    total_novel = 0
    count = 0
    
    for user_idx, rec_items in recommendations.items():
        if user_idx not in user_history:
            continue
        
        history = user_history[user_idx]
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()
        if not rec_items:
            continue
        
        # Calculate novelty for this user
        novel_items = [item for item in rec_items if item not in history]
        total_novel += len(novel_items) / len(rec_items)
        count += 1
    
    novelty = total_novel / count if count > 0 else 0
    all_genres = set()
    genre_counts = {}
    for genres in movie_genres.values():
        for genre in genres:
            all_genres.add(genre)
            genre_counts[genre] = 0
    
    # Count genre occurrences in recommendations
    total_recommendations = 0
    for user_idx, rec_items in recommendations.items():
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()
        
        for item in rec_items:
            if item in movie_genres:
                for genre in movie_genres[item]:
                    genre_counts[genre] += 1
                total_recommendations += 1
    
    # Calculate diversity using Gini coefficient
    if not genre_counts or total_recommendations == 0:
        return novelty, 0.5  # Default diversity value
    props = np.array([count/total_recommendations for count in genre_counts.values()])
    props.sort()
    n = len(props)
    gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * props)) / n
    diversity = 1 - gini

    all_items = set(range(len(movie_genres)))
    recommended_items = set()
    for rec_items in recommendations.values():
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()
        recommended_items.update(rec_items)
    
    coverage = len(recommended_items) / len(all_items) if all_items else 0
    
    return novelty, diversity, coverage

def plot_training_history(train_losses, val_losses, metrics_history):
    """Plot training history"""
    os.makedirs('results', exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot RMSE
    plt.subplot(2, 2, 2)
    plt.plot([m['rmse'] for m in metrics_history])
    plt.title('RMSE History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    
    # Plot Precision, Recall, F1
    plt.subplot(2, 2, 3)
    plt.plot([m['precision'] for m in metrics_history], label='Precision')
    plt.plot([m['recall'] for m in metrics_history], label='Recall')
    plt.plot([m['f1'] for m in metrics_history], label='F1')
    plt.title('Classification Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/cf_training_history.png')
    plt.close()

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    ratings_path = 'data/ratings.csv'
    movies_path = 'data/movies.csv'
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print("Error: Data files not found. Please ensure 'ratings.csv' and 'movies.csv' are in the 'data' directory.")
        return
    
    # Parameters
    min_ratings = 20  # Minimum number of ratings for a movie
    test_size = 0.2   # Proportion of test set
    batch_size = 1024  # Batch size for training
    epochs = 30       # Number of training epochs

    print("\nLoading and preprocessing data...")
    ratings_df, movies_df, user_id_map, movie_id_map = load_and_preprocess_data(
        ratings_path, movies_path, min_ratings
    )
    
    print(f"Loaded data: {len(ratings_df)} ratings, {len(user_id_map)} users, {len(movie_id_map)} movies")

    X = ratings_df[['user_idx', 'item_idx']].values
    y = ratings_df['rating'].values
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_train[:, 0]),
        torch.LongTensor(X_train[:, 1]),
        torch.FloatTensor(y_train)
    )
    
    test_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_test[:, 0]),
        torch.LongTensor(X_test[:, 1]),
        torch.FloatTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    num_users = len(user_id_map)
    num_items = len(movie_id_map)
    
    model = MatrixFactorizationCF(num_users, num_items).to(device)
    print("Initialized Matrix Factorization CF model")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model_path = 'results/cf_model.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train model
        print("\nStarting model training...")
        train_losses = []
        val_losses = []
        metrics_history = []
        
        for epoch in range(epochs):
            # Train
            train_loss = train_model(model, optimizer, train_loader, device, criterion)
            train_losses.append(train_loss)
            
            # Evaluate
            val_loss, rmse, precision, recall, f1 = evaluate_model(model, test_loader, device, criterion)
            val_losses.append(val_loss)
            
            metrics_history.append({
                'rmse': rmse,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"RMSE: {rmse:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Plot training history
        plot_training_history(train_losses, val_losses, metrics_history)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(model, user_id_map, movie_id_map, ratings_df, device)

    user_history = {}
    for _, row in ratings_df.iterrows():
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        if user_idx not in user_history:
            user_history[user_idx] = set()
        user_history[user_idx].add(item_idx)
    
    # Create movie genres dictionary
    movie_genres = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        if movie_id in movie_id_map:
            item_idx = movie_id_map[movie_id]
            genres = row['genres'].split('|')
            movie_genres[item_idx] = genres
    
    # Calculate recommendation metrics
    print("\nCalculating recommendation metrics...")
    novelty, diversity, coverage = calculate_metrics(recommendations, user_history, movie_genres)
    
    print(f"Novelty: {novelty:.4f}")
    print(f"Diversity: {diversity:.4f}")
    print(f"Coverage: {coverage:.4f}")
    
    # Final evaluation on test set
    test_loss, rmse, precision, recall, f1 = evaluate_model(model, test_loader, device, criterion)
    
    print("\nFinal model evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'novelty': novelty,
        'diversity': diversity,
        'coverage': coverage
    }
    
    with open('results/cf_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nEvaluation complete! Results saved to 'results' directory.")

if __name__ == "__main__":
    main()
