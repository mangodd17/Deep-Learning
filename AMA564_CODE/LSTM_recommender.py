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

class LSTMRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.2):
        super(LSTMRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.num_users = num_users
        self.num_items = num_items
    
    def forward(self, user_indices, item_indices, user_history=None):
        # Get user and item embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        if user_history is not None and user_history.size(0) > 0:
            batch_size = user_indices.size(0)
            history_emb = self.item_embedding(user_history)
            
            # Apply LSTM to get user sequential preference
            _, (h_n, _) = self.lstm(history_emb)
            lstm_out = h_n[-1]
        else:
            lstm_out = torch.zeros(user_emb.size(0), self.lstm.hidden_size, device=user_emb.device)
        
        # Concatenate LSTM output with item embedding for prediction
        combined = torch.cat([lstm_out, item_emb], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x).squeeze()
        
        return torch.sigmoid(output)  # Scale to 0-1

def load_and_preprocess_data(ratings_path, movies_path, min_ratings=20):
    # Load data
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # Filter movies with at least min_ratings ratings
    if min_ratings > 0:
        movie_counts = ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings].index.tolist()
        print(f"Filtering to {len(valid_movies)} movies with at least {min_ratings} ratings")
        
        # Filter ratings to only include valid movies
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movies)]
        print(f"Filtered to {len(ratings_df)} ratings")
    
    # Create mapping dictionaries for users and items
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

def time_aware_split(ratings_df, test_size=0.2, validation_size=0.1):
    # Sort by timestamp
    sorted_df = ratings_df.sort_values('timestamp')
    
    # Calculate split points
    n = len(sorted_df)
    train_end = int(n * (1 - test_size - validation_size))
    val_end = int(n * (1 - test_size))
    
    # Split datasets
    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]
    
    print(f"Training set: {len(train_df)} records ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Validation set: {len(val_df)} records ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")  
    print(f"Test set: {len(test_df)} records ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return train_df, val_df, test_df

def user_aware_split(ratings_df, test_size=0.2, min_history=5):
    # Sort by user ID and timestamp
    sorted_df = ratings_df.sort_values(['userId', 'timestamp'])
    
    train_indices = []
    test_indices = []

    user_ids = sorted_df['userId'].unique()
    
    for user_id in user_ids:
        # Get all ratings for current user
        user_ratings = sorted_df[sorted_df['userId'] == user_id]
        n_ratings = len(user_ratings)
        
        if n_ratings < min_history:
            train_indices.extend(user_ratings.index)
        else:
            # Calculate test set size
            n_test = max(int(n_ratings * test_size), 1)
            n_train = n_ratings - n_test
            
            # Ensure training set has at least min_history records
            n_train = max(n_train, min_history)
            n_test = n_ratings - n_train
            
            # Get indices
            user_train = user_ratings.iloc[:n_train]
            user_test = user_ratings.iloc[n_train:]
            
            train_indices.extend(user_train.index)
            test_indices.extend(user_test.index)
    
    # Create training and test sets
    train_df = sorted_df.loc[train_indices]
    test_df = sorted_df.loc[test_indices]
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    print(f"Users in training set: {train_df['userId'].nunique()}")
    print(f"Users in test set: {test_df['userId'].nunique()}")
    
    return train_df, test_df

def get_user_sequences(ratings_df, max_seq_length=10):
    """Create sequence data for LSTM model"""
    ratings_df = ratings_df.sort_values(['user_idx', 'timestamp'])
    
    # Group by user
    user_sequences = {}
    for user_idx, group in ratings_df.groupby('user_idx'):
        items = group['item_idx'].values
        # Only keep last max_seq_length items
        if len(items) > max_seq_length:
            items = items[-max_seq_length:]
        user_sequences[user_idx] = items
    
    return user_sequences

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
    
    # Calculate metrics
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
    
    # Get all unique users and items
    user_indices = list(range(len(user_id_map)))
    num_items = len(movie_id_map)

    reverse_user_map = {idx: user_id for user_id, idx in user_id_map.items()}
    reverse_movie_map = {idx: movie_id for movie_id, idx in movie_id_map.items()}
    
    # Get user history to avoid recommending already rated items
    user_history = {}
    for _, row in ratings_df.iterrows():
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        if user_idx not in user_history:
            user_history[user_idx] = set()
        user_history[user_idx].add(item_idx)
    
    for user_idx in tqdm(user_indices, desc="Generating recommendations"):
        if user_idx not in user_history:
            continue
        
        # Get all unrated items for this user
        unrated_items = list(set(range(num_items)) - user_history[user_idx])
        if not unrated_items:
            continue

        user_tensor = torch.tensor([user_idx] * len(unrated_items), dtype=torch.long, device=device)
        item_tensor = torch.tensor(unrated_items, dtype=torch.long, device=device)

        with torch.no_grad():
            predictions = model(user_tensor, item_tensor).cpu().numpy()

        item_scores = list(zip(unrated_items, predictions))
        
        # Sort by predicted score
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
        

        novel_items = [item for item in rec_items if item not in history]
        total_novel += len(novel_items) / len(rec_items)
        count += 1
    
    novelty = total_novel / count if count > 0 else 0

    all_genres = set()
    genre_counts = {}
    
    # Collect all genres
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
        return novelty, 0.5
    
    # Calculate proportions
    props = np.array([count/total_recommendations for count in genre_counts.values()])
    props.sort()
    n = len(props)
    
    # Calculate Gini coefficient
    gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * props)) / n
    diversity = 1 - gini
    
    # Calculate coverage
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
    # Create directory for results
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
    plt.savefig('results/lstm_training_history.png')
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
    val_size = 0.1    # Proportion of validation set
    batch_size = 128  # Batch size for training
    epochs = 100      # Number of training epochs
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    ratings_df, movies_df, user_id_map, movie_id_map = load_and_preprocess_data(
        ratings_path, movies_path, min_ratings
    )
    
    print(f"Loaded data: {len(ratings_df)} ratings, {len(user_id_map)} users, {len(movie_id_map)} movies")
    
    # Time-aware splitting
    print("\nApplying time-aware data splitting...")
    try:
        train_df, val_df, test_df = time_aware_split(ratings_df, test_size=test_size, validation_size=val_size)

        X_train = train_df[['user_idx', 'item_idx']].values
        y_train = train_df['rating'].values
        
        X_val = val_df[['user_idx', 'item_idx']].values
        y_val = val_df['rating'].values
        
        X_test = test_df[['user_idx', 'item_idx']].values
        y_test = test_df['rating'].values
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"Error in time-aware splitting: {e}")
        print("Falling back to random splitting...")
        
        # Prepare training data using the traditional approach
        X = ratings_df[['user_idx', 'item_idx']].values
        y = ratings_df['rating'].values
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
        
        # Create validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=SEED)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_train[:, 0]),
        torch.LongTensor(X_train[:, 1]),
        torch.FloatTensor(y_train)
    )
    
    val_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_val[:, 0]),
        torch.LongTensor(X_val[:, 1]),
        torch.FloatTensor(y_val)
    )
    
    test_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_test[:, 0]),
        torch.LongTensor(X_test[:, 1]),
        torch.FloatTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    num_users = len(user_id_map)
    num_items = len(movie_id_map)
    
    model = LSTMRecommender(num_users, num_items).to(device)
    print("Initialized LSTM recommendation model")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Check if model already exists
    model_path = 'results/lstm_model.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train model
        print("\nStarting model training...")
        train_losses = []
        val_losses = []
        metrics_history = []

        test_metrics = []
        
        best_val_loss = float('inf')
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = train_model(model, optimizer, train_loader, device, criterion)
            train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_loss, val_rmse, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device, criterion)
            val_losses.append(val_loss)
            
            metrics_history.append({
                'rmse': val_rmse,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            })
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"RMSE: {val_rmse:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

            if (epoch + 1) % 5 == 0:
                test_loss, test_rmse, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device, criterion)
                test_metrics.append((test_rmse, test_precision, test_recall, test_f1))
                print(f"Test metrics at epoch {epoch+1}: RMSE: {test_rmse:.4f}, Precision: {test_precision:.4f}, "
                      f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the current best model
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved (validation loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Plot training history
        plot_training_history(train_losses, val_losses, metrics_history)
        
        # NEW: Plot evaluation metrics
        if test_metrics:
            epochs = range(5, epochs + 1, 5)
            # Truncate epochs to match the actual number of test metrics collected
            epochs = epochs[:len(test_metrics)]
            
            rmse_values = [m[0] for m in test_metrics]
            precision_values = [m[1] for m in test_metrics]
            recall_values = [m[2] for m in test_metrics]
            f1_values = [m[3] for m in test_metrics]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epochs, rmse_values)
            plt.title('RMSE')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2, 2)
            plt.plot(epochs, precision_values)
            plt.title('Precision')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2, 3)
            plt.plot(epochs, recall_values)
            plt.title('Recall')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2, 4)
            plt.plot(epochs, f1_values)
            plt.title('F1 Score')
            plt.xlabel('Epoch')
            
            plt.tight_layout()
            plt.savefig('results/lstm_evaluation_metrics.png')
            
            print("Evaluation metric charts saved")
        
        # Load the best model for evaluation
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model from {model_path}")
    
    # Evaluate on test set
    test_loss, test_rmse, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device, criterion)
    print("\nTest set evaluation:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    if 'train_df' in locals():
        ratings_for_rec = pd.concat([train_df, val_df])
        recommendations = generate_recommendations(model, user_id_map, movie_id_map, ratings_for_rec, device)
    else:
        recommendations = generate_recommendations(model, user_id_map, movie_id_map, ratings_df, device)
    
    # Create user history for metrics calculation
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
    
    # Save metrics
    metrics = {
        'rmse': test_rmse,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'novelty': novelty,
        'diversity': diversity,
        'coverage': coverage
    }
    
    with open('results/lstm_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nEvaluation complete! Results saved to 'results' directory.")

if __name__ == "__main__":
    main()
