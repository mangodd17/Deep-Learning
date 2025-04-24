import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
def time_aware_split(ratings_df, test_size=0.2, validation_size=0.1):

    sorted_df = ratings_df.sort_values('timestamp')

    n = len(sorted_df)
    train_end = int(n * (1 - test_size - validation_size))
    val_end = int(n * (1 - test_size))

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]
    
    print(f"Training set: {len(train_df)} records ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Validation set: {len(val_df)} records ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")  
    print(f"Test set: {len(test_df)} records ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return train_df, val_df, test_df
def user_aware_split(ratings_df, test_size=0.2, min_history=5):

    sorted_df = ratings_df.sort_values(['userId', 'timestamp'])
    
    train_indices = []
    test_indices = []
    user_ids = sorted_df['userId'].unique()
    
    for user_id in user_ids:
        user_ratings = sorted_df[sorted_df['userId'] == user_id]
        n_ratings = len(user_ratings)
        
        if n_ratings < min_history:
            train_indices.extend(user_ratings.index)
        else:
            n_test = max(int(n_ratings * test_size), 1)
            n_train = n_ratings - n_test

            n_train = max(n_train, min_history)
            n_test = n_ratings - n_train

            user_train = user_ratings.iloc[:n_train]
            user_test = user_ratings.iloc[n_train:]
            
            train_indices.extend(user_train.index)
            test_indices.extend(user_test.index)

    train_df = sorted_df.loc[train_indices]
    test_df = sorted_df.loc[test_indices]
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    print(f"Users in training set: {train_df['userId'].nunique()}")
    print(f"Users in test set: {test_df['userId'].nunique()}")
    
    return train_df, test_df
def leave_one_out_split(ratings_df):
    
    # Sort by user ID and timestamp
    sorted_df = ratings_df.sort_values(['userId', 'timestamp'])
    last_interactions = sorted_df.groupby('userId').tail(1)
    test_df = last_interactions
    

    train_df = sorted_df.drop(last_interactions.index)
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    print(f"Users in training set: {train_df['userId'].nunique()}")
    print(f"Users in test set: {test_df['userId'].nunique()}")
    
    return train_df, test_df
class AttnGraphRec(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
        super(AttnGraphRec, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.num_users = num_users
        self.num_items = num_items
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph neural network layers
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, embedding_dim)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, user_indices, item_indices, edge_index=None):
        user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
        item_indices = torch.clamp(item_indices, 0, self.num_items - 1)

        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        if edge_index is not None and edge_index.size(1) > 0:
            try:
                x = torch.zeros((self.num_users + self.num_items, self.embedding_dim), device=user_indices.device)

                unique_users = torch.unique(user_indices)
                unique_items = torch.unique(item_indices)
                
                # Set user embeddings
                x[unique_users] = self.user_embedding(unique_users)
                
                # Set item embeddings
                x[unique_items + self.num_users] = self.item_embedding(unique_items)

                x = F.relu(self.gcn1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gcn2(x, edge_index)

                user_emb_graph = x[user_indices]
                item_emb_graph = x[item_indices + self.num_users]

                user_emb = user_emb + user_emb_graph
                item_emb = item_emb + item_emb_graph

        interaction = torch.cat([user_emb, item_emb], dim=1)
        prediction = self.interaction_mlp(interaction)
        
        return prediction.squeeze()
def load_and_preprocess_data(ratings_path, movies_path, tags_path=None, min_ratings=20):
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

    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    
    user_id_map = {id: i for i, id in enumerate(user_ids)}
    movie_id_map = {id: i for i, id in enumerate(movie_ids)}

    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
    ratings_df['item_idx'] = ratings_df['movieId'].map(movie_id_map)
    ratings_df['rating'] = ratings_df['rating'] / 5.0

    if tags_path and os.path.exists(tags_path):
        tags_df = pd.read_csv(tags_path)
        tags_df['user_idx'] = tags_df['userId'].map(user_id_map)
        tags_df['item_idx'] = tags_df['movieId'].map(movie_id_map)
        return ratings_df, movies_df, tags_df, user_id_map, movie_id_map
    
    return ratings_df, movies_df, None, user_id_map, movie_id_map
def create_graph_data(ratings_df):
    # Get user and item indices
    user_indices = ratings_df['user_idx'].values
    item_indices = ratings_df['item_idx'].values

    edge_index = torch.tensor([
        np.concatenate([user_indices, item_indices + ratings_df['user_idx'].max() + 1]),
        np.concatenate([item_indices + ratings_df['user_idx'].max() + 1, user_indices])
    ], dtype=torch.long)
    
    # Create edge attributes
    edge_attr = torch.tensor(
        np.concatenate([ratings_df['rating'].values, ratings_df['rating'].values]),
        dtype=torch.float
    )

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
    print(f"Created graph with {edge_index.shape[1]} edges")
    return data
def train_model(model, optimizer, train_loader, device):
    """Train the model"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        user_indices, item_indices, ratings = batch
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(user_indices, item_indices)
        
        # Calculate loss
        loss = F.mse_loss(predictions, ratings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
def evaluate_model(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in data_loader:
            user_indices, item_indices, ratings = batch
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            
            # Get predictions
            pred = model(user_indices, item_indices)
            
            predictions.append(pred.cpu())
            actuals.append(ratings)
    
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # Convert predictions to binary (like/dislike) for precision and recall calculation
    # Assume rating>=0.7 (3.5/5) means "like"
    binary_preds = (predictions >= 0.7).astype(int)
    binary_actuals = (actuals >= 0.7).astype(int)
    
    # Calculate precision and recall
    precision = precision_score(binary_actuals, binary_preds, zero_division=0)
    recall = recall_score(binary_actuals, binary_preds, zero_division=0)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return rmse, precision, recall, f1
def calculate_novelty(recommendations, user_history):
    """Calculate novelty of recommendations"""
    total_novel = 0
    count = 0
    
    for user, rec_items in recommendations.items():
        history_items = user_history.get(user, set())
        
        # Convert numpy array to list if needed
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()
            
        # Skip if no recommendations
        if not rec_items:
            continue
            
        # Calculate novelty for this user
        novel_items = [item for item in rec_items if item not in history_items]
        total_novel += len(novel_items) / len(rec_items)
        count += 1
    
    # Return average novelty
    return total_novel / count if count > 0 else 0
def calculate_genre_diversity(recommendations, movie_genres):
    all_genres = set()
    genre_counts = {}

    for genres in movie_genres.values():
        for genre in genres:
            all_genres.add(genre)
            genre_counts[genre] = 0
    
    # Count genre occurrences in recommendations
    total_recommendations = 0
    for user, rec_items in recommendations.items():
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()
            
        for item in rec_items:
            if item in movie_genres:
                for genre in movie_genres[item]:
                    genre_counts[genre] += 1
                total_recommendations += 1

    if total_recommendations == 0 or len(genre_counts) <= 1:
        return 0.5
    genre_proportions = []
    for genre, count in genre_counts.items():
        proportion = count / total_recommendations if total_recommendations > 0 else 0
        genre_proportions.append(proportion)
    
    # Calculate Gini coefficient
    genre_proportions.sort()
    n = len(genre_proportions)

    cumulative_proportion = 0
    total_inequality = 0
    
    for i, proportion in enumerate(genre_proportions):
        rank = (i + 1) / n
        cumulative_proportion += proportion

        total_inequality += rank - (cumulative_proportion / sum(genre_proportions))
    
    # Normalize Gini coefficient
    gini = 2 * total_inequality / (n - 1) if n > 1 else 0
    
    # Convert Gini coefficient to diversity metric (1-Gini)
    # Ensure value is within 0-1 range
    diversity = 1 - min(max(gini, 0), 1)
    
    return diversity
def generate_recommendations(model, num_users, num_items, device, top_k=10):
    """Generate recommendations for all users"""
    model.eval()
    recommendations = {}
    
    with torch.no_grad():
        for user_idx in range(num_users):
            # Predict scores for all items for this user
            user_indices = torch.full((num_items,), user_idx, dtype=torch.long).to(device)
            item_indices = torch.arange(num_items, dtype=torch.long).to(device)
            
            # Get predictions
            scores = model(user_indices, item_indices)
            
            # Get top-k items with highest scores
            _, top_indices = torch.topk(scores, min(top_k, len(scores)))
            recommendations[user_idx] = top_indices.cpu().numpy()
    
    return recommendations
def main():
    ratings_path = 'data/ratings.csv'
    movies_path = 'data/movies.csv'
    tags_path = 'data/tags.csv'
    
    # Minimum ratings threshold
    min_ratings = 20
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    if not os.path.exists(ratings_path):
        print(f"Error: Could not find {ratings_path}. Please check the file path.")
        return
    
    if not os.path.exists(movies_path):
        print(f"Error: Could not find {movies_path}. Please check the file path.")
        return
    print("Loading and preprocessing data...")
    try:
        ratings_df, movies_df, tags_df, user_id_map, movie_id_map = load_and_preprocess_data(
            ratings_path, movies_path, tags_path, min_ratings=min_ratings
        )
        print(f"Data loaded successfully. Found {len(user_id_map)} users and {len(movie_id_map)} movies.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    try:
        print("\nUsing time-aware dataset splitting method...")
        train_df, val_df, test_df = time_aware_split(ratings_df, test_size=0.1, validation_size=0.1)
        
        # Create data points
        X_train = train_df[['user_idx', 'item_idx']].values
        y_train = train_df['rating'].values
        
        X_val = val_df[['user_idx', 'item_idx']].values
        y_val = val_df['rating'].values
        
        X_test = test_df[['user_idx', 'item_idx']].values
        y_test = test_df['rating'].values
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
    except Exception as e:
        # If time-based splitting fails, fall back to traditional random splitting
        print(f"Time-based splitting failed: {e}")
        print("Falling back to traditional random splitting...")
        
        X = ratings_df[['user_idx', 'item_idx']].values
        y = ratings_df['rating'].values
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, random_state=SEED)
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
    
    # Create graph structure for training
    train_graph_df = train_df.copy()
    graph_data = create_graph_data(train_graph_df)
    edge_index = graph_data.edge_index
    
    # Get maximum indices for users and items to use for model dimensions
    max_user_idx = ratings_df['user_idx'].max()
    max_item_idx = ratings_df['item_idx'].max()

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
    
    batch_size = 1024
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_users = max_user_idx + 1
    num_items = max_item_idx + 1
    print(f"Initializing model with {num_users} users and {num_items} items...")
    model = AttnGraphRec(num_users, num_items).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-5)
    
    # Train model
    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_rmse = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0

    test_metrics = []
    epochs_tracked = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train model
        train_loss = train_model(model, optimizer, train_loader, device)
        train_losses.append(train_loss)

        val_rmse, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        val_losses.append(val_rmse)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        if (epoch + 1) % 5 == 0:
            test_rmse, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
            test_metrics.append((test_rmse, test_precision, test_recall, test_f1))
            epochs_tracked.append(epoch + 1)
            print(f"Test metrics at epoch {epoch+1}: RMSE: {test_rmse:.4f}, Precision: {test_precision:.4f}, "
                  f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Training completed!")
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation RMSE: {best_val_rmse:.4f}")
    
    # Final evaluation on test set
    test_rmse, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(f"\nTest set evaluation:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    try:
        previous_model_path = 'results/attngraphrec_model.pt'
        if os.path.exists(previous_model_path):
            previous_model = AttnGraphRec(num_users, num_items).to(device)
            previous_model.load_state_dict(torch.load(previous_model_path, map_location=device))
            
            prev_test_rmse, prev_test_precision, prev_test_recall, prev_test_f1 = evaluate_model(
                previous_model, test_loader, device
            )
            
            print("\nComparison with non-temporal model:")
            print(f"Non-temporal model - RMSE: {prev_test_rmse:.4f}, F1: {prev_test_f1:.4f}")
            print(f"Temporal model - RMSE: {test_rmse:.4f}, F1: {test_f1:.4f}")
            
            improvement = (prev_test_rmse - test_rmse) / prev_test_rmse * 100
            print(f"RMSE Improvement: {improvement:.2f}%")
    except Exception as e:
        print(f"Could not compare with non-temporal model: {e}")

    torch.save(model.state_dict(), 'results/attngraphrec_time_aware_model.pt')
    print("Time-aware model saved to 'results/attngraphrec_time_aware_model.pt'")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation RMSE')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/time_aware_training_progress.png')
    print("Training progress plot saved to 'results/time_aware_training_progress.png'")

    if test_metrics:
        epochs = range(5, num_epochs + 1, 5)
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
        plt.savefig('results/evaluation_metrics.png')

    print("Metric charts saved")

if __name__ == "__main__":
    main()