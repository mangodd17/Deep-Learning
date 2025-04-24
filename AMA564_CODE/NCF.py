import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class NCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32], dropout=0.3):
        super(NCF, self).__init__()
        self.embedding_dim = embedding_dim


        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        self.mlp_layers = []
        input_size = embedding_dim * 2

        # Build MLP layers
        for i, layer_size in enumerate(layers):
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size

        self.mlp_layers = nn.ModuleList(self.mlp_layers)

        self.output_layer = nn.Linear(layers[-1] + embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
        item_indices = torch.clamp(item_indices, 0, self.num_items - 1)

        # GMF
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_gmf * item_gmf  # Element-wise product

        # MLP
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=1)  # Concatenation

        # Process through MLP layers
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        # Concatenate GMF and MLP outputs
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=1)

        # Final prediction
        prediction = self.output_layer(concat_vector)

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

    if tags_path and os.path.exists(tags_path):
        tags_df = pd.read_csv(tags_path)
        tags_df['user_idx'] = tags_df['userId'].map(user_id_map)
        tags_df['item_idx'] = tags_df['movieId'].map(movie_id_map)
        return ratings_df, movies_df, tags_df, user_id_map, movie_id_map

    return ratings_df, movies_df, None, user_id_map, movie_id_map


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

    # RMSE
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
    # Calculate proportion of items in recommendations that are not in user history
    total_novel = 0
    count = 0

    for user, rec_items in recommendations.items():
        history_items = user_history.get(user, set())

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
    """
    Calculate the genre diversity of recommendations using the Gini coefficient.
    Values between 0-1, with higher values indicating more diversity.
    """
    all_genres = set()
    genre_counts = {}

    # Collect all genres
    for genres in movie_genres.values():
        for genre in genres:
            all_genres.add(genre)
            genre_counts[genre] = 0
    total_recommendations = 0
    for user, rec_items in recommendations.items():
        if hasattr(rec_items, 'tolist'):
            rec_items = rec_items.tolist()

        for item in rec_items:
            if item in movie_genres:
                for genre in movie_genres[item]:
                    genre_counts[genre] += 1
                total_recommendations += 1

    # If no recommendations or only one genre, return default value
    if total_recommendations == 0 or len(genre_counts) <= 1:
        return 0.5

    # Convert counts to proportions
    genre_proportions = []
    for genre, count in genre_counts.items():
        proportion = count / total_recommendations if total_recommendations > 0 else 0
        genre_proportions.append(proportion)

    # Calculate Gini coefficient
    genre_proportions.sort()
    n = len(genre_proportions)

    # Calculate using Lorenz curve
    cumulative_proportion = 0
    total_inequality = 0

    for i, proportion in enumerate(genre_proportions):
        rank = (i + 1) / n
        cumulative_proportion += proportion

        total_inequality += rank - (cumulative_proportion / sum(genre_proportions))

    # Normalize Gini coefficient
    gini = 2 * total_inequality / (n - 1) if n > 1 else 0
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

            scores = model(user_indices, item_indices)

            # Get top-k items with highest scores
            _, top_indices = torch.topk(scores, min(top_k, len(scores)))
            recommendations[user_idx] = top_indices.cpu().numpy()

    return recommendations


def main():
    # Data paths
    ratings_path = 'data/ratings.csv'
    movies_path = 'data/movies.csv'
    tags_path = 'data/tags.csv'

    # Minimum ratings threshold
    min_ratings = 20

    # Make sure the data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if not os.path.exists(ratings_path):
        print(f"Error: Could not find {ratings_path}. Please check the file path.")
        return

    if not os.path.exists(movies_path):
        print(f"Error: Could not find {movies_path}. Please check the file path.")
        return

    # Load data
    print("Loading and preprocessing data...")
    try:
        ratings_df, movies_df, tags_df, user_id_map, movie_id_map = load_and_preprocess_data(
            ratings_path, movies_path, tags_path, min_ratings=min_ratings
        )
        print(f"Data loaded successfully. Found {len(user_id_map)} users and {len(movie_id_map)} movies.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get maximum indices for users and items to use for model dimensions
    max_user_idx = ratings_df['user_idx'].max()
    max_item_idx = ratings_df['item_idx'].max()

    ratings_df = ratings_df.sort_values('timestamp')
    test_size = int(0.2 * len(ratings_df))

    train_df = ratings_df[:-test_size]
    test_df = ratings_df[-test_size:]
    X_train = train_df[['user_idx', 'item_idx']].values
    y_train = train_df['rating'].values
    X_test = test_df[['user_idx', 'item_idx']].values
    y_test = test_df['rating'].values

    print(f"Time-aware split:")
    print(f"- Training set: {len(X_train)} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"- Test set: {len(X_test)} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    # Create data loaders
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

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_users = max_user_idx + 1
    num_items = max_item_idx + 1
    print(f"Initializing NCF model with {num_users} users and {num_items} items...")

    # Define MLP layer sizes
    mlp_layers = [128, 64, 32]
    model = NCF(num_users, num_items, embedding_dim=64, layers=mlp_layers).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Train model
    num_epochs = 30
    train_losses = []
    test_metrics = []

    print("Starting training...")
    for epoch in range(num_epochs):
        # Train model
        train_loss = train_model(model, optimizer, train_loader, device)
        train_losses.append(train_loss)

        # Evaluate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            rmse, precision, recall, f1 = evaluate_model(model, test_loader, device)
            test_metrics.append((rmse, precision, recall, f1))
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, RMSE: {rmse:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    print("Training completed!")

    # Generate recommendations and calculate novelty and diversity metrics
    print("Generating recommendations and calculating metrics...")
    recommendations = generate_recommendations(model, num_users, num_items, device)

    # Create user history dictionary
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

    # Calculate novelty and diversity
    novelty = calculate_novelty(recommendations, user_history)
    diversity = calculate_genre_diversity(recommendations, movie_genres)

    print(f"Recommendation Novelty: {novelty:.4f}")
    print(f"Genre Diversity: {diversity:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'results/ncf_model.pt')
    print("Model saved to 'results/ncf_model.pt'")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')

    # Plot evaluation metrics
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