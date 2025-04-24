import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from attngraphrec import AttnGraphRec, load_and_preprocess_data, create_graph_data, generate_recommendations

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_model_and_data(model_path, ratings_path, movies_path, tags_path=None, min_ratings=20):
    # Load data
    try:
        ratings_df, movies_df, tags_df, user_id_map, movie_id_map = load_and_preprocess_data(
            ratings_path, movies_path, tags_path, min_ratings=min_ratings
        )
    except ValueError as e:
        print(f"Warning: {e}")
        print("Trying alternative function signature...")
        ratings_df, movies_df, user_id_map, movie_id_map = load_and_preprocess_data(
            ratings_path, movies_path, min_ratings=min_ratings
        )
        tags_df = None
    
    # Create graph data
    graph_data = create_graph_data(ratings_df)
    edge_index = graph_data.edge_index

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_users = max(ratings_df['user_idx'].max() + 1, len(user_id_map))
    num_items = max(ratings_df['item_idx'].max() + 1, len(movie_id_map))
    model = AttnGraphRec(num_users, num_items).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    reverse_user_map = {v: k for k, v in user_id_map.items()}
    reverse_movie_map = {v: k for k, v in movie_id_map.items()}
    
    return model, ratings_df, movies_df, edge_index, device, user_id_map, movie_id_map, reverse_user_map, reverse_movie_map

def get_user_history(ratings_df, user_id, movie_id_map, reverse_movie_map, movies_df, reverse_user_map=None, top_n=10):
    """Get user's viewing history"""
    if isinstance(user_id, int):
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
    else:
        if reverse_user_map is None:
            print("Warning: reverse_user_map not provided, assuming user_id is already the original ID")
            user_id_orig = user_id
        else:
            user_id_orig = reverse_user_map[user_id]
        user_ratings = ratings_df[ratings_df['userId'] == user_id_orig]
    
    # Sort by rating
    top_rated = user_ratings.sort_values('rating', ascending=False).head(top_n)
    
    # Get movie information
    history = []
    for _, row in top_rated.iterrows():
        movie_id = row['movieId']
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if movie_info.empty:
            continue
        movie_info = movie_info.iloc[0]
        
        history.append({
            'movieId': int(movie_id),
            'title': str(movie_info['title']),
            'genres': str(movie_info['genres']),
            'rating': float(row['rating'] * 5.0)  # Convert back to 5-star scale
        })
    
    return history

def generate_user_recommendations(model, user_idx, edge_index, device, num_items, top_n=10):
    """Generate movie recommendations for a specific user"""
    model.eval()
    
    with torch.no_grad():
        # Predict scores for all movies for this user
        user_indices = torch.full((num_items,), user_idx, dtype=torch.long).to(device)
        item_indices = torch.arange(num_items, dtype=torch.long).to(device)

        try:
            scores = model(user_indices, item_indices, edge_index.to(device))
        except (TypeError, AttributeError) as e:
            print(f"Warning: {e}")
            try:
                scores = model(user_indices, item_indices)
            except Exception as e2:
                print(f"Error generating recommendations: {e2}")
                return np.arange(top_n), np.ones(num_items) * 0.8
        
        # Get top-k items with highest scores
        _, top_indices = torch.topk(scores, min(top_n, len(scores)))
        
        return top_indices.cpu().numpy(), scores.cpu().numpy()

def visualize_recommendations(user_history, recommendations, output_path):
    """Visualize user history and recommendations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # User history visualization
    history_titles = [item['title'][:30] + '...' if len(item['title']) > 30 else item['title'] for item in user_history]
    history_ratings = [item['rating'] for item in user_history]
    
    # Reverse the order for better display
    history_titles.reverse()
    history_ratings.reverse()
    
    bars1 = ax1.barh(history_titles, history_ratings, color='lightblue')
    ax1.set_title('User Watch History', fontsize=16)
    ax1.set_xlabel('Rating', fontsize=14)
    ax1.set_xlim(0, 5.5)

    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                ha='left', va='center')

    rec_titles = [item['title'][:30] + '...' if len(item['title']) > 30 else item['title'] for item in recommendations]
    rec_scores = [item['score'] * 5 for item in recommendations]  # Convert back to 5-point scale

    rec_titles.reverse()
    rec_scores.reverse()
    
    bars2 = ax2.barh(rec_titles, rec_scores, color='lightgreen')
    ax2.set_title('Movie Recommendations', fontsize=16)
    ax2.set_xlabel('Predicted Rating', fontsize=14)
    ax2.set_xlim(0, 5.5)

    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('results/' + output_path)
    plt.close()
    
    return 'results/' + output_path

def create_genre_recommendation_distribution(recommendations):
    genre_counts = {}

    for item in recommendations:
        genres = item['genres'].split('|')
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    # pie chart
    plt.figure(figsize=(10, 10))
    labels = [genre for genre, count in sorted_genres]
    sizes = [count for genre, count in sorted_genres]

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.axis('equal')
    plt.title('Genre Distribution in Recommendations', fontsize=16)
    plt.tight_layout()
    
    output_path = 'results/recommendation_genres.png'
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def analyze_recommendation_diversity(user_history, recommendations):

    # Extract genres from user history
    history_genres = set()
    for item in user_history:
        genres = item['genres'].split('|')
        history_genres.update(genres)
    
    # Extract genres from recommendations
    rec_genres = set()
    for item in recommendations:
        genres = item['genres'].split('|')
        rec_genres.update(genres)
    
    # Calculate common and new genres
    common_genres = history_genres.intersection(rec_genres)
    new_genres = rec_genres - history_genres

    plt.figure(figsize=(12, 6))
    
    # Chart data
    categories = ['History Genres', 'Recommendation Genres', 'Common Genres', 'New Genres']
    counts = [len(history_genres), len(rec_genres), len(common_genres), len(new_genres)]

    bars = plt.bar(categories, counts, color=['blue', 'green', 'purple', 'orange'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title('Genre Diversity Analysis', fontsize=16)
    plt.ylabel('Number of Genres', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = 'results/genre_diversity.png'
    plt.savefig(output_path)
    plt.close()

    # Modified to better represent actual diversity
    novelty_ratio = len(new_genres) / len(rec_genres) if rec_genres else 0

    plt.figure(figsize=(14, 8))
    
    # Prepare data
    hist_only = list(history_genres - rec_genres)
    rec_only = list(new_genres)
    common = list(common_genres)

    genre_df = pd.DataFrame({
        'Genre': hist_only + common + rec_only,
        'Category': ['History Only'] * len(hist_only) + 
                   ['Common'] * len(common) + 
                   ['Recommendation Only'] * len(rec_only)
    })

    category_counts = genre_df['Category'].value_counts().reindex(
        ['History Only', 'Common', 'Recommendation Only']
    )

    ax = category_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
    

    plt.title('Genre Distribution between History and Recommendations', fontsize=16)
    plt.ylabel('Number of Genres', fontsize=14)
    plt.xlabel('')

    for i, v in enumerate(category_counts):
        plt.text(i, v/2, str(v), ha='center', fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    detail_path = 'results/genre_distribution_detail.png'
    plt.savefig(detail_path)
    plt.close()
    genre_counts = {}
    for item in recommendations:
        genres = item['genres'].split('|')
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Calculate Gini coefficient
    if not genre_counts:
        diversity = 0
    else:
        values = np.array(list(genre_counts.values()))
        values = values / np.sum(values)
        values.sort()
        
        n = len(values)
        cumulative_values = np.cumsum(values)

        gini = (n + 1 - 2 * np.sum(cumulative_values) / np.sum(values)) / n

        diversity = 1 - gini
    
    return {
        'overlap_genres': list(common_genres),
        'new_genres': list(new_genres),
        'novelty_ratio': novelty_ratio,
        'diversity': diversity,
        'viz_path': output_path,
        'detail_viz_path': detail_path
    }

def visualize_similarity_map(recommendations, movies_df=None, output_path='recommendation_similarity_map.png'):
    # Extract genres from recommendations
    all_genres = set()
    movie_genre_matrix = []
    movie_titles = []
    
    for item in recommendations:
        genres = item['genres'].split('|')
        all_genres.update(genres)
        movie_titles.append(item['title'])

    all_genres = list(all_genres)
    
    # Create one-hot encoding for genres
    for item in recommendations:
        item_genres = set(item['genres'].split('|'))
        genre_vector = [1 if genre in item_genres else 0 for genre in all_genres]
        movie_genre_matrix.append(genre_vector)

    movie_genre_matrix = np.array(movie_genre_matrix)

    if len(movie_genre_matrix) > 2:

        perplexity = min(30, len(movie_genre_matrix)-1)
        if perplexity < 5:
            perplexity = 5  # Minimum perplexity value
            
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
        movie_tsne = tsne.fit_transform(movie_genre_matrix)
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.scatter(movie_tsne[:, 0], movie_tsne[:, 1], s=100, alpha=0.7)

        for i, title in enumerate(movie_titles):
            short_title = title.split('(')[0].strip()
            if len(short_title) > 20:
                short_title = short_title[:17] + '...'
            plt.annotate(short_title, (movie_tsne[i, 0], movie_tsne[i, 1]), 
                        fontsize=9, ha='center', va='bottom')
        
        plt.title('Movie Recommendations Similarity Map (t-SNE)', fontsize=16)
        plt.tight_layout()
        
        output_file = 'results/' + output_path
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    else:
        print("Not enough movies for t-SNE visualization")
        return None

def analyze_ratings_distribution(user_history, recommendations):
    # Extract ratings
    history_ratings = [item['rating'] for item in user_history]
    rec_ratings = [item['score'] * 5 for item in recommendations]  # Convert to 5-star scale

    plt.figure(figsize=(12, 6))

    plt.hist(history_ratings, bins=9, alpha=0.5, label='User History', color='blue', 
             range=(0.5, 5.0))
    plt.hist(rec_ratings, bins=9, alpha=0.5, label='Recommendations', color='green',
             range=(0.5, 5.0))
    
    plt.title('Rating Distribution Comparison', fontsize=16)
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = 'results/rating_distribution_comparison.png'
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def create_interactive_demo(model_path='results/attngraphrec_time_aware_model.pt', 
                         ratings_path='data/ratings.csv', 
                         movies_path='data/movies.csv',
                         tags_path='data/tags.csv',
                         min_ratings=20):
    # Load model and data
    model, ratings_df, movies_df, edge_index, device, user_id_map, movie_id_map, reverse_user_map, reverse_movie_map = load_model_and_data(
        model_path, ratings_path, movies_path, tags_path, min_ratings=min_ratings
    )

    sample_users = random.sample(list(user_id_map.keys()), min(5, len(user_id_map)))
    
    for user_id in sample_users:
        user_idx = user_id_map[user_id]
        
        print(f"\nGenerating recommendations for user {user_id}...\n")
        
        # Get user history
        user_history = get_user_history(ratings_df, user_id, movie_id_map, reverse_movie_map, movies_df, reverse_user_map)

        try:
            top_indices, scores = generate_user_recommendations(model, user_idx, edge_index, device, len(movie_id_map))
        except TypeError:
            print("Using non-graph version of recommendation generation")
            top_indices, scores = generate_user_recommendations(model, user_idx, None, device, len(movie_id_map))

        recommendations = []
        for i, movie_idx in enumerate(top_indices):
            if movie_idx >= len(reverse_movie_map):
                continue
                
            movie_id = reverse_movie_map[movie_idx]
            movie_info = movies_df[movies_df['movieId'] == movie_id]

            if movie_info.empty:
                continue
                
            movie_info = movie_info.iloc[0]

            recommendations.append({
                'movieId': int(movie_id),
                'title': str(movie_info['title']),
                'genres': str(movie_info['genres']),
                'score': float(scores[movie_idx])
            })
        
        # Visualize recommendations
        viz_path = visualize_recommendations(user_history, recommendations, f'user_{user_id}_recommendations.png')
        print(f"Recommendation visualization saved to: {viz_path}")
        
        # Analyze genre distribution
        genre_dist_path = create_genre_recommendation_distribution(recommendations)
        print(f"Genre distribution visualization saved to: {genre_dist_path}")
        
        # Analyze diversity
        diversity_results = analyze_recommendation_diversity(user_history, recommendations)
        print(f"New genres in recommendations: {', '.join(diversity_results['new_genres'])}")
        print(f"Novelty ratio: {diversity_results['novelty_ratio']:.2f}")
        print(f"Genre diversity: {diversity_results['diversity']:.2f}")
        print(f"Diversity analysis saved to: {diversity_results['viz_path']}")
        print(f"Detailed genre distribution saved to: {diversity_results['detail_viz_path']}")

        similarity_map = visualize_similarity_map(recommendations, movies_df)
        if similarity_map:
            print(f"Movie similarity map saved to: {similarity_map}")

        rating_dist_path = analyze_ratings_distribution(user_history, recommendations)
        print(f"Rating distribution comparison saved to: {rating_dist_path}")

        with open(f'results/user_{user_id}_recommendation_results.json', 'w') as f:
            json.dump({
                'user_id': int(user_id),
                'history': user_history,
                'recommendations': recommendations,
                'diversity_metrics': {
                    'new_genres': diversity_results['new_genres'],
                    'novelty_ratio': float(diversity_results['novelty_ratio']),
                    'diversity': float(diversity_results['diversity'])
                }
            }, f, indent=2, cls=NumpyEncoder)
        
        print("\n" + "="*50)

def main():
    os.makedirs('results', exist_ok=True)

    model_path = 'results/attngraphrec_time_aware_model.pt'
    ratings_path = 'data/ratings.csv'
    movies_path = 'data/movies.csv'
    tags_path = 'data/tags.csv'

    min_ratings = 20
    
    required_files = [ratings_path, movies_path]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}")
        print("Please ensure the data files are in the 'data' directory!")
        return

    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        print("Please run the model training script first!")

        alt_model_path = 'attngraphrec_time_aware_model.pt'
        if os.path.exists(alt_model_path):
            print(f"Found model file in current directory: {alt_model_path}")
            model_path = alt_model_path
        else:
            print("No model file found. Please run the training script first.")
            return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()