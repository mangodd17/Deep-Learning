import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx

plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

def load_data(ratings_path, movies_path, tags_path=None):
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    if tags_path:
        tags_df = pd.read_csv(tags_path)
        return ratings_df, movies_df, tags_df
    
    return ratings_df, movies_df

def rating_distribution(ratings_df):

    plt.figure(figsize=(10, 6))
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    bars = plt.bar(rating_counts.index.astype(str), rating_counts.values, color='steelblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 500,
                 f'{height:,}',
                 ha='center', va='bottom', rotation=0)
    
    plt.title('Rating Distribution', fontsize=16)
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/rating_distribution.png')
    plt.close()

def genre_analysis(movies_df):
    """Analyze movie genre distribution"""
    all_genres = []
    for genres in movies_df['genres']:
        all_genres.extend(genres.split('|'))

    genre_counts = Counter(all_genres)

    top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    plt.figure(figsize=(12, 7))

    bars = plt.barh(list(top_genres.keys()), list(top_genres.values()), color='mediumseagreen')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 50, bar.get_y() + bar.get_height()/2., 
                 f'{width:,}',
                 ha='left', va='center')
    
    plt.title('Top 15 Movie Genres', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/genre_distribution.png')
    plt.close()

def user_activity_analysis(ratings_df):
    """Analyze user activity"""
    user_ratings = ratings_df.groupby('userId').size()
    
    plt.figure(figsize=(10, 6))
    plt.hist(user_ratings, bins=50, color='coral', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Ratings per User', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=14)
    plt.ylabel('Number of Users', fontsize=14)
    plt.axvline(user_ratings.mean(), color='red', linestyle='--', 
                label=f'Mean: {user_ratings.mean():.1f}')
    plt.axvline(user_ratings.median(), color='green', linestyle='--', 
                label=f'Median: {user_ratings.median():.1f}')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/user_activity.png')
    plt.close()

def movie_popularity_analysis(ratings_df, movies_df):
    """Analyze movie popularity"""
    # Merge data to get movie titles
    movie_ratings = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    })
    movie_ratings.columns = ['rating_count', 'avg_rating']
    movie_ratings = movie_ratings.reset_index()
    
    # Merge movie titles
    movie_ratings = movie_ratings.merge(movies_df[['movieId', 'title']], on='movieId')

    popular_movies = movie_ratings[movie_ratings['rating_count'] >= 50]
    top_movies_by_count = popular_movies.sort_values('rating_count', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))

    bars = plt.barh(top_movies_by_count['title'].str.slice(0, 30), 
                   top_movies_by_count['rating_count'],
                   color='lightblue')
    

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 50, bar.get_y() + bar.get_height()/2., 
                 f'{width:,}',
                 ha='left', va='center')
    
    plt.title('Top 20 Movies by Number of Ratings', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=14)
    plt.gca().invert_yaxis()  # Reverse order, largest value at top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/movie_popularity.png')
    plt.close()

    top_movies_by_rating = popular_movies.sort_values('avg_rating', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    
    # Create colored bar chart based on average rating
    colors = plt.cm.viridis(top_movies_by_rating['avg_rating'] / 5.0)
    bars = plt.barh(top_movies_by_rating['title'].str.slice(0, 30), 
                   top_movies_by_rating['avg_rating'],
                   color=colors)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2., 
                 f'{width:.2f}',
                 ha='left', va='center')
    
    plt.title('Top 20 Movies by Average Rating (min 50 ratings)', fontsize=16)
    plt.xlabel('Average Rating', fontsize=14)
    plt.xlim(0, 5.5)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/top_rated_movies.png')
    plt.close()

def rating_time_analysis(ratings_df):
    """Analyze rating time patterns"""
    # Convert timestamps to datetime
    ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df['year'] = ratings_df['date'].dt.year
    ratings_df['month'] = ratings_df['date'].dt.month

    yearly_ratings = ratings_df.groupby('year').size()
    
    plt.figure(figsize=(12, 6))
    
    # Plot yearly rating trend
    plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o', markersize=8, 
             color='purple', linewidth=2)
    
    plt.title('Number of Ratings by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Ratings', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_ratings.index, rotation=45)

    for i, v in enumerate(yearly_ratings.values):
        plt.text(yearly_ratings.index[i], v + 500, f'{v:,}', 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/rating_time_trend.png')
    plt.close()

def create_user_item_graph(ratings_df, num_users=100, num_items=100):
    """Create user-item interaction graph visualization"""
    top_users = ratings_df['userId'].value_counts().nlargest(num_users).index
    filtered_ratings = ratings_df[ratings_df['userId'].isin(top_users)]
    
    top_items = filtered_ratings['movieId'].value_counts().nlargest(num_items).index
    filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(top_items)]

    G = nx.Graph()
    for user_id in filtered_ratings['userId'].unique():
        G.add_node(f'U{user_id}', type='user')
    
    for movie_id in filtered_ratings['movieId'].unique():
        G.add_node(f'M{movie_id}', type='movie')

    for _, row in filtered_ratings.iterrows():
        G.add_edge(f'U{row["userId"]}', f'M{row["movieId"]}', weight=row['rating'])

    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 12))
    user_nodes = [node for node in G.nodes() if node.startswith('U')]
    movie_nodes = [node for node in G.nodes() if node.startswith('M')]
    
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='red', 
                           node_size=50, alpha=0.8, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_color='blue', 
                           node_size=30, alpha=0.8, label='Movies')
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    plt.title('User-Item Interaction Graph (Sample)', fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/user_item_graph.png', dpi=300)
    plt.close()

def rating_matrix_sparsity(ratings_df):
    """Visualize rating matrix sparsity"""
    user_ids = sorted(ratings_df['userId'].unique())
    movie_ids = sorted(ratings_df['movieId'].unique())

    num_users = min(100, len(user_ids))
    num_movies = min(500, len(movie_ids))

    top_users = ratings_df['userId'].value_counts().nlargest(num_users).index
    filtered_ratings = ratings_df[ratings_df['userId'].isin(top_users)]
    
    top_movies = filtered_ratings['movieId'].value_counts().nlargest(num_movies).index
    filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(top_movies)]

    # Create sparsity matrix visualization
    plt.figure(figsize=(10, 8))

    plt.scatter(
        filtered_ratings['movieId'].apply(lambda x: movie_ids.index(x)), 
        filtered_ratings['userId'].apply(lambda x: user_ids.index(x)),
        s=1, alpha=0.7, c='black'
    )
    
    plt.title(f'Rating Matrix Sparsity Visualization\n(Sample: {num_users} users × {num_movies} movies)', fontsize=16)
    plt.xlabel('Movies', fontsize=14)
    plt.ylabel('Users', fontsize=14)
    plt.xlim(-5, num_movies+5)
    plt.ylim(-5, num_users+5)

    total_possible = num_users * num_movies
    actual_ratings = len(filtered_ratings)
    sparsity = 1 - (actual_ratings / total_possible)
    
    plt.figtext(0.5, 0.01, f'Sparsity: {sparsity:.4f} ({actual_ratings:,} ratings out of {total_possible:,} possible)',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('results/rating_matrix_sparsity.png', dpi=300)
    plt.close()

def main():

    os.makedirs('results', exist_ok=True)

    ratings_df, movies_df, tags_df = load_data('data/ratings.csv', 'data/movies.csv', 'data/tags.csv')
    
    print("Generating rating distribution chart...")
    rating_distribution(ratings_df)
    
    print("Generating movie genre analysis chart...")
    genre_analysis(movies_df)
    
    print("Generating user activity analysis chart...")
    user_activity_analysis(ratings_df)
    
    print("Generating movie popularity analysis chart...")
    movie_popularity_analysis(ratings_df, movies_df)
    
    print("Generating time trend analysis chart...")
    rating_time_analysis(ratings_df)
    
    print("Generating user-item interaction graph...")
    create_user_item_graph(ratings_df)
    
    print("Generating rating matrix sparsity visualization...")
    rating_matrix_sparsity(ratings_df)
    
    print("Data analysis completed!")

if __name__ == "__main__":
    main()