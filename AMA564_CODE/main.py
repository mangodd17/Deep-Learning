import os
import argparse
import subprocess
import sys
import torch
import importlib.util

def check_dependencies():

    required_packages = [
        'torch', 'torch_geometric', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'sklearn', 'scipy', 'surprise', 'tqdm', 'networkx'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing dependencies: {', '.join(missing_packages)}")
        install = input("Would you like to install these packages automatically? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"Installed {package}")
                except Exception as e:
                    print(f"Failed to install {package}: {e}")
            print("Dependency installation completed, please restart the program")
        return False
    
    return True

def print_header():
    """Print program title"""
    print("\n" + "="*70)
    print(" "*20 + "AttnGraphRec Recommendation System" + " "*20)
    print(" "*10 + "A Hybrid Deep Learning Framework with Attention and Graph Networks" + " "*10)
    print("="*70 + "\n")

def setup_environment():
    """Set up the project environment"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Check if all data files are in the data directory
    data_files = ['data/ratings.csv', 'data/movies.csv', 'data/tags.csv', 'data/links.csv']
    missing_files = [file for file in data_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"Error: Missing data files: {', '.join(missing_files)}")
        print("Please ensure MovieLens dataset files are in the 'data' directory")
        return False
        
    return True

def check_module_exists(module_name, file_path):
    """Check if a Python module exists at the specified path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return False
    return True

def main():
    """Main function"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Set up environment
    if not setup_environment():
        return
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Device: {device}" + (" (GPU acceleration available)" if cuda_available else " (CPU mode only)"))
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AttnGraphRec Recommendation System')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['data_analysis', 'train', 'visualize', 'all'],
                        help='Run mode: data_analysis/train/visualize/all')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--min_ratings', type=int, default=20,
                        help='Minimum number of ratings for a movie to be included')
    
    args = parser.parse_args()
    
    # Check if required modules exist
    required_modules = {
        'data_analysis': 'data_analysis.py',
        'attngraphrec': 'attngraphrec.py',
        'recommendation_visualization': 'recommendation_visualization.py'
    }
    
    missing_modules = [name for name, path in required_modules.items() 
                      if not check_module_exists(name, path)]
    
    if missing_modules:
        print(f"Error: Missing Python modules: {', '.join(missing_modules)}")
        print("Please ensure all Python scripts are in the current directory")
        return
    
    # Check if model file exists
    model_exists = os.path.exists('results/attngraphrec_model.pt')
    
    # Run requested mode
    if args.mode in ['data_analysis', 'all']:
        print("\nStarting data analysis...")
        try:
            from data_analysis import main as run_data_analysis
            run_data_analysis()
            print("Data analysis completed!")
        except Exception as e:
            print(f"Error in data analysis: {e}")
    
    if args.mode in ['train', 'all'] or (args.mode in ['visualize'] and not model_exists):
        if args.mode in ['visualize'] and not model_exists:
            print("\nModel file not found, starting training first...")
        else:
            print("\nStarting model training...")
        
        try:
            # Import and modify attngraphrec to use the specified parameters
            import attngraphrec
            original_main = attngraphrec.main
            
            def modified_main():
                # Save original parameters
                original_min_ratings = 20  # Default value
                original_num_epochs = 50   # Default value
                
                # Modify the module's global variables with our parameters
                if hasattr(attngraphrec, 'min_ratings'):
                    original_min_ratings = attngraphrec.min_ratings
                    attngraphrec.min_ratings = args.min_ratings
                
                if hasattr(attngraphrec, 'num_epochs'):
                    original_num_epochs = attngraphrec.num_epochs
                    attngraphrec.num_epochs = args.epochs
                
                # Run the original main function
                result = original_main()
                
                # Restore original values
                if hasattr(attngraphrec, 'min_ratings'):
                    attngraphrec.min_ratings = original_min_ratings
                
                if hasattr(attngraphrec, 'num_epochs'):
                    attngraphrec.num_epochs = original_num_epochs
                
                return result
            attngraphrec.main = modified_main
            modified_main()
            attngraphrec.main = original_main
            
            print("Model training completed!")
        except Exception as e:
            print(f"Error in model training: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode in ['visualize', 'all']:
        print("\nStarting recommendation visualization...")
        try:
            from recommendation_visualization import main as run_visualization
            import recommendation_visualization
            original_viz_main = recommendation_visualization.main
            
            def modified_viz_main():
                if hasattr(recommendation_visualization, 'min_ratings'):
                    original_min_ratings = recommendation_visualization.min_ratings
                    recommendation_visualization.min_ratings = args.min_ratings
                    result = original_viz_main()
                    recommendation_visualization.min_ratings = original_min_ratings
                    return result
                else:
                    return original_viz_main()
            
            # Replace main function temporarily
            recommendation_visualization.main = modified_viz_main
            
            # Run the modified visualization
            modified_viz_main()
            
            # Restore original main function
            recommendation_visualization.main = original_viz_main
            
            print("Recommendation visualization completed!")
        except Exception as e:
            print(f"Error in recommendation visualization: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nProgram execution completed!")
    print("Result files and visualization charts have been saved in the 'results' directory")

if __name__ == "__main__":
    main()
