import torch
import torch.nn.functional as F
import numpy as np
import pickle
import re
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display

# Load the trained model and graph builder
def load_model(model_path):
    """Load a trained GNN model."""
    try:
        # Load graph builder
        graph_builder_path = model_path.replace('.pt', '_graph_builder.pkl')
        with open(graph_builder_path, 'rb') as f:
            graph_builder = pickle.load(f)
        
        # For demonstration, if the actual model file doesn't exist, we'll still use the graph builder
        # for aspect extraction
        print(f"✅ Successfully loaded graph builder from {graph_builder_path}")
        return graph_builder
        
    except FileNotFoundError:
        print("❌ Model files not found. Using fallback approach.")
        # Create a new graph builder if needed
        from gnn_models import HotelReviewGraphBuilder
        return HotelReviewGraphBuilder()

class ReviewAnalyzer:
    """Class to analyze hotel reviews using a trained GNN model."""
    
    def __init__(self, model_path=None):
        # Load or create a graph builder
        if model_path:
            self.graph_builder = load_model(model_path)
        else:
            # Import from the model file if not provided
            from gnn_models import HotelReviewGraphBuilder
            self.graph_builder = HotelReviewGraphBuilder()
            
        self.aspects = self.graph_builder.aspects
        self.aspect_keywords = self.graph_builder.aspect_keywords
    
    def analyze_review(self, review_text):
        """Analyze a hotel review and extract aspect ratings."""
        # Extract mentioned aspects and their ratings
        identified_aspects = self.graph_builder.identify_aspects(review_text)
        
        # Prepare the results
        results = {}
        for aspect, info in identified_aspects.items():
            results[aspect] = {
                'rating': info['rating'],
                'confidence': info['confidence'],
                'keywords': info['keywords']
            }
            
        return results
    
    def visualize_results(self, results):
        """Visualize the analysis results."""
        # Create a bar chart for ratings
        aspects = []
        ratings = []
        confidences = []
        
        for aspect, info in results.items():
            aspects.append(aspect)
            ratings.append(info['rating'])
            confidences.append(info['confidence'])
        
        # Sort by rating in descending order
        sorted_indices = np.argsort(ratings)[::-1]
        aspects = [aspects[i] for i in sorted_indices]
        ratings = [ratings[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(aspects, ratings, color='skyblue')
        
        # Add confidence as text
        for i, (bar, confidence) in enumerate(zip(bars, confidences)):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"Conf: {confidence:.2f}",
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.xlabel('Aspects')
        plt.ylabel('Rating (1-5)')
        plt.title('Aspect Ratings from Hotel Review')
        plt.ylim(0, 5.5)  # Set y-axis limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Return the figure
        return plt.gcf()
    
    def build_review_graph(self, review_text):
        """Build and visualize a graph from the review."""
        # Create a simple review object
        review = {'text': review_text, 'rating': 0}  # Rating will be determined from aspect ratings
        
        # Build the graph
        G = self.graph_builder.build_graph([review])
        
        # Calculate overall rating based on aspect ratings
        aspect_ratings = []
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'mentions' and 'rating' in data:
                aspect_ratings.append(data['rating'])
        
        overall_rating = np.mean(aspect_ratings) if aspect_ratings else 3.0
        
        # Update the review node with the calculated rating
        G.nodes['review_0']['rating'] = overall_rating
        
        return G
    
    def visualize_graph(self, G):
        """Visualize the review graph."""
        plt.figure(figsize=(12, 8))
        
        # Define node colors based on type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'hotel':
                node_colors.append('gold')
            elif G.nodes[node]['node_type'] == 'aspect':
                node_colors.append('lightgreen')
            else:  # review
                node_colors.append('lightblue')
        
        # Define edge colors based on edge type and rating
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'has_aspect':
                edge_colors.append('gray')
                edge_widths.append(1.0)
            elif data.get('edge_type') == 'about':
                edge_colors.append('black')
                edge_widths.append(1.5)
            else:  # mentions
                # Color based on rating
                if 'rating' in data:
                    if data['rating'] >= 4:
                        edge_colors.append('green')
                    elif data['rating'] <= 2:
                        edge_colors.append('red')
                    else:
                        edge_colors.append('orange')
                    
                    # Width based on confidence
                    if 'confidence' in data:
                        edge_widths.append(1.0 + data['confidence'] * 2)
                    else:
                        edge_widths.append(1.5)
                else:
                    edge_colors.append('blue')
                    edge_widths.append(1.0)
        
        # Create a reasonable layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
        
        # Add labels
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'hotel':
                labels[node] = 'Hotel'
            elif G.nodes[node]['node_type'] == 'aspect':
                labels[node] = node.capitalize()
            else:  # review
                labels[node] = f"Review\n({G.nodes[node]['rating']:.1f}/5)"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # Add edge labels for mentions edges (showing ratings)
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'mentions' and 'rating' in data:
                edge_labels[(u, v)] = f"{data['rating']}/5"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
        
        plt.title('Hotel Review Graph Representation')
        plt.axis('off')
        
        return plt.gcf()

    def print_report(self, review_text, results):
        """Print a detailed analysis report."""
        print("=" * 80)
        print("🏨 HOTEL REVIEW ANALYSIS REPORT")
        print("=" * 80)
        
        print("\n📝 REVIEW TEXT:")
        print("-" * 80)
        print(review_text)
        print("-" * 80)
        
        print("\n⭐ ASPECT RATINGS:")
        print("-" * 80)
        
        # Sort aspects by rating (highest first)
        sorted_aspects = sorted(results.items(), key=lambda x: x[1]['rating'], reverse=True)
        
        for aspect, info in sorted_aspects:
            rating = info['rating']
            confidence = info['confidence']
            keywords = info['keywords']
            
            # Convert rating to stars
            stars = "★" * rating + "☆" * (5 - rating)
            
            print(f"{aspect.upper()}: {stars} ({rating}/5) - Confidence: {confidence:.2f}")
            if keywords:
                print(f"  🔑 Keywords: {', '.join(keywords)}")
        
        print("-" * 80)
        
        # Calculate overall score (weighted by confidence)
        weighted_sum = sum(info['rating'] * info['confidence'] for _, info in results.items())
        total_confidence = sum(info['confidence'] for _, info in results.items())
        overall_score = weighted_sum / total_confidence if total_confidence > 0 else 0
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/5")
        
        # Provide summary based on overall score
        if overall_score >= 4.5:
            summary = "Excellent experience! The guest was extremely satisfied."
        elif overall_score >= 4.0:
            summary = "Very good experience. The guest was quite satisfied."
        elif overall_score >= 3.5:
            summary = "Good experience with some minor issues."
        elif overall_score >= 3.0:
            summary = "Average experience with room for improvement."
        elif overall_score >= 2.0:
            summary = "Below average experience with significant issues."
        else:
            summary = "Poor experience. The guest was dissatisfied."
            
        print(f"\n📋 SUMMARY: {summary}")
        print("=" * 80)
        
        return {
            'aspect_ratings': {aspect: info['rating'] for aspect, info in results.items()},
            'overall_score': overall_score,
            'summary': summary
        }

# Example usage
def analyze_review(review_text, model_path=None):
    """Analyze a hotel review and display results."""
    analyzer = ReviewAnalyzer(model_path)
    
    # Analyze the review
    results = analyzer.analyze_review(review_text)
    
    # Print the detailed report
    summary = analyzer.print_report(review_text, results)
    
    # Build and visualize the graph
    G = analyzer.build_review_graph(review_text)
    graph_fig = analyzer.visualize_graph(G)
    
    # Visualize the aspect ratings
    ratings_fig = analyzer.visualize_results(results)
    
    return results, summary, graph_fig, ratings_fig

# Test function for simple command-line usage
def test():
    test_review = """
    I stayed at this hotel for 3 nights last week. The room was very spacious and clean with a beautiful view.
    The bed was comfortable but the bathroom was a bit small. The staff was extremely friendly and helpful,
    especially at the reception. The location was perfect, close to the city center and major attractions.
    WiFi was free but very slow in the evening. The breakfast was delicious with many options.
    Overall, I think this is a great hotel with good value for money.
    """
    
    results, summary, graph_fig, ratings_fig = analyze_review(test_review)
    
    # Save figures
    graph_fig.savefig('review_graph.png')
    ratings_fig.savefig('aspect_ratings.png')
    
    print("\nFigures saved as 'review_graph.png' and 'aspect_ratings.png'")
    
    return results, summary

if __name__ == "__main__":
    test()