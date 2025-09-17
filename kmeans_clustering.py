import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import plotly.express as px
import plotly.graph_objects as go

class KMeansClusterer:
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.optimal_k = None
    
    def set_data(self, expression_data, labels_df=None):
        """Set preprocessed data"""
        self.X = expression_data
        
        if labels_df is not None:
            self.true_labels = labels_df['label'].values
            self.sample_names = labels_df['sample'].values
        else:
            self.true_labels = None
            self.sample_names = self.X.columns
        
        print(f"Data set: {self.X.shape}")
        
        return self.X

    def load_preprocessed_data(self, expression_file, labels_file=None):
        """Load preprocessed data from files"""
        expression_data = pd.read_csv(expression_file, index_col=0)
        labels_df = None
        if labels_file:
            labels_df = pd.read_csv(labels_file)
        
        return self.set_data(expression_data, labels_df)
    
    def prepare_data_for_clustering(self, scale=True, n_components=None):
        """Prepare data for clustering"""
        # Transpose so samples are rows
        X_samples = self.X.T
        
        # Scale data
        if scale:
            X_scaled = self.scaler.fit_transform(X_samples)
            print("Data scaled using StandardScaler")
        else:
            X_scaled = X_samples.values
        
        # Optional PCA for dimensionality reduction
        if n_components:
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"PCA applied: {X_scaled.shape}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        self.X_processed = X_scaled
        return X_scaled
    
    def find_optimal_k(self, max_k=15, methods=['elbow', 'silhouette', 'calinski']):
        """Find optimal number of clusters using multiple methods"""
        k_range = range(2, max_k + 1)
        results = {}
        
        for method in methods:
            results[method] = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_processed)
            
            if 'elbow' in methods:
                results['elbow'].append(kmeans.inertia_)
            
            if 'silhouette' in methods:
                sil_score = silhouette_score(self.X_processed, cluster_labels)
                results['silhouette'].append(sil_score)
            
            if 'calinski' in methods:
                cal_score = calinski_harabasz_score(self.X_processed, cluster_labels)
                results['calinski'].append(cal_score)
        
        # Plot results
        n_plots = len(methods)
        plt.figure(figsize=(5 * n_plots, 4))
        
        for i, method in enumerate(methods):
            plt.subplot(1, n_plots, i + 1)
            plt.plot(k_range, results[method], 'bo-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel(method.capitalize() + ' Score')
            plt.title(f'{method.capitalize()} Method')
            plt.grid(True)
            
            # Find optimal k for each method
            if method == 'elbow':
                # Use elbow method (look for the "knee")
                diffs = np.diff(results[method])
                diffs2 = np.diff(diffs)
                optimal_idx = np.argmax(diffs2) + 2  # +2 because of double diff
                optimal_k_method = k_range[optimal_idx]
            elif method == 'silhouette':
                optimal_k_method = k_range[np.argmax(results[method])]
            elif method == 'calinski':
                optimal_k_method = k_range[np.argmax(results[method])]
            
            plt.axvline(x=optimal_k_method, color='red', linestyle='--', 
                       label=f'Optimal k={optimal_k_method}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('geo_data/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Choose optimal k (majority vote or silhouette score)
        if 'silhouette' in methods:
            self.optimal_k = k_range[np.argmax(results['silhouette'])]
        else:
            self.optimal_k = k_range[np.argmax(results[methods[0]])]
        
        print(f"Optimal k selected: {self.optimal_k}")
        
        return results, self.optimal_k
    
    def perform_kmeans_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.X_processed)
        
        print(f"K-means clustering completed with {n_clusters} clusters")
        
        # Cluster statistics
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique_labels, counts))}")
        
        return self.cluster_labels
    
    def evaluate_clustering(self):
        """Evaluate clustering performance"""
        # Silhouette score
        sil_score = silhouette_score(self.X_processed, self.cluster_labels)
        print(f"Silhouette Score: {sil_score:.3f}")
        
        # Calinski-Harabasz score
        cal_score = calinski_harabasz_score(self.X_processed, self.cluster_labels)
        print(f"Calinski-Harabasz Score: {cal_score:.3f}")
        
        # If true labels are available
        if self.true_labels is not None:
            ari_score = adjusted_rand_score(self.true_labels, self.cluster_labels)
            print(f"Adjusted Rand Index: {ari_score:.3f}")
        
        return sil_score, cal_score
    
    def visualize_clusters(self):
        """Visualize clustering results"""
        # PCA for 2D visualization
        if self.X_processed.shape[1] > 2:
            pca_2d = PCA(n_components=2)
            X_pca = pca_2d.fit_transform(self.X_processed)
            explained_var = pca_2d.explained_variance_ratio_
        else:
            X_pca = self.X_processed
            explained_var = [1, 1]
        
        plt.figure(figsize=(20, 5))
        
        # Plot 1: K-means clusters
        plt.subplot(1, 4, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels, 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        
        # Plot cluster centers
        if hasattr(self, 'kmeans') and self.kmeans.cluster_centers_ is not None:
            if self.kmeans.cluster_centers_.shape[1] > 2:
                centers_pca = pca_2d.transform(self.kmeans.cluster_centers_)
            else:
                centers_pca = self.kmeans.cluster_centers_
            
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=2)
        
        plt.xlabel(f'PC1 ({explained_var[0]:.1%})')
        plt.ylabel(f'PC2 ({explained_var[1]:.1%})')
        plt.title('K-means Clustering Results')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: True labels (if available)
        if self.true_labels is not None:
            plt.subplot(1, 4, 2)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.true_labels, 
                                cmap='plasma', alpha=0.6)
            plt.colorbar(scatter)
            plt.xlabel(f'PC1 ({explained_var[0]:.1%})')
            plt.ylabel(f'PC2 ({explained_var[1]:.1%})')
            plt.title('True Labels')
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Hierarchical clustering dendrogram
        plt.subplot(1, 4, 3)
        # Calculate linkage matrix
        linkage_matrix = linkage(self.X_processed, method='ward')
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30, 
                  leaf_rotation=90, leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        # Plot 4: Cluster composition heatmap
        plt.subplot(1, 4, 4)
        if self.true_labels is not None:
            # Create confusion matrix between true labels and clusters
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.true_labels, self.cluster_labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Clusters')
            plt.ylabel('True Labels')
            plt.title('Cluster vs True Label Confusion Matrix')
        else:
            # Show cluster sizes
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            plt.bar(unique_labels, counts)
            plt.xlabel('Cluster')
            plt.ylabel('Number of Samples')
            plt.title('Cluster Sizes')
        
        plt.tight_layout()
        plt.savefig('geo_data/clustering_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plot(self):
        """Create interactive 3D plot using plotly"""
        if self.X_processed.shape[1] < 3:
            # Apply PCA for 3D visualization
            pca_3d = PCA(n_components=3)
            X_3d = pca_3d.fit_transform(self.X_processed)
            explained_var = pca_3d.explained_variance_ratio_
        else:
            X_3d = self.X_processed[:, :3]
            explained_var = [1/3, 1/3, 1/3]  # Placeholder
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add scatter points
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        unique_clusters = np.unique(self.cluster_labels)
        
        for i, cluster in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster
            fig.add_trace(go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode='markers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=5,
                    opacity=0.7
                ),
                name=f'Cluster {cluster}',
                text=[f'Sample: {name}' for name in np.array(self.sample_names)[mask]],
                hovertemplate='<b>%{text}</b><br>PC1: %{x}<br>PC2: %{y}<br>PC3: %{z}<extra></extra>'
            ))
        
        # Add cluster centers if available
        if hasattr(self, 'kmeans') and self.kmeans.cluster_centers_ is not None:
            if self.kmeans.cluster_centers_.shape[1] >= 3:
                centers_3d = self.kmeans.cluster_centers_[:, :3]
            elif self.kmeans.cluster_centers_.shape[1] < 3 and self.X_processed.shape[1] >= 3:
                centers_3d = pca_3d.transform(self.kmeans.cluster_centers_)
            else:
                centers_3d = None
            
            if centers_3d is not None:
                fig.add_trace(go.Scatter3d(
                    x=centers_3d[:, 0],
                    y=centers_3d[:, 1],
                    z=centers_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=15,
                        symbol='x'
                    ),
                    name='Cluster Centers'
                ))
        
        fig.update_layout(
            title='Interactive 3D Clustering Visualization',
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.1%})' if len(explained_var) > 0 else 'Dim 1',
                yaxis_title=f'PC2 ({explained_var[1]:.1%})' if len(explained_var) > 1 else 'Dim 2',
                zaxis_title=f'PC3 ({explained_var[2]:.1%})' if len(explained_var) > 2 else 'Dim 3'
            ),
            width=800,
            height=600
        )
        
        fig.write_html('geo_data/interactive_clustering_plot.html')
        fig.show()
        
        print("Interactive plot saved as 'interactive_clustering_plot.html'")
    
    def analyze_cluster_characteristics(self):
        """Analyze characteristics of each cluster"""
        # Create DataFrame with cluster assignments
        cluster_df = pd.DataFrame({
            'sample': self.sample_names,
            'cluster': self.cluster_labels
        })
        
        if self.true_labels is not None:
            cluster_df['true_label'] = self.true_labels
        
        # Statistical analysis
        print("=== Cluster Analysis ===")
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            print(f"\nCluster {cluster_id} (n={cluster_size}):")
            
            if self.true_labels is not None:
                # Distribution of true labels in this cluster
                true_labels_in_cluster = self.true_labels[cluster_mask]
                unique_true, counts_true = np.unique(true_labels_in_cluster, return_counts=True)
                print(f"  True label distribution: {dict(zip(unique_true, counts_true))}")
            
            # Statistical summary of gene expression
            cluster_expression = self.X.T[cluster_mask]  # Samples x Genes
            mean_expression = cluster_expression.mean(axis=0)
            
            # Find top differentially expressed genes
            overall_mean = self.X.T.mean(axis=0)
            fold_change = mean_expression / overall_mean
            
            # Top upregulated genes
            top_up_genes = fold_change.nlargest(10)
            print(f"  Top upregulated genes: {list(top_up_genes.index[:5])}")
            
            # Top downregulated genes
            top_down_genes = fold_change.nsmallest(10)
            print(f"  Top downregulated genes: {list(top_down_genes.index[:5])}")
        
        # Save cluster assignments
        cluster_df.to_csv('geo_data/cluster_assignments.csv', index=False)
        
        return cluster_df
    
    def compare_clustering_methods(self):
        """Compare K-means with other clustering methods"""
        from sklearn.cluster import AgglomerativeClustering, DBSCAN
        from sklearn.mixture import GaussianMixture
        
        methods = {}
        
        # K-means (already performed)
        methods['K-means'] = self.cluster_labels
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=len(np.unique(self.cluster_labels)))
        methods['Hierarchical'] = hierarchical.fit_predict(self.X_processed)
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=len(np.unique(self.cluster_labels)), random_state=42)
        methods['GMM'] = gmm.fit_predict(self.X_processed)
        
        # DBSCAN (eps needs to be tuned)
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            dbscan_labels = dbscan.fit_predict(self.X_processed)
            methods['DBSCAN'] = dbscan_labels
        except:
            print("DBSCAN failed - try adjusting eps parameter")
        
        # Compare methods
        comparison_df = pd.DataFrame(methods)
        comparison_df['sample'] = self.sample_names
        
        if self.true_labels is not None:
            comparison_df['true_label'] = self.true_labels
            
            # Calculate ARI for each method
            print("\n=== Clustering Method Comparison (Adjusted Rand Index) ===")
            for method in ['K-means', 'Hierarchical', 'GMM', 'DBSCAN']:
                if method in comparison_df.columns:
                    ari = adjusted_rand_score(self.true_labels, comparison_df[method])
                    print(f"{method}: {ari:.3f}")
        
        # Visualize comparison
        if len(methods) > 1:
            n_methods = len(methods)
            plt.figure(figsize=(5 * n_methods, 4))
            
            # PCA for visualization
            pca_2d = PCA(n_components=2)
            X_pca = pca_2d.fit_transform(self.X_processed)
            
            for i, (method, labels) in enumerate(methods.items()):
                plt.subplot(1, n_methods, i + 1)
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                    cmap='viridis', alpha=0.6)
                plt.colorbar(scatter)
                plt.title(f'{method} Clustering')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('geo_data/clustering_methods_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save comparison
        comparison_df.to_csv('geo_data/clustering_methods_comparison.csv', index=False)
        
        return comparison_df

# Usage example
if __name__ == "__main__":
    clusterer = KMeansClusterer()
    
    # Load data
    X = clusterer.load_preprocessed_data(
        "geo_data/preprocessed_expression.csv",
        "geo_data/labels.csv"
    )
    
    # Prepare data
    X_processed = clusterer.prepare_data_for_clustering(scale=True, n_components=50)
    
    # Find optimal k
    results, optimal_k = clusterer.find_optimal_k(max_k=10)
    
    # Perform clustering
    cluster_labels = clusterer.perform_kmeans_clustering(n_clusters=optimal_k)
    
    # Evaluate clustering
    sil_score, cal_score = clusterer.evaluate_clustering()
    
    # Visualize results
    clusterer.visualize_clusters()
    
    # Create interactive plot
    clusterer.create_interactive_plot()
    
    # Analyze cluster characteristics
    cluster_df = clusterer.analyze_cluster_characteristics()
    
    # Compare with other methods
    comparison_df = clusterer.compare_clustering_methods()