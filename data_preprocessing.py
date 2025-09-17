import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class GEODataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoder = LabelEncoder()
    
    def load_data(self, expression_file, metadata_file):
        """Load expression and metadata files"""
        self.expression_data = pd.read_csv(expression_file, index_col=0)
        self.metadata = pd.read_csv(metadata_file)
        
        print(f"Loaded expression data: {self.expression_data.shape}")
        print(f"Loaded metadata: {self.metadata.shape}")
        
        return self.expression_data, self.metadata
    
    def handle_missing_values(self, threshold=0.1):
        """Remove genes/samples with too many missing values"""
        # Remove genes with >threshold missing values
        gene_missing_ratio = self.expression_data.isnull().sum(axis=1) / self.expression_data.shape[1]
        genes_to_keep = gene_missing_ratio <= threshold
        
        # Remove samples with >threshold missing values
        sample_missing_ratio = self.expression_data.isnull().sum(axis=0) / self.expression_data.shape[0]
        samples_to_keep = sample_missing_ratio <= threshold
        
        self.expression_data = self.expression_data.loc[genes_to_keep, samples_to_keep]
        
        # Impute remaining missing values
        self.expression_data = pd.DataFrame(
            self.imputer.fit_transform(self.expression_data),
            index=self.expression_data.index,
            columns=self.expression_data.columns
        )
        
        print(f"After cleaning: {self.expression_data.shape}")
        return self.expression_data
    
    def filter_low_variance_genes(self, percentile=25):
        """Remove genes with low variance"""
        gene_variance = self.expression_data.var(axis=1)
        threshold = np.percentile(gene_variance, percentile)
        high_var_genes = gene_variance >= threshold
        
        self.expression_data = self.expression_data.loc[high_var_genes]
        print(f"After variance filtering: {self.expression_data.shape}")
        
        return self.expression_data
    
    def normalize_data(self, method='zscore'):
        """Normalize expression data"""
        if method == 'zscore':
            # Z-score normalization (across samples for each gene)
            self.expression_data = self.expression_data.apply(
                lambda x: (x - x.mean()) / x.std(), axis=1
            )
        elif method == 'standard':
            # Standard scaling (across genes for each sample)
            self.expression_data = pd.DataFrame(
                self.scaler.fit_transform(self.expression_data.T).T,
                index=self.expression_data.index,
                columns=self.expression_data.columns
            )
        
        print(f"Data normalized using {method} method")
        return self.expression_data
    
    def create_labels_from_metadata(self, label_column=None):
        """Create labels from metadata for classification"""
        if label_column and label_column in self.metadata.columns:
            labels = self.metadata[label_column].values
        else:
            # Try to extract labels from characteristics or title
            if 'characteristics' in self.metadata.columns:
                # Parse characteristics to find disease/condition info
                labels = []
                for char_list in self.metadata['characteristics']:
                    # Simple parsing - you may need to customize this
                    if isinstance(char_list, str):
                        if 'cancer' in char_list.lower() or 'tumor' in char_list.lower():
                            labels.append('cancer')
                        elif 'normal' in char_list.lower() or 'healthy' in char_list.lower():
                            labels.append('normal')
                        else:
                            labels.append('unknown')
                    else:
                        labels.append('unknown')
            else:
                # Create dummy labels based on sample names or other criteria
                labels = ['group_' + str(i % 3) for i in range(len(self.metadata))]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        self.labels = encoded_labels
        self.label_names = self.label_encoder.classes_
        
        print(f"Created labels with classes: {self.label_names}")
        return encoded_labels
    
    def plot_data_distribution(self):
        """Plot data distribution for quality check"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Expression distribution
        plt.subplot(1, 3, 1)
        plt.hist(self.expression_data.values.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Expression Value')
        plt.ylabel('Frequency')
        plt.title('Expression Distribution')
        
        # Plot 2: Sample correlation heatmap
        plt.subplot(1, 3, 2)
        sample_corr = self.expression_data.corr()
        sns.heatmap(sample_corr, cmap='coolwarm', center=0)
        plt.title('Sample Correlation')
        
        # Plot 3: PCA visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.expression_data.T)
        
        plt.subplot(1, 3, 3)
        if hasattr(self, 'labels'):
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=self.labels, cmap='viridis')
            plt.colorbar(scatter)
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA Visualization')
        
        plt.tight_layout()
        plt.savefig('geo_data/data_quality_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    preprocessor = GEODataPreprocessor()
    
    # Load data
    expression_data, metadata = preprocessor.load_data(
        "geo_data/expression_matrix.csv",
        "geo_data/sample_metadata.csv"
    )
    
    # Preprocess data
    expression_data = preprocessor.handle_missing_values()
    expression_data = preprocessor.filter_low_variance_genes()
    expression_data = preprocessor.normalize_data(method='zscore')
    
    # Create labels
    labels = preprocessor.create_labels_from_metadata()
    
    # Plot data distribution
    preprocessor.plot_data_distribution()
    
    # Save preprocessed data
    expression_data.to_csv("geo_data/preprocessed_expression.csv")
    pd.DataFrame({'sample': metadata['sample_id'], 'label': labels}).to_csv(
        "geo_data/labels.csv", index=False
    )