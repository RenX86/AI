# GEO Database Analysis Workflow: SVM Classification and K-means Clustering on Windows

## Overview
This workflow guides you through downloading genomic data from the Gene Expression Omnibus (GEO) database and performing machine learning analysis using Support Vector Machines (SVM) for classification and K-means for clustering.

## Prerequisites

### Software Requirements
1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
2. **Git** (optional) - Download from [git-scm.com](https://git-scm.com/)
3. **Text Editor/IDE** - VS Code, PyCharm, or Jupyter Notebook

### Hardware Requirements
- Minimum 8GB RAM (16GB recommended for large datasets)
- 10GB+ free disk space
- Stable internet connection

## Step 1: Environment Setup

### 1.1 Install Python and Package Manager
```bash
# Verify Python installation
python --version

# Upgrade pip
python -m pip install --upgrade pip
```

### 1.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv geo_analysis_env

# Activate virtual environment (Windows)
geo_analysis_env\Scripts\activate
```

### 1.3 Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install GEOparse
pip install requests beautifulsoup4
pip install jupyter
pip install plotly
pip install biopython
```

## Step 2: Data Download from GEO Database

### 2.1 Understanding GEO Database Structure
- **GDS (Gene Expression Dataset)**: Curated datasets
- **GSE (Gene Expression Series)**: Original submitter-supplied records
- **GSM (Gene Expression Sample)**: Individual sample records
- **GPL (Gene Expression Platform)**: Platform/technology information

### 2.2 Python Script for Data Download

Create `geo_downloader.py`:

```python
import GEOparse
import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
import gzip

class GEODataDownloader:
    def __init__(self, output_dir="geo_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_gse_dataset(self, gse_id):
        """Download GSE dataset using GEOparse"""
        print(f"Downloading {gse_id}...")
        try:
            gse = GEOparse.get_GEO(geo=gse_id, destdir=self.output_dir)
            return gse
        except Exception as e:
            print(f"Error downloading {gse_id}: {e}")
            return None
    
    def extract_expression_matrix(self, gse):
        """Extract expression matrix from GSE object"""
        # Get the first platform (assumes single platform)
        gpl_name = list(gse.gpls.keys())[0]
        gpl = gse.gpls[gpl_name]
        
        # Extract sample data
        samples_data = []
        sample_names = []
        
        for gsm_name, gsm in gse.gsms.items():
            sample_names.append(gsm_name)
            samples_data.append(gsm.table['VALUE'].values)
        
        # Create expression matrix
        expression_matrix = pd.DataFrame(samples_data).T
        expression_matrix.columns = sample_names
        expression_matrix.index = gse.gsms[sample_names[0]].table.index
        
        return expression_matrix
    
    def get_sample_metadata(self, gse):
        """Extract sample metadata"""
        metadata = []
        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                'sample_id': gsm_name,
                'title': gsm.metadata.get('title', [''])[0],
                'characteristics': gsm.metadata.get('characteristics_ch1', []),
                'source': gsm.metadata.get('source_name_ch1', [''])[0]
            }
            metadata.append(sample_info)
        
        return pd.DataFrame(metadata)

# Usage example
if __name__ == "__main__":
    downloader = GEODataDownloader()
    
    # Example: Download a cancer dataset
    gse_id = "GSE2034"  # Breast cancer dataset
    gse = downloader.download_gse_dataset(gse_id)
    
    if gse:
        expression_data = downloader.extract_expression_matrix(gse)
        metadata = downloader.get_sample_metadata(gse)
        
        # Save data
        expression_data.to_csv("geo_data/expression_matrix.csv")
        metadata.to_csv("geo_data/sample_metadata.csv", index=False)
        
        print(f"Expression matrix shape: {expression_data.shape}")
        print(f"Metadata shape: {metadata.shape}")
```

### 2.3 Alternative Direct Download Method

For specific datasets, you can download directly:

```python
def download_geo_direct(geo_id, output_dir="geo_data"):
    """Direct download from GEO FTP"""
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    
    # Construct URL based on GEO ID
    series_num = geo_id[:3] + "nnn"  # e.g., GSE2034 -> GSEnnn
    url = f"{base_url}{series_num}/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"
    
    filename = f"{output_dir}/{geo_id}_series_matrix.txt.gz"
    
    try:
        urlretrieve(url, filename)
        print(f"Downloaded {filename}")
        
        # Extract and read
        with gzip.open(filename, 'rt') as f:
            content = f.read()
        
        # Save extracted file
        with open(filename.replace('.gz', ''), 'w') as f:
            f.write(content)
            
        return filename.replace('.gz', '')
    except Exception as e:
        print(f"Error downloading {geo_id}: {e}")
        return None
```

## Step 3: Data Preprocessing

### 3.1 Data Cleaning and Preprocessing Script

Create `data_preprocessing.py`:

```python
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
```

## Step 4: SVM Classification

### 4.1 SVM Implementation Script

Create `svm_classification.py`:

```python
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class SVMClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_selector = SelectKBest(f_classif)
        self.scaler = StandardScaler()
    
    def load_preprocessed_data(self, expression_file, labels_file):
        """Load preprocessed data"""
        self.X = pd.read_csv(expression_file, index_col=0)
        labels_df = pd.read_csv(labels_file)
        self.y = labels_df['label'].values
        self.sample_names = labels_df['sample'].values
        
        print(f"Loaded data: {self.X.shape}")
        print(f"Label distribution: {np.bincount(self.y)}")
        
        return self.X, self.y
    
    def feature_selection(self, k=1000):
        """Select top k features based on univariate statistics"""
        self.feature_selector.set_params(k=min(k, self.X.shape[0]))
        X_selected = self.feature_selector.fit_transform(self.X.T, self.y)
        
        # Get selected feature names
        selected_features = self.X.index[self.feature_selector.get_support()]
        
        print(f"Selected {X_selected.shape[1]} features out of {self.X.shape[0]}")
        
        return X_selected, selected_features
    
    def split_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        # Transpose X so samples are rows
        X_transposed = self.X.T
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_transposed, self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def hyperparameter_tuning(self, cv_folds=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        # Define parameter grid
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        # Create pipeline with feature selection and scaling
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=min(1000, self.X_train.shape[1]))),
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return self.model
    
    def train_model(self):
        """Train SVM model with best parameters"""
        if self.model is None:
            # Use default parameters if no tuning was performed
            self.model = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=min(1000, self.X_train.shape[1]))),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
            ])
        
        self.model.fit(self.X_train, self.y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC curve (for binary classification)
        if len(np.unique(self.y)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            
            plt.subplot(1, 3, 2)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        
        # Feature importance (for linear kernel)
        if hasattr(self.model.named_steps['svm'], 'coef_'):
            feature_importance = np.abs(self.model.named_steps['svm'].coef_[0])
            top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
            
            plt.subplot(1, 3, 3)
            plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
            plt.xlabel('Feature Importance (|Coefficient|)')
            plt.title('Top 20 Most Important Features')
            plt.ylabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig('geo_data/svm_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_pred_proba
    
    def save_model(self, filename='svm_model.joblib'):
        """Save trained model"""
        joblib.dump(self.model, f"geo_data/{filename}")
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='svm_model.joblib'):
        """Load trained model"""
        self.model = joblib.load(f"geo_data/{filename}")
        print(f"Model loaded from {filename}")

# Usage example
if __name__ == "__main__":
    svm_classifier = SVMClassifier()
    
    # Load data
    X, y = svm_classifier.load_preprocessed_data(
        "geo_data/preprocessed_expression.csv",
        "geo_data/labels.csv"
    )
    
    # Split data
    X_train, X_test, y_train, y_test = svm_classifier.split_data()
    
    # Hyperparameter tuning
    model = svm_classifier.hyperparameter_tuning()
    
    # Train model
    svm_classifier.train_model()
    
    # Evaluate model
    y_pred, y_pred_proba = svm_classifier.evaluate_model()
    
    # Save model
    svm_classifier.save_model()
```

## Step 5: K-means Clustering

### 5.1 K-means Implementation Script

Create `kmeans_clustering.py`:

```python
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
    
    def load_preprocessed_data(self, expression_file, labels_file=None):
        """Load preprocessed data"""
        self.X = pd.read_csv(expression_file, index_col=0)
        
        if labels_file:
            labels_df = pd.read_csv(labels_file)
            self.true_labels = labels_df['label'].values
            self.sample_names = labels_df['sample'].values
        else:
            self.true_labels = None
            self.sample_names = self.X.columns
        
        print(f"Loaded data: {self.X.shape}")
        
        return self.X
    
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
```

## Step 6: Complete Analysis Pipeline

### 6.1 Main Pipeline Script

Create `main_analysis_pipeline.py`:

```python
import os
import sys
import argparse
import logging
from datetime import datetime

# Import custom modules
from geo_downloader import GEODataDownloader
from data_preprocessing import GEODataPreprocessor
from svm_classification import SVMClassifier
from kmeans_clustering import KMeansClusterer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geo_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GEOAnalysisPipeline:
    def __init__(self, output_dir="geo_analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.downloader = GEODataDownloader(output_dir)
        self.preprocessor = GEODataPreprocessor()
        self.classifier = SVMClassifier()
        self.clusterer = KMeansClusterer()
        
        logger.info(f"Pipeline initialized. Results will be saved to: {output_dir}")
    
    def run_complete_analysis(self, gse_id, perform_svm=True, perform_clustering=True):
        """Run complete analysis pipeline"""
        logger.info(f"Starting analysis for dataset: {gse_id}")
        
        try:
            # Step 1: Download data
            logger.info("Step 1: Downloading data from GEO...")
            gse = self.downloader.download_gse_dataset(gse_id)
            
            if gse is None:
                logger.error(f"Failed to download {gse_id}")
                return False
            
            expression_data = self.downloader.extract_expression_matrix(gse)
            metadata = self.downloader.get_sample_metadata(gse)
            
            # Save raw data
            expression_data.to_csv(f"{self.output_dir}/expression_matrix.csv")
            metadata.to_csv(f"{self.output_dir}/sample_metadata.csv", index=False)
            
            logger.info(f"Data downloaded successfully. Shape: {expression_data.shape}")
            
            # Step 2: Preprocessing
            logger.info("Step 2: Preprocessing data...")
            self.preprocessor.expression_data = expression_data
            self.preprocessor.metadata = metadata
            
            # Clean and preprocess
            processed_data = self.preprocessor.handle_missing_values()
            processed_data = self.preprocessor.filter_low_variance_genes()
            processed_data = self.preprocessor.normalize_data(method='zscore')
            
            # Create labels
            labels = self.preprocessor.create_labels_from_metadata()
            
            # Plot data quality
            self.preprocessor.plot_data_distribution()
            
            # Save preprocessed data
            processed_data.to_csv(f"{self.output_dir}/preprocessed_expression.csv")
            pd.DataFrame({
                'sample': metadata['sample_id'], 
                'label': labels
            }).to_csv(f"{self.output_dir}/labels.csv", index=False)
            
            logger.info("Preprocessing completed successfully")
            
            # Step 3: SVM Classification
            if perform_svm:
                logger.info("Step 3: Running SVM classification...")
                
                # Load data
                self.classifier.load_preprocessed_data(
                    f"{self.output_dir}/preprocessed_expression.csv",
                    f"{self.output_dir}/labels.csv"
                )
                
                # Split data
                self.classifier.split_data()
                
                # Hyperparameter tuning
                self.classifier.hyperparameter_tuning()
                
                # Train model
                self.classifier.train_model()
                
                # Evaluate model
                y_pred, y_pred_proba = self.classifier.evaluate_model()
                
                # Save model
                self.classifier.save_model(f"svm_model_{gse_id}.joblib")
                
                logger.info("SVM classification completed successfully")
            
            # Step 4: K-means Clustering
            if perform_clustering:
                logger.info("Step 4: Running K-means clustering...")
                
                # Load data
                self.clusterer.load_preprocessed_data(
                    f"{self.output_dir}/preprocessed_expression.csv",
                    f"{self.output_dir}/labels.csv"
                )
                
                # Prepare data
                self.clusterer.prepare_data_for_clustering(scale=True, n_components=50)
                
                # Find optimal k
                results, optimal_k = self.clusterer.find_optimal_k(max_k=10)
                
                # Perform clustering
                cluster_labels = self.clusterer.perform_kmeans_clustering(n_clusters=optimal_k)
                
                # Evaluate clustering
                self.clusterer.evaluate_clustering()
                
                # Visualize results
                self.clusterer.visualize_clusters()
                self.clusterer.create_interactive_plot()
                
                # Analyze clusters
                cluster_df = self.clusterer.analyze_cluster_characteristics()
                
                # Compare methods
                comparison_df = self.clusterer.compare_clustering_methods()
                
                logger.info("K-means clustering completed successfully")
            
            # Step 5: Generate report
            logger.info("Step 5: Generating analysis report...")
            self.generate_analysis_report(gse_id, expression_data.shape, 
                                        perform_svm, perform_clustering)
            
            logger.info(f"Complete analysis finished successfully for {gse_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return False
    
    def generate_analysis_report(self, gse_id, data_shape, performed_svm, performed_clustering):
        """Generate a comprehensive analysis report"""
        report_content = f"""
# GEO Analysis Report: {gse_id}

## Analysis Summary
- **Dataset**: {gse_id}
- **Data Shape**: {data_shape[0]} genes × {data_shape[1]} samples
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **SVM Classification**: {'✓ Performed' if performed_svm else '✗ Skipped'}
- **K-means Clustering**: {'✓ Performed' if performed_clustering else '✗ Skipped'}

## Files Generated

### Raw Data
- `expression_matrix.csv` - Raw gene expression matrix
- `sample_metadata.csv` - Sample metadata and annotations

### Preprocessed Data
- `preprocessed_expression.csv` - Cleaned and normalized expression data
- `labels.csv` - Sample labels for supervised learning
- `data_quality_plots.png` - Data quality assessment plots

### Classification Results (if performed)
- `svm_model_{gse_id}.joblib` - Trained SVM model
- `svm_evaluation.png` - Model performance plots

### Clustering Results (if performed)
- `optimal_k_analysis.png` - Optimal cluster number analysis
- `clustering_visualization.png` - Cluster visualization plots
- `interactive_clustering_plot.html` - Interactive 3D cluster plot
- `cluster_assignments.csv` - Sample cluster assignments
- `clustering_methods_comparison.png` - Comparison of clustering methods
- `clustering_methods_comparison.csv` - Clustering comparison results

## Next Steps

1. **Review Results**: Examine the generated plots and metrics
2. **Biological Interpretation**: Analyze cluster characteristics and differentially expressed genes
3. **Further Analysis**: Consider pathway analysis, gene set enrichment, or validation experiments
4. **Parameter Tuning**: Adjust preprocessing or algorithm parameters if needed

## Log File
- `geo_analysis.log` - Detailed analysis log

---
*Report generated by GEO Analysis Pipeline*
        """
        
        with open(f"{self.output_dir}/analysis_report.md", 'w') as f:
            f.write(report_content)
        
        logger.info("Analysis report generated: analysis_report.md")

def main():
    parser = argparse.ArgumentParser(description='GEO Database Analysis Pipeline')
    parser.add_argument('gse_id', help='GEO Series ID (e.g., GSE2034)')
    parser.add_argument('--output-dir', default='geo_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--skip-svm', action='store_true', 
                       help='Skip SVM classification')
    parser.add_argument('--skip-clustering', action='store_true', 
                       help='Skip K-means clustering')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = GEOAnalysisPipeline(args.output_dir)
    
    # Run analysis
    success = pipeline.run_complete_analysis(
        args.gse_id,
        perform_svm=not args.skip_svm,
        perform_clustering=not args.skip_clustering
    )
    
    if success:
        print(f"\n✓ Analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Check analysis_report.md for summary")
    else:
        print(f"\n✗ Analysis failed. Check geo_analysis.log for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Step 7: Running the Complete Workflow

### 7.1 Quick Start Guide

Create `run_analysis.bat` for Windows:

```batch
@echo off
echo GEO Analysis Pipeline - Quick Start
echo ===================================

REM Activate virtual environment
call geo_analysis_env\Scripts\activate

REM Run analysis with example dataset
python main_analysis_pipeline.py GSE2034 --output-dir results_GSE2034

echo.
echo Analysis completed! Check results_GSE2034 folder for outputs.
pause
```

### 7.2 Example Usage Commands

```bash
# Basic analysis
python main_analysis_pipeline.py GSE2034

# Skip SVM classification
python main_analysis_pipeline.py GSE2034 --skip-svm

# Skip clustering
python main_analysis_pipeline.py GSE2034 --skip-clustering

# Custom output directory
python main_analysis_pipeline.py GSE2034 --output-dir my_analysis_results

# Run individual components
python geo_downloader.py
python data_preprocessing.py
python svm_classification.py
python kmeans_clustering.py
```

## Step 8: Troubleshooting Guide

### 8.1 Common Issues and Solutions

**Issue 1: Download Failures**
- Check internet connection
- Verify GEO ID exists
- Try alternative download methods
- Use VPN if geographic restrictions apply

**Issue 2: Memory Issues**
- Reduce dataset size using gene filtering
- Increase virtual memory
- Use dimensionality reduction (PCA)
- Process data in chunks

**Issue 3: Poor Classification Performance**
- Check label quality and distribution
- Increase feature selection threshold
- Try different preprocessing methods
- Adjust SVM hyperparameters

**Issue 4: Clustering Issues**
- Verify data scaling
- Try different distance metrics
- Adjust cluster number range
- Use different clustering algorithms

### 8.2 Performance Optimization Tips

1. **Data Size Management**
   - Filter low-variance genes early
   - Use PCA for dimensionality reduction
   - Sample large datasets if needed

2. **Algorithm Optimization**
   - Use parallel processing (`n_jobs=-1`)
   - Reduce hyperparameter search space
   - Use early stopping where available

3. **Memory Management**
   - Process data in batches
   - Use memory-mapped arrays
   - Clear unused variables

## Step 9: Advanced Extensions

### 9.1 Additional Analysis Options

- **Pathway Analysis**: Integrate with KEGG/GO databases
- **Differential Expression**: Add DESeq2-style analysis
- **Time Series**: Handle temporal data
- **Multi-omics**: Integrate different data types
- **Survival Analysis**: Add clinical outcome analysis

### 9.2 Visualization Enhancements

- **Heatmaps**: Gene expression heatmaps
- **Network Analysis**: Gene co-expression networks
- **Dimensionality Reduction**: t-SNE, UMAP visualizations
- **Interactive Dashboards**: Streamlit/Dash applications

## Conclusion

This comprehensive workflow provides a complete solution for downloading, analyzing, and visualizing GEO database genomic data using SVM classification and K-means clustering on Windows. The modular design allows for easy customization and extension based on specific research needs.