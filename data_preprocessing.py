import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
    
    def analyze_metadata(self):
        """Analyze metadata to understand available phenotype information"""
        print("\n" + "="*50)
        print("METADATA ANALYSIS")
        print("="*50)
        
        print("Available columns:")
        for col in self.metadata.columns:
            print(f"  - {col}")
        
        print(f"\nFirst few rows:")
        print(self.metadata.head())
        
        print(f"\nDetailed analysis:")
        for col in self.metadata.columns:
            print(f"\n{col}:")
            if self.metadata[col].dtype == 'object':
                unique_values = self.metadata[col].unique()[:10]  # Show first 10 unique values
                print(f"  Unique values ({len(self.metadata[col].unique())} total): {unique_values}")
            else:
                print(f"  Type: {self.metadata[col].dtype}")
                print(f"  Range: {self.metadata[col].min()} to {self.metadata[col].max()}")
        
        return self.metadata
    
    def create_labels_from_metadata(self, label_column=None, strategy='auto'):
        """Create labels from metadata for classification with improved logic"""
        
        print("\n" + "="*50)
        print("LABEL CREATION")
        print("="*50)
        
        # First, analyze metadata
        self.analyze_metadata()
        
        labels = None
        strategy_used = None
        
        if label_column and label_column in self.metadata.columns:
            print(f"\nUsing specified column: {label_column}")
            labels = self.metadata[label_column].values
            strategy_used = f"specified_column_{label_column}"
            
        else:
            print(f"\nTrying automatic label detection strategies...")
            
            # Strategy 1: Look for common phenotype columns
            phenotype_columns = ['phenotype', 'disease', 'condition', 'treatment', 'group', 
                               'class', 'type', 'status', 'outcome']
            
            for col in phenotype_columns:
                if col in self.metadata.columns:
                    unique_vals = self.metadata[col].nunique()
                    if unique_vals > 1 and unique_vals <= 10:  # Reasonable number of classes
                        labels = self.metadata[col].values
                        strategy_used = f"phenotype_column_{col}"
                        print(f"Found phenotype column: {col}")
                        break
            
            # Strategy 2: Parse characteristics column
            if labels is None and 'characteristics' in self.metadata.columns:
                print("Trying to parse characteristics column...")
                labels = self._parse_characteristics_column()
                if labels is not None:
                    strategy_used = "characteristics_parsing"
            
            # Strategy 3: Parse title/description columns
            if labels is None:
                for col in ['title', 'description', 'sample_title', 'source_name']:
                    if col in self.metadata.columns:
                        print(f"Trying to parse {col} column...")
                        labels = self._parse_text_column(self.metadata[col])
                        if labels is not None:
                            strategy_used = f"text_parsing_{col}"
                            break
            
            # Strategy 4: Numeric patterns in sample names
            if labels is None:
                print("Trying numeric patterns in sample names...")
                labels = self._extract_numeric_patterns()
                if labels is not None:
                    strategy_used = "numeric_patterns"
            
            # Strategy 5: Binary split as last resort
            if labels is None:
                print("Using binary split as fallback...")
                n_samples = len(self.metadata)
                labels = ['group_A'] * (n_samples // 2) + ['group_B'] * (n_samples - n_samples // 2)
                strategy_used = "binary_split_fallback"
        
        # Validate labels
        unique_labels = list(set(labels))
        print(f"\nStrategy used: {strategy_used}")
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        if len(unique_labels) < 2:
            raise ValueError(f"Only found {len(unique_labels)} unique label(s). Classification requires at least 2 classes.")
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(labels)
        self.label_names = self.label_encoder.classes_
        self.original_labels = labels
        
        # Print label distribution
        label_counts = pd.Series(labels).value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        print(f"Encoded labels: {np.unique(self.labels, return_counts=True)}")
        
        return self.labels
    
    def _parse_characteristics_column(self):
        """Parse characteristics column to extract phenotype information"""
        labels = []
        
        for char_list in self.metadata['characteristics']:
            label = 'unknown'
            
            if isinstance(char_list, str):
                char_lower = char_list.lower()
                
                # Look for disease patterns
                if any(word in char_lower for word in ['cancer', 'tumor', 'carcinoma', 'malignant']):
                    label = 'disease'
                elif any(word in char_lower for word in ['normal', 'healthy', 'control']):
                    label = 'control'
                elif any(word in char_lower for word in ['treated', 'treatment', 'drug']):
                    label = 'treated'
                elif any(word in char_lower for word in ['untreated', 'mock', 'vehicle']):
                    label = 'untreated'
                
                # Look for specific patterns like "group: A" or "condition: X"
                patterns = [
                    r'group[:\s]+([A-Za-z0-9]+)',
                    r'condition[:\s]+([A-Za-z0-9]+)',
                    r'treatment[:\s]+([A-Za-z0-9]+)',
                    r'phenotype[:\s]+([A-Za-z0-9]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, char_lower)
                    if match:
                        label = match.group(1)
                        break
                
                # Look for binary indicators
                if '1=yes' in char_lower and '0=no' in char_lower:
                    if ': 1' in char_list:
                        label = 'positive'
                    elif ': 0' in char_list:
                        label = 'negative'
            
            labels.append(label)
        
        # Check if we found meaningful labels
        unique_labels = set(labels)
        if len(unique_labels) > 1 and 'unknown' not in unique_labels:
            return labels
        elif len(unique_labels) > 1 and len([l for l in labels if l != 'unknown']) > 1:
            return labels
        else:
            return None
    
    def _parse_text_column(self, text_series):
        """Parse text column to extract phenotype information"""
        labels = []
        
        for text in text_series:
            label = 'unknown'
            
            if isinstance(text, str):
                text_lower = text.lower()
                
                # Look for common patterns
                if any(word in text_lower for word in ['control', 'normal', 'healthy']):
                    label = 'control'
                elif any(word in text_lower for word in ['disease', 'patient', 'cancer', 'tumor']):
                    label = 'disease'
                elif any(word in text_lower for word in ['treated', 'drug', 'compound']):
                    label = 'treated'
                elif any(word in text_lower for word in ['untreated', 'mock', 'vehicle']):
                    label = 'untreated'
                
                # Look for time points
                time_match = re.search(r'(\d+)\s*(h|hour|d|day|w|week|m|month)', text_lower)
                if time_match:
                    label = f"time_{time_match.group(1)}{time_match.group(2)[0]}"
                
                # Look for dose information
                dose_match = re.search(r'(\d+)\s*(mg|μg|ng|μm|mm|nm)', text_lower)
                if dose_match:
                    label = f"dose_{dose_match.group(1)}{dose_match.group(2)}"
            
            labels.append(label)
        
        # Check if we found meaningful labels
        unique_labels = set(labels)
        if len(unique_labels) > 1 and len([l for l in labels if l != 'unknown']) >= len(labels) * 0.5:
            return labels
        else:
            return None
    
    def _extract_numeric_patterns(self):
        """Extract numeric patterns from sample names/IDs"""
        if 'sample_id' in self.metadata.columns:
            sample_col = 'sample_id'
        elif 'sample' in self.metadata.columns:
            sample_col = 'sample'
        else:
            # Use index
            sample_names = self.metadata.index.astype(str)
            return self._extract_patterns_from_names(sample_names)
        
        return self._extract_patterns_from_names(self.metadata[sample_col])
    
    def _extract_patterns_from_names(self, names):
        """Extract patterns from sample names"""
        labels = []
        
        for name in names:
            name_str = str(name)
            
            # Look for patterns like GSM123_1, GSM123_2, etc.
            pattern = re.search(r'_(\d+)$', name_str)
            if pattern:
                num = int(pattern.group(1))
                labels.append(f"group_{num % 3}")  # Create 3 groups
                continue
            
            # Look for patterns in the middle
            pattern = re.search(r'(\d+)', name_str)
            if pattern:
                num = int(pattern.group(1))
                labels.append(f"group_{num % 3}")  # Create 3 groups
                continue
                
            labels.append('group_0')
        
        # Check if we have multiple groups
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            return labels
        else:
            return None
    
    def interactive_label_creation(self):
        """Interactive label creation when automatic methods fail"""
        print("\n" + "="*50)
        print("INTERACTIVE LABEL CREATION")
        print("="*50)
        
        print("Automatic label detection failed. Let's create labels manually.")
        print(f"You have {len(self.metadata)} samples.")
        
        print("\nSample information:")
        for i, (idx, row) in enumerate(self.metadata.iterrows()):
            print(f"{i}: {dict(row)}")
        
        print("\nOptions:")
        print("1. Binary split (first half vs second half)")
        print("2. Groups of 3 (A, B, C, A, B, C, ...)")
        print("3. Manual entry")
        
        choice = input("\nChoose option (1, 2, or 3): ").strip()
        
        if choice == '1':
            n_samples = len(self.metadata)
            labels = ['group_A'] * (n_samples // 2) + ['group_B'] * (n_samples - n_samples // 2)
        elif choice == '2':
            labels = [f"group_{chr(65 + i % 3)}" for i in range(len(self.metadata))]  # A, B, C
        elif choice == '3':
            labels = []
            unique_groups = set()
            for i in range(len(self.metadata)):
                while True:
                    label = input(f"Enter label for sample {i}: ").strip()
                    if label:
                        labels.append(label)
                        unique_groups.add(label)
                        break
                    else:
                        print("Please enter a non-empty label.")
        else:
            print("Invalid choice. Using binary split.")
            n_samples = len(self.metadata)
            labels = ['group_A'] * (n_samples // 2) + ['group_B'] * (n_samples - n_samples // 2)
        
        return labels
    
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
            
            # Add labels to points if we have original labels
            if hasattr(self, 'original_labels'):
                for i, label in enumerate(self.original_labels):
                    plt.annotate(label, (pca_result[i, 0], pca_result[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA Visualization')
        
        plt.tight_layout()
        plt.savefig('geo_data/data_quality_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example with improved error handling
if __name__ == "__main__":
    preprocessor = GEODataPreprocessor()
    
    try:
        # Load data
        expression_data, metadata = preprocessor.load_data(
            "geo_data/expression_matrix.csv",
            "geo_data/sample_metadata.csv"
        )
        
        # Preprocess data
        expression_data = preprocessor.handle_missing_values()
        expression_data = preprocessor.filter_low_variance_genes()
        expression_data = preprocessor.normalize_data(method='zscore')
        
        # Create labels with improved logic
        try:
            labels = preprocessor.create_labels_from_metadata()
        except ValueError as e:
            print(f"Automatic label creation failed: {e}")
            print("Switching to interactive mode...")
            manual_labels = preprocessor.interactive_label_creation()
            labels = preprocessor.label_encoder.fit_transform(manual_labels)
            preprocessor.labels = labels
            preprocessor.label_names = preprocessor.label_encoder.classes_
            preprocessor.original_labels = manual_labels
        
        # Plot data distribution
        preprocessor.plot_data_distribution()
        
        # Save preprocessed data
        expression_data.to_csv("geo_data/preprocessed_expression.csv")
        
        # Create proper labels DataFrame
        sample_names = metadata.get('sample_id', metadata.index).values
        labels_df = pd.DataFrame({
            'sample': sample_names,
            'label': labels
        })
        labels_df.to_csv("geo_data/labels.csv", index=False)
        
        print(f"\nData preprocessing completed successfully!")
        print(f"Expression data shape: {expression_data.shape}")
        print(f"Labels shape: {labels_df.shape}")
        print(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error occurred during preprocessing: {e}")
        print("Please check your data files and ensure they contain valid data.")