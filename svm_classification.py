import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

class SVMClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_selector = SelectKBest(f_classif)
        self.scaler = StandardScaler()
    
    def set_data(self, expression_data, labels_df):
        """Set preprocessed data"""
        self.X = expression_data
        self.y = labels_df['label'].values
        self.sample_names = labels_df['sample'].values
        
        print(f"Data set: {self.X.shape}")
        print(f"Label distribution: {np.bincount(self.y)}")
        
        # Check for data quality issues
        self._check_data_quality()
        
        return self.X, self.y

    def _check_data_quality(self):
        """Check for common data quality issues"""
        # Check for NaN values
        if np.isnan(self.X.values).any():
            print("⚠️ Warning: NaN values detected in features")
            
        # Check for infinite values
        if np.isinf(self.X.values).any():
            print("⚠️ Warning: Infinite values detected in features")
            
        # Check for constant features (zero variance)
        feature_vars = np.var(self.X.values, axis=1)
        constant_features = np.sum(feature_vars == 0)
        if constant_features > 0:
            print(f"⚠️ Warning: {constant_features} constant features detected")
            
        # Check class distribution
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        min_class_size = np.min(class_counts)
        print(f"Minimum class size: {min_class_size}")
        
        if len(unique_classes) < 2:
            raise ValueError("Dataset must contain at least 2 classes")
            
        if min_class_size < 5:
            print(f"⚠️ Warning: Very small class size ({min_class_size}). Consider collecting more data.")

    def load_preprocessed_data(self, expression_file, labels_file):
        """Load preprocessed data from files"""
        expression_data = pd.read_csv(expression_file, index_col=0)
        labels_df = pd.read_csv(labels_file)
        return self.set_data(expression_data, labels_df)
    
    def preprocess_features(self):
        """Clean and preprocess features"""
        print("Preprocessing features...")
        
        # Remove features with NaN or infinite values
        X_clean = self.X.copy()
        
        # Replace NaN with median
        X_clean = X_clean.apply(lambda row: row.fillna(row.median()), axis=1)
        
        # Replace infinite values with column max/min
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.apply(lambda row: row.fillna(row.median()), axis=1)
        
        # Remove constant features (zero variance)
        feature_vars = X_clean.var(axis=1)
        non_constant_features = feature_vars > 1e-10
        X_clean = X_clean[non_constant_features]
        
        print(f"Removed {np.sum(~non_constant_features)} constant features")
        print(f"Final feature count: {X_clean.shape[0]}")
        
        self.X = X_clean
        return self.X
    
    def feature_selection(self, k=1000):
        """Select top k features based on univariate statistics"""
        # Suppress warnings for feature selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.feature_selector.set_params(k=min(k, self.X.shape[1]))
            
            # Transpose for sklearn (samples as rows)
            X_transposed = self.X.T
            
            try:
                X_selected = self.feature_selector.fit_transform(X_transposed, self.y)
                selected_features = self.X.columns[self.feature_selector.get_support()]
                print(f"Selected {X_selected.shape[1]} features out of {self.X.shape[1]}")
                return X_selected, selected_features
            except Exception as e:
                print(f"Feature selection failed: {e}")
                print("Using all features...")
                return X_transposed, self.X.columns
    
    def split_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        # Transpose X so samples are rows
        X_transposed = self.X.T
        
        # Check if we have enough samples for stratification
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        min_class_size = np.min(class_counts)
        
        if min_class_size < 2:
            print("⚠️ Warning: Some classes have only 1 sample. Stratification disabled.")
            stratify = None
        else:
            stratify = self.y
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_transposed, self.y, test_size=test_size, 
            random_state=random_state, stratify=stratify
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"y_train class distribution: {np.bincount(self.y_train)}")
        print(f"y_test class distribution: {np.bincount(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def hyperparameter_tuning(self, cv_folds=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        # Check class distribution and adjust CV folds
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)
        min_class_samples = np.min(class_counts)
        
        # Ensure we have enough samples for CV
        max_possible_folds = min_class_samples
        cv_folds = min(cv_folds, max_possible_folds)
        
        if cv_folds < 2:
            print("⚠️ Warning: Not enough samples for cross-validation. Using train-test split only.")
            return self._simple_hyperparameter_search()
        
        print(f"Using {cv_folds} CV folds (min class size: {min_class_samples})")
        
        # Simplified parameter grid to reduce computation
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': ['scale', 'auto']
        }
        
        # Create pipeline with feature selection and scaling
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=min(500, self.X_train.shape[1]))),
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        # Stratified CV to maintain class distribution
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        print("Performing hyperparameter tuning...")
        
        try:
            # Grid search with error handling
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv_strategy, 
                scoring='accuracy', n_jobs=1, verbose=1,
                error_score='raise'  # This will help us see the exact error
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
            
            return self.model
            
        except Exception as e:
            print(f"Grid search failed: {e}")
            print("Falling back to simple parameter search...")
            return self._simple_hyperparameter_search()
    
    def _simple_hyperparameter_search(self):
        """Simple parameter search without CV when data is too small"""
        print("Using simple train-validation split for parameter selection...")
        
        # Split training data further for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train, test_size=0.3, random_state=42, 
            stratify=self.y_train if len(np.unique(self.y_train)) > 1 else None
        )
        
        best_score = -1
        best_params = {}
        
        # Test different parameter combinations
        param_combinations = [
            {'C': 0.1, 'kernel': 'linear'},
            {'C': 1.0, 'kernel': 'linear'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'}
        ]
        
        for params in param_combinations:
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=min(500, X_train_split.shape[1]))),
                ('scaler', StandardScaler()),
                ('svm', SVC(probability=True, random_state=42, **params))
            ])
            
            try:
                pipeline.fit(X_train_split, y_train_split)
                score = pipeline.score(X_val_split, y_val_split)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    self.model = pipeline
                    
                print(f"Params {params}: Score = {score:.3f}")
                
            except Exception as e:
                print(f"Failed with params {params}: {e}")
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.3f}")
        
        return self.model
    
    def train_model(self):
        """Train SVM model with best parameters"""
        if self.model is None:
            # Use default parameters if no tuning was performed
            self.model = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=min(500, self.X_train.shape[1]))),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
            ])
            
            print("Training with default parameters...")
        
        self.model.fit(self.X_train, self.y_train)
        
        # Cross-validation scores (if possible)
        try:
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=3)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except:
            print("Cross-validation not possible with current data")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
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
        if len(np.unique(self.y)) == 2 and len(np.unique(self.y_test)) == 2:
            try:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                
                plt.subplot(1, 3, 2)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
            except:
                print("Could not generate ROC curve")
        
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
    
    try:
        # Load data
        X, y = svm_classifier.load_preprocessed_data(
            "geo_data/preprocessed_expression.csv",
            "geo_data/labels.csv"
        )
        
        # Preprocess features to handle data quality issues
        svm_classifier.preprocess_features()
        
        # Split data
        X_train, X_test, y_train, y_test = svm_classifier.split_data()
        
        # Hyperparameter tuning
        model = svm_classifier.hyperparameter_tuning(cv_folds=3)  # Reduced CV folds
        
        # Train model
        svm_classifier.train_model()
        
        # Evaluate model
        y_pred, y_pred_proba = svm_classifier.evaluate_model()
        
        # Save model
        svm_classifier.save_model()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your data files and ensure they contain valid data.")