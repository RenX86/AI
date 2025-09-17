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