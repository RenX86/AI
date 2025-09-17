import os
import sys
import argparse
import logging
import pandas as pd
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
            
            # Create labels dataframe
            labels_df = pd.DataFrame({
                'sample': metadata['sample_id'], 
                'label': labels
            })

            # Save preprocessed data (optional, as data is passed in memory)
            processed_data.to_csv(f"{self.output_dir}/preprocessed_expression.csv")
            labels_df.to_csv(f"{self.output_dir}/labels.csv", index=False)
            
            logger.info("Preprocessing completed successfully")
            
            # Step 3: SVM Classification
            if perform_svm:
                logger.info("Step 3: Running SVM classification...")
                
                # Pass data directly to the classifier
                self.classifier.set_data(processed_data, labels_df)
                
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
                
                # Pass data directly to the clusterer
                self.clusterer.set_data(processed_data, labels_df)
                
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
