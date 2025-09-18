# GEO Data Analysis Pipeline

This project is a Python-based pipeline for downloading, preprocessing, and analyzing gene expression data from the Gene Expression Omnibus (GEO) database. It performs SVM classification and K-means clustering to identify patterns and classify samples based on their gene expression profiles.

## Features

- **Data Download**: Downloads GEO datasets using a GSE ID.
- **Preprocessing**: Handles missing values, filters low variance genes, and normalizes the data.
- **SVM Classification**: Performs SVM classification with hyperparameter tuning to classify samples.
- **K-means Clustering**: Performs K-means clustering to identify distinct sample groups, with automatic detection of the optimal number of clusters.
- **Report Generation**: Generates a comprehensive analysis report in Markdown format, summarizing the results and generated files.
- **Visualization**: Creates various plots to visualize the data and analysis results, including heatmaps, PCA plots, and interactive 3D cluster plots.

## Workflow

The pipeline follows these main steps:

1.  **Data Download**: The `geo_downloader.py` module downloads the specified GEO dataset.
2.  **Preprocessing**: The `data_preprocessing.py` module cleans and prepares the data for analysis.
3.  **SVM Classification**: If enabled, the `svm_classification.py` module trains and evaluates an SVM classifier.
4.  **K-means Clustering**: If enabled, the `kmeans_clustering.py` module performs K-means clustering and evaluates the results.
5.  **Report Generation**: The `main_analysis_pipeline.py` script orchestrates the entire workflow and generates a final report summarizing the analysis.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/RenX86/AI.git
    cd AI
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the complete analysis pipeline using the `main_analysis_pipeline.py` script. You need to provide the GEO Series ID (e.g., `GSE2034`) as a command-line argument.

```bash
python main_analysis_pipeline.py <GSE_ID>
```

For example:

```bash
python main_analysis_pipeline.py GSE2034
```

### Command-line Options

-   `--output-dir`: Specify the directory to save the results (default: `geo_analysis_results`).
-   `--skip-svm`: Skip the SVM classification step.
-   `--skip-clustering`: Skip the K-means clustering step.

Example with options:

```bash
python main_analysis_pipeline.py GSE2034 --output-dir my_results --skip-svm
```

Alternatively, you can use the `run_analysis.bat` script (on Windows) to run the pipeline with a predefined GSE ID.

## File Descriptions

-   `main_analysis_pipeline.py`: The main script that orchestrates the entire analysis pipeline.
-   `geo_downloader.py`: Contains the `GEODataDownloader` class for downloading data from the GEO database.
-   `data_preprocessing.py`: Contains the `GEODataPreprocessor` class for cleaning and preprocessing the gene expression data.
-   `svm_classification.py`: Contains the `SVMClassifier` class for performing SVM classification.
-   `kmeans_clustering.py`: Contains the `KMeansClusterer` class for performing K-means clustering.
-   `run_analysis.bat`: A simple batch script for running the main pipeline on Windows.
-   `Readme.md`: This file.


