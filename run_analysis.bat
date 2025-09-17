@echo off
echo GEO Analysis Pipeline - Quick Start
echo ===================================

REM Run analysis with example dataset
python main_analysis_pipeline.py GSE2034 --output-dir results_GSE2034

echo.
echo Analysis completed! Check results_GSE2034 folder for outputs.
pause