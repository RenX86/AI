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
    gse_id = "GSE307516"  # Breast cancer dataset
    gse = downloader.download_gse_dataset(gse_id)
    
    if gse:
        expression_data = downloader.extract_expression_matrix(gse)
        metadata = downloader.get_sample_metadata(gse)
        
        # Save data
        expression_data.to_csv("geo_data/expression_matrix.csv")
        metadata.to_csv("geo_data/sample_metadata.csv", index=False)
        
        print(f"Expression matrix shape: {expression_data.shape}")
        print(f"Metadata shape: {metadata.shape}")