# fMRI Connectivity Analysis

A Python package for analyzing fMRI connectivity data using different approaches:
- Searchlight analysis
- ROI-ROI connectivity analysis
- Seed-based connectivity analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/LazyCyborg/fmripy.git

# Move into the directory
cd fmripy

# Install the package
pip install -e .
```

## Usage

```python
from fmripy import (
    SearchlightAnalysis,
    ROIConnectivityAnalysis,
    SeedConnectivityAnalysis
)

# Initialize analyses
searchlight = SearchlightAnalysis(
    derivatives_path="path/to/derivatives",
    participants_file="path/to/participants.tsv",
    output_dir="analysis_results"
)

roi_analysis = ROIConnectivityAnalysis(
    derivatives_path="path/to/derivatives",
    participants_file="path/to/participants.tsv",
    output_dir="analysis_results"
)

seed_analysis = SeedConnectivityAnalysis(
    derivatives_path="path/to/derivatives",
    participants_file="path/to/participants.tsv",
    output_dir="analysis_results"
)

# Run analyses
searchlight_results = searchlight.run_analysis(
    session='01',
    group1='patient',
    group2='control'
)

roi_results = roi_analysis.run_analysis(
    roi_files=['path/to/roi1.nii.gz', 'path/to/roi2.nii.gz'],
    cluster_img_path='searchlight_clusters.nii.gz',
    sessions=['01', '02']
)

seed_results = seed_analysis.run_analysis(
    roi_files=['path/to/roi1.nii.gz', 'path/to/roi2.nii.gz'],
    cluster_img_path='searchlight_clusters.nii.gz',
    sessions=['01', '02']
)
```

## Requirements

- Python >=3.7
- numpy >=1.19.0
- pandas >=1.2.0
- nilearn >=0.9.0
- scikit-learn >=0.24.0
- scipy >=1.6.0
- matplotlib >=3.3.0
- seaborn >=0.11.0
- nibabel >=3.2.0
- statsmodels >=0.12.0
- joblib >=1.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.