# fmripy - fMRI Connectivity Analysis

`fmripy` is a Python package for performing various fMRI connectivity analyses, including:

1. **Searchlight Analysis**: A multivariate pattern analysis technique that examines local patterns of brain activity.
2. **ROI-to-ROI Connectivity Analysis**: Analyzes functional connectivity between predefined regions of interest (ROIs).
3. **Seed-Based Connectivity Analysis**: Examines whole-brain functional connectivity with respect to a seed region.

This package is designed to provide fMRI connectivity analyses, with support for parallelization and GPU acceleration (where available) using pytorch.

## Installation

To install the `fmripy` package, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/LazyCyborg/fmripy.git
```
2. Navigate to the project directory:
```bash
cd fmripy
```
3. Install the package using `pip`:
```bash
pip install .
```

## Usage

Here's an example of how to use the `fmripy` package:

```python
from fmripy.searchlight import SearchlightAnalysis
from fmripy.roitoroi import ROIConnectivityAnalysis
from fmripy.seedbased import SeedConnectivityAnalysis
from fmripy.main_analysis import DERIVATIVES_PATH, PARTICIPANTS_FILE, OUTPUT_DIR, ANALYSIS_PARAMS, ROIS

# Searchlight Analysis
searchlight_analysis = SearchlightAnalysis(
 derivatives_path=DERIVATIVES_PATH,
 participants_file=PARTICIPANTS_FILE,
 output_dir=OUTPUT_DIR,
 n_jobs=ANALYSIS_PARAMS['n_jobs']
)

searchlight_results = searchlight_analysis.run_analysis(
 session=ANALYSIS_PARAMS['analysis_types']['between_groups']['session'],
 group1=ANALYSIS_PARAMS['group1'],
 group2=ANALYSIS_PARAMS['group2'],
 radius=4.0,
 n_permutations=1000,
 min_cluster_size=10
)

# ROI Connectivity Analysis
roi_analysis = ROIConnectivityAnalysis(
 derivatives_path=DERIVATIVES_PATH,
 participants_file=PARTICIPANTS_FILE,
 output_dir=OUTPUT_DIR,
 n_jobs=ANALYSIS_PARAMS['n_jobs']
)

roi_results = roi_analysis.run_analysis(
 roi_files=ROIS,
 sessions=list(ANALYSIS_PARAMS['analysis_types'].keys()),
 group1=ANALYSIS_PARAMS['group1'],
 group2=ANALYSIS_PARAMS['group2']
)

# Seed-Based Connectivity Analysis
seed_analysis = SeedConnectivityAnalysis(
 derivatives_path=DERIVATIVES_PATH,
 participants_file=PARTICIPANTS_FILE,
 output_dir=OUTPUT_DIR,
 n_jobs=ANALYSIS_PARAMS['n_jobs'],
 use_mps=True
)

seed_results = seed_analysis.run_analysis(
 seed_coords=[(0, 0, 0), (10, 20, 30), (-20, -10, 15)],
 seed_names=['Seed 1', 'Seed 2', 'Seed 3'],
 session=ANALYSIS_PARAMS['analysis_types']['between_groups']['session'],
 group1=ANALYSIS_PARAMS['group1'],
 group2=ANALYSIS_PARAMS['group2']
)

```

This example demonstrates how to use the different analysis classes provided by the fmripy package, including Searchlight Analysis, ROI-to-ROI Connectivity Analysis, and Seed-Based Connectivity Analysis. The example also shows how to import the necessary paths and configuration parameters.
For more detailed information on the functionality and usage of the fmripy package, please refer to the API documentation.
Acknowledgements
nilearn 

This package was developed by Alexander Engelmark and is licensed under the MIT License.

Contributions
Contributions to the fmripy package are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the GitHub repository.
