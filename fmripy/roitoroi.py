# fmri_analysis.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')

import nibabel as nib
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker  # new path
from nilearn.masking import compute_brain_mask
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

@dataclass
class ConnectivityResult:
    """Data class to store connectivity analysis results"""
    connectivity_matrix: np.ndarray
    group_label: int
    subject: str

@dataclass
class SessionResult:
    """Data class to store session analysis results"""
    connectivity_matrices: np.ndarray
    group_labels: np.ndarray
    subjects: List[str]
    t_stats: np.ndarray
    p_values: np.ndarray
    p_corrected: np.ndarray
    significant: np.ndarray
    roi_names: List[str]

class ROIConnectivityAnalysis:
    def __init__(
        self, 
        derivatives_path: Union[str, Path], 
        participants_file: Union[str, Path], 
        output_dir: Union[str, Path], 
        n_jobs: int = 1,
        memory: str = 'nilearn_cache'
    ):
        """Initialize ROI Connectivity Analysis with improved type hints and memory management"""
        self.derivatives_path = Path(derivatives_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.memory = memory
        
        # Load and validate participant data
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        self._validate_participants()
        
        # Initialize connectivity measure with optimized settings
        self.correlation_measure = ConnectivityMeasure(
            kind='correlation',
            vectorize=False,
            discard_diagonal=True
        )
    
    def _prepare_rois(
    self,
        roi_files: List[str],
        cluster_img_path: Optional[str],
        target_affine: np.ndarray,
        target_shape: Tuple[int, ...]
    ) -> Tuple[nib.Nifti1Image, List[str]]:
        """Prepare ROI masks and labels"""
        roi_data = np.zeros(target_shape)
        roi_names = []
        
        # Process individual ROI files
        for i, roi_file in enumerate(roi_files, start=1):
            try:
                roi_img = image.load_img(str(roi_file))
                resampled_roi = image.resample_img(
                    roi_img,
                    target_affine=target_affine,
                    target_shape=target_shape,
                    interpolation='nearest'
                )
                roi_data[resampled_roi.get_fdata() > 0] = i
                roi_names.append(Path(roi_file).stem)
            except Exception as e:
                logging.error(f"Error loading ROI {roi_file}: {e}")
                continue
        
        # Process cluster image if provided
        if cluster_img_path and Path(cluster_img_path).exists():
            try:
                cluster_img = image.load_img(str(cluster_img_path))
                resampled_clusters = image.resample_img(
                    cluster_img,
                    target_affine=target_affine,
                    target_shape=target_shape,
                    interpolation='nearest'
                )
                cluster_data = resampled_clusters.get_fdata()
                max_roi = int(roi_data.max())
                
                for i in range(1, int(cluster_data.max()) + 1):
                    roi_data[cluster_data == i] = max_roi + i
                    roi_names.append(f'Cluster_{i}')
            except Exception as e:
                logging.error(f"Error loading clusters: {e}")
        
        if not roi_names:
            raise ValueError("No valid ROIs or clusters found")
        
        return nib.Nifti1Image(roi_data, target_affine), roi_names

    def _validate_participants(self) -> None:
        """Validate participant data and setup groups"""
        required_columns = {'participant_id', 'group'}
        if not all(col in self.participants_df.columns for col in required_columns):
            raise ValueError(f"participants.tsv must contain columns: {required_columns}")
            
        valid_groups = {'patient', 'control'}
        invalid_groups = set(self.participants_df['group']) - valid_groups
        if invalid_groups:
            raise ValueError(f"Invalid groups found: {invalid_groups}. Must be one of: {valid_groups}")
            
        self.subject_groups = dict(zip(
            self.participants_df['participant_id'],
            self.participants_df['group']
        ))
        self.subjects = list(self.subject_groups.keys())

    def _load_reference_image(self, session: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load reference image to get target affine and shape"""
        for subject in self.subjects:
            func_file = self.derivatives_path / subject / f'ses-{session}' / 'func' / \
                       f'{subject}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            if func_file.exists():
                ref_img = image.load_img(str(func_file))
                return ref_img.affine, ref_img.shape[:3]
        return None

    # In ROIConnectivityAnalysis class
    def process_subject(
        self, 
        subject: str, 
        session: str, 
        labels_img: nib.Nifti1Image, 
        roi_names: List[str]
    ) -> Optional[ConnectivityResult]:
        try:
            group_label = 1 if self.subject_groups[subject] == 'patient' else 0
            
            func_file = self.derivatives_path / subject / f'ses-{session}' / 'func' / \
                    f'{subject}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            
            if not func_file.exists():
                logging.warning(f"Missing functional file: {func_file}")
                return None
                
            func_img = image.load_img(str(func_file))
            
            # Update standardization strategy
            masker = NiftiLabelsMasker(
                labels_img=labels_img,
                standardize='zscore_sample',  # Changed from zscore to zscore_sample
                memory=self.memory,
                memory_level=2,
                n_jobs=1
            )
            
            time_series = masker.fit_transform(func_img)
            
            # Update correlation measure standardization
            self.correlation_measure = ConnectivityMeasure(
                kind='correlation',
                vectorize=False,
                discard_diagonal=True,
                standardize='zscore_sample'  # Added standardize parameter
            )
            
            connectivity_matrix = self.correlation_measure.fit_transform([time_series])[0]
            
            return ConnectivityResult(
                connectivity_matrix=connectivity_matrix,
                group_label=group_label,
                subject=subject
            )
            
        except Exception as e:
            logging.error(f"Error processing subject {subject}, session {session}: {e}")
            return None

    def run_analysis(
        self,
        roi_files: List[str],
        cluster_img_path: Optional[str] = None,
        sessions: List[str] = ['01'],
        group1: str = 'patient',
        group2: str = 'control'
    ) -> Dict[str, SessionResult]:
        """Run ROI-ROI connectivity analysis with improved parallelization and error handling"""
        results = {}
        
        for session in sessions:
            logging.info(f"\nProcessing session {session}")
            
            # Get reference image parameters
            ref_params = self._load_reference_image(session)
            if ref_params is None:
                logging.error(f"No valid reference image found for session {session}")
                continue
                
            target_affine, target_shape = ref_params
            
            try:
                # Load and prepare ROIs
                labels_img, roi_names = self._prepare_rois(
                    roi_files, 
                    cluster_img_path, 
                    target_affine, 
                    target_shape
                )
                
                # Process subjects in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    future_to_subject = {
                        executor.submit(
                            self.process_subject, 
                            subject, 
                            session, 
                            labels_img, 
                            roi_names
                        ): subject 
                        for subject in self.subjects
                    }
                    
                    processed_results = []
                    for future in as_completed(future_to_subject):
                        result = future.result()
                        if result is not None:
                            processed_results.append(result)
                
                if not processed_results:
                    logging.warning(f"No valid results for session {session}")
                    continue
                    
                # Analyze results
                session_result = self._analyze_session_results(
                    processed_results,
                    roi_names,
                    len(roi_names)
                )
                
                if session_result is not None:
                    results[session] = session_result
                    self._plot_and_save_results(
                        session_result,
                        f'{group1}_vs_{group2}_session_{session}'
                    )
                    
            except Exception as e:
                logging.error(f"Error processing session {session}: {e}")
                continue
                
        return results

    def _analyze_session_results(
        self,
        processed_results: List[ConnectivityResult],
        roi_names: List[str],
        n_rois: int
    ) -> Optional[SessionResult]:
        """Analyze processed results for a session"""
        try:
            # Prepare arrays
            connectivity_matrices = []
            group_labels = []
            subjects = []
            
            for result in processed_results:
                if result.connectivity_matrix.shape == (n_rois, n_rois):
                    connectivity_matrices.append(result.connectivity_matrix)
                    group_labels.append(result.group_label)
                    subjects.append(result.subject)
                    
            if not connectivity_matrices:
                return None
                
            # Convert to numpy arrays
            connectivity_matrices = np.array(connectivity_matrices)
            group_labels = np.array(group_labels)
            
            # Perform statistical analysis
            group1_matrices = connectivity_matrices[group_labels == 1]
            group2_matrices = connectivity_matrices[group_labels == 0]
            
            if len(group1_matrices) == 0 or len(group2_matrices) == 0:
                return None
                
            # Calculate statistics
            t_stats, p_values = ttest_ind(group1_matrices, group2_matrices, axis=0)
            
            # FDR correction
            mask = np.triu(np.ones(t_stats.shape, dtype=bool), k=1)
            p_values_masked = p_values[mask]
            reject, p_corrected, _, _ = multipletests(p_values_masked, method='fdr_bh')
            
            return SessionResult(
                connectivity_matrices=connectivity_matrices,
                group_labels=group_labels,
                subjects=subjects,
                t_stats=t_stats,
                p_values=p_values,
                p_corrected=p_corrected,
                significant=reject,
                roi_names=roi_names
            )
            
        except Exception as e:
            logging.error(f"Error analyzing session results: {e}")
            return None

    def _plot_and_save_results(self, session_result: SessionResult, output_prefix: str) -> None:
        """Plot and save analysis results with improved visualization"""
        plt.figure(figsize=(15, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(session_result.t_stats, dtype=bool), k=1)
        
        # Plot heatmap
        sns.heatmap(
            session_result.t_stats * mask,
            xticklabels=session_result.roi_names,
            yticklabels=session_result.roi_names,
            cmap='coolwarm',
            center=0,
            square=True,
            mask=~mask,
            cbar_kws={'label': 't-statistic'},
            annot=False,
            fmt=".2f"
        )
        
        # Mark significant connections
        if np.any(session_result.significant):
            i_indices, j_indices = np.triu_indices_from(session_result.t_stats, k=1)
            sig_i = i_indices[session_result.significant]
            sig_j = j_indices[session_result.significant]
            plt.scatter(
                sig_j + 0.5, 
                sig_i + 0.5, 
                marker='*', 
                color='k', 
                s=100, 
                label='Significant'
            )
            plt.legend(loc='upper right')
            
        plt.title(f'Group Differences in ROI Connectivity\n{output_prefix}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save results
        try:
            # Save plot
            plot_path = self.output_dir / f'roi_connectivity_{output_prefix}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save significant connections
            if np.any(session_result.significant):
                self._save_significant_connections(session_result, output_prefix)
                
        except Exception as e:
            logging.error(f"Error saving results: {e}")

    def _save_significant_connections(
        self,
        session_result: SessionResult,
        output_prefix: str
    ) -> None:
        """Save significant connections to CSV"""
        try:
            sig_connections = []
            i_indices, j_indices = np.triu_indices_from(session_result.t_stats, k=1)
            
            for idx, is_sig in enumerate(session_result.significant):
                if is_sig:
                    sig_connections.append({
                        'ROI_1': session_result.roi_names[i_indices[idx]],
                        'ROI_2': session_result.roi_names[j_indices[idx]],
                        't_statistic': session_result.t_stats[i_indices[idx], j_indices[idx]],
                        'p_value_fdr': session_result.p_corrected[idx]
                    })
            
            if sig_connections:
                sig_df = pd.DataFrame(sig_connections)
                csv_path = self.output_dir / f'significant_connections_{output_prefix}.csv'
                sig_df.to_csv(csv_path, index=False)
                
        except Exception as e:
            logging.error(f"Error saving significant connections: {e}")