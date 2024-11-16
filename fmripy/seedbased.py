# seed_connectivity_analysis.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')
import gc

import nibabel as nib
from nilearn import image, datasets
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker
from nilearn.masking import compute_brain_mask
from nilearn.image import math_img, mean_img, new_img_like
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

@dataclass
class SeedResult:
    """Data class to store seed connectivity results"""
    t_map: nib.Nifti1Image
    significant_map: nib.Nifti1Image
    t_stats: np.ndarray
    p_values: np.ndarray
    maps: List[np.ndarray]
    groups: List[int]

@dataclass
class SessionResult:
    """Data class to store session analysis results"""
    seed_results: Dict[str, SeedResult]
    session: str

class SeedConnectivityAnalysis:
    def __init__(
        self,
        derivatives_path: Union[str, Path],
        participants_file: Union[str, Path],
        output_dir: Union[str, Path],
        n_jobs: int = 1,
        memory: str = 'nilearn_cache'
    ):
        """Initialize Seed-Based Connectivity Analysis with improved configuration"""
        self.derivatives_path = Path(derivatives_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.memory = memory
        
        # Load and validate participant data
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        self._validate_participants()
        
        # Get MNI template for visualization
        self.mni_template = datasets.load_mni152_template()
        
        # Initialize analysis parameters
        self.seed_params = {
            'radius': 8,
            'standardize': 'zscore_sample',
            'memory_level': 1
        }

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

    def load_rois_and_clusters(
        self,
        roi_files: List[str],
        cluster_img_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Load ROI masks and clusters with improved error handling"""
        coords = []
        names = []
        
        # Load ROIs
        for roi_file in roi_files:
            try:
                roi_img = image.load_img(roi_file)
                roi_data = roi_img.get_fdata()
                
                if not np.any(roi_data > 0):
                    logging.warning(f"Empty ROI found in {roi_file}")
                    continue
                    
                coord = np.mean(np.where(roi_data > 0), axis=1)
                coords.append(coord)
                names.append(Path(roi_file).stem)
                
            except Exception as e:
                logging.error(f"Error loading ROI {roi_file}: {e}")
                continue
        
        # Load clusters if provided
        if cluster_img_path and Path(cluster_img_path).exists():
            try:
                cluster_img = image.load_img(cluster_img_path)
                cluster_data = cluster_img.get_fdata()
                n_clusters = int(cluster_data.max())
                
                cluster_names = self._get_cluster_names(n_clusters)
                
                for i in range(1, n_clusters + 1):
                    cluster_mask = cluster_data == i
                    if np.any(cluster_mask):
                        coord = np.mean(np.where(cluster_mask), axis=1)
                        coords.append(coord)
                        names.append(cluster_names[i-1])
                        
            except Exception as e:
                logging.error(f"Error loading clusters: {e}")
        
        if not coords:
            raise ValueError("No valid ROIs or clusters found")
            
        return np.array(coords), names

    def _get_cluster_names(self, n_clusters: int) -> List[str]:
        """Get cluster names from file or generate default names"""
        cluster_info_path = self.output_dir / 'cluster_locations.csv'
        if cluster_info_path.exists():
            cluster_info = pd.read_csv(cluster_info_path)
            return [f"{row['Cluster']}_{row['Anatomical_Region']}" 
                    for _, row in cluster_info.iterrows()]
        return [f'Cluster_{i+1}' for i in range(n_clusters)]

    def compute_seed_connectivity(
        self,
        func_img: nib.Nifti1Image,
        seed_coord: np.ndarray,
        brain_masker: NiftiMasker
    ) -> np.ndarray:
        """Compute whole-brain connectivity map for a seed"""
        try:
            # Create seed masker with proper coordinates format
            seed_coords = seed_coord.reshape(1, -1)  # Reshape to 2D array with shape (1, 3)
            seed_masker = NiftiSpheresMasker(
                seeds=seed_coords,
                radius=self.seed_params['radius'],
                standardize=self.seed_params['standardize'],
                memory=self.memory,
                memory_level=self.seed_params['memory_level']
            )
            
            seed_time_series = seed_masker.fit_transform(func_img)
            brain_time_series = brain_masker.transform(func_img)
            
            correlations = np.zeros(brain_time_series.shape[1])
            for i in range(brain_time_series.shape[1]):
                correlations[i] = np.corrcoef(
                    seed_time_series.ravel(),
                    brain_time_series[:, i]
                )[0, 1]
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error computing seed connectivity: {e}")
            raise

    def _process_subject(
        self,
        subject: str,
        session: str,
        brain_masker: NiftiMasker,
        seed_coords: np.ndarray,
        seed_names: List[str],
        group1: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single subject with error handling"""
        try:
            func_file = (self.derivatives_path / subject / f'ses-{session}' / 'func' /
                        f'{subject}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
            
            if not func_file.exists():
                logging.warning(f"Missing file for {subject}")
                return None
                
            func_img = image.load_img(str(func_file))
            results = {}
            
            for seed_coord, seed_name in zip(seed_coords, seed_names):
                connectivity_map = self.compute_seed_connectivity(
                    func_img, seed_coord, brain_masker
                )
                results[seed_name] = {
                    'map': connectivity_map,
                    'group': 1 if self.subject_groups[subject] == group1 else 0
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing {subject}: {e}")
            return None

    def _compute_statistics(
        self,
        maps: np.ndarray,
        groups: np.ndarray,
        brain_masker: NiftiMasker
    ) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, np.ndarray]:
        """Compute statistics for connectivity maps"""
        group1_maps = maps[groups == 1]
        group2_maps = maps[groups == 0]
        
        t_stats = np.zeros(maps.shape[1])
        p_values = np.zeros(maps.shape[1])
        
        for i in range(maps.shape[1]):
            t_stat, p_val = ttest_ind(group1_maps[:, i], group2_maps[:, i])
            t_stats[i] = t_stat
            p_values[i] = p_val
        
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        t_map = brain_masker.inverse_transform(t_stats)
        significant_map = brain_masker.inverse_transform(
            np.where(p_corrected < 0.05, t_stats, 0)
        )
        
        return t_map, significant_map, t_stats, p_corrected

    def plot_seed_results(
        self,
        t_map: nib.Nifti1Image,
        significant_map: nib.Nifti1Image,
        seed_name: str,
        output_prefix: str
    ) -> None:
        """Plot seed connectivity results with error handling"""
        try:
            # Plot t-statistics map
            plt.figure(figsize=(15, 5))
            plot_stat_map(
                t_map,
                bg_img=self.mni_template,
                display_mode='ortho',
                title=f'Seed Connectivity Differences\n{seed_name}',
                colorbar=True,
                cut_coords=(0, 0, 0)
            )
            plt.savefig(
                self.output_dir / f't_map_{output_prefix}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Plot significant results if any exist
            if np.any(significant_map.get_fdata() != 0):
                plt.figure(figsize=(15, 5))
                plot_stat_map(
                    significant_map,
                    bg_img=self.mni_template,
                    display_mode='ortho',
                    title=f'Significant Connectivity Differences\n{seed_name}',
                    colorbar=True,
                    cut_coords=(0, 0, 0)
                )
                plt.savefig(
                    self.output_dir / f'significant_map_{output_prefix}.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
        except Exception as e:
            logging.error(f"Error plotting results: {e}")

    def run_analysis(
        self,
        roi_files: List[str],
        cluster_img_path: Optional[str] = None,
        sessions: List[str] = ['01'],
        group1: str = 'patient',
        group2: str = 'control'
    ) -> Dict[str, SessionResult]:
        """Run complete seed-based connectivity analysis"""
        results = {}
        
        try:
            # Load ROIs and clusters
            seed_coords, seed_names = self.load_rois_and_clusters(
                roi_files,
                cluster_img_path
            )
            
            for session in sessions:
                logging.info(f"\nProcessing session {session}")
                
                # Get subjects for specified groups
                selected_subjects = [subj for subj in self.subjects 
                                   if self.subject_groups[subj] in [group1, group2]]
                
                # Initialize brain masker
                brain_masker = self._initialize_brain_masker(
                    selected_subjects[0],
                    session
                )
                
                # Process subjects in parallel
                processed_results = self._process_subjects_parallel(
                    selected_subjects,
                    session,
                    brain_masker,
                    seed_coords,
                    seed_names,
                    group1
                )
                
                # Analyze results for each seed
                session_results = {}
                for seed_name in seed_names:
                    seed_results = self._analyze_seed_results(
                        processed_results,
                        seed_name,
                        brain_masker,
                        group1,
                        group2,
                        session
                    )
                    session_results[seed_name] = seed_results
                
                results[session] = SessionResult(
                    seed_results=session_results,
                    session=session
                )
                
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            raise
        
        return results

    def _initialize_brain_masker(
        self,
        first_subject: str,
        session: str
    ) -> NiftiMasker:
        """Initialize brain masker with first subject"""
        first_func_file = (self.derivatives_path / first_subject / f'ses-{session}' / 'func' /
                          f'{first_subject}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        
        brain_mask = compute_brain_mask(first_func_file)
        brain_masker = NiftiMasker(
            mask_img=brain_mask,
            standardize=self.seed_params['standardize'],
            memory=self.memory,
            memory_level=self.seed_params['memory_level']
        )
        brain_masker.fit()
        return brain_masker

    def _process_subjects_parallel(
        self,
        subjects: List[str],
        session: str,
        brain_masker: NiftiMasker,
        seed_coords: np.ndarray,
        seed_names: List[str],
        group1: str
    ) -> List[Dict[str, Any]]:
        """Process subjects in parallel"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for subject in subjects:
                futures.append(
                    executor.submit(
                        self._process_subject,
                        subject,
                        session,
                        brain_masker,
                        seed_coords,
                        seed_names,
                        group1
                    )
                )
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logging.error(f"Error in parallel processing: {e}")
                    continue
                    
        return results

    def _analyze_seed_results(
        self,
        processed_results: List[Dict[str, Any]],
        seed_name: str,
        brain_masker: NiftiMasker,
        group1: str,
        group2: str,
        session: str
    ) -> Optional[SeedResult]:
        """Analyze processed results for a specific seed"""
        try:
            # Extract maps and groups for the current seed
            maps = []
            groups = []
            
            for result in processed_results:
                if seed_name in result:
                    maps.append(result[seed_name]['map'])
                    groups.append(result[seed_name]['group'])
            
            if not maps:
                logging.warning(f"No valid results for seed {seed_name}")
                return None
                
            maps = np.array(maps)
            groups = np.array(groups)
            
            # Compute statistics
            t_map, significant_map, t_stats, p_corrected = self._compute_statistics(
                maps, groups, brain_masker
            )
            
            # Save results
            self._save_seed_results(
                t_map,
                significant_map,
                seed_name,
                group1,
                group2,
                session
            )
            
            # Plot results
            self.plot_seed_results(
                t_map,
                significant_map,
                seed_name,
                f"{group1}_vs_{group2}_session_{session}_{seed_name}"
            )
            
            return SeedResult(
                t_map=t_map,
                significant_map=significant_map,
                t_stats=t_stats,
                p_values=p_corrected,
                maps=list(maps),
                groups=list(groups)
            )
            
        except Exception as e:
            logging.error(f"Error analyzing seed {seed_name}: {e}")
            return None

    def _save_seed_results(
        self,
        t_map: nib.Nifti1Image,
        significant_map: nib.Nifti1Image,
        seed_name: str,
        group1: str,
        group2: str,
        session: str
    ) -> None:
        """Save seed analysis results to files"""
        try:
            output_prefix = f"{group1}_vs_{group2}_session_{session}_{seed_name}"
            
            # Save statistical maps
            t_map.to_filename(
                self.output_dir / f"t_map_{output_prefix}.nii.gz"
            )
            significant_map.to_filename(
                self.output_dir / f"significant_map_{output_prefix}.nii.gz"
            )
            
        except Exception as e:
            logging.error(f"Error saving results for seed {seed_name}: {e}")

    def _clean_memory(self) -> None:
        """Clean up memory after processing"""
        gc.collect()

    def _validate_inputs(
        self,
        roi_files: List[str],
        cluster_img_path: Optional[str],
        sessions: List[str],
        group1: str,
        group2: str
    ) -> None:
        """Validate input parameters"""
        # Validate ROI files
        for roi_file in roi_files:
            if not Path(roi_file).exists():
                raise ValueError(f"ROI file not found: {roi_file}")
        
        # Validate cluster image if provided
        if cluster_img_path and not Path(cluster_img_path).exists():
            raise ValueError(f"Cluster image not found: {cluster_img_path}")
        
        # Validate groups
        valid_groups = {'patient', 'control'}
        if group1 not in valid_groups or group2 not in valid_groups:
            raise ValueError(f"Invalid group(s). Must be one of: {valid_groups}")
        
        if group1 == group2:
            raise ValueError("group1 and group2 must be different")

    def get_analysis_summary(
        self,
        results: Dict[str, SessionResult]
    ) -> pd.DataFrame:
        """Generate summary of analysis results"""
        summary_data = []
        
        for session, session_result in results.items():
            for seed_name, seed_result in session_result.seed_results.items():
                if seed_result is not None:
                    n_significant = np.sum(seed_result.p_values < 0.05)
                    max_t = np.max(np.abs(seed_result.t_stats))
                    
                    summary_data.append({
                        'Session': session,
                        'Seed': seed_name,
                        'Significant_Voxels': n_significant,
                        'Max_Absolute_T': max_t,
                        'Total_Subjects': len(seed_result.groups),
                        'Group1_Subjects': np.sum(np.array(seed_result.groups) == 1),
                        'Group2_Subjects': np.sum(np.array(seed_result.groups) == 0)
                    })
        
        return pd.DataFrame(summary_data)

    def plot_analysis_overview(
        self,
        results: Dict[str, SessionResult],
        output_prefix: str
    ) -> None:
        """Create overview plots of analysis results"""
        try:
            summary_df = self.get_analysis_summary(results)
            
            # Plot number of significant voxels by seed
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=summary_df,
                x='Seed',
                y='Significant_Voxels',
                hue='Session'
            )
            plt.xticks(rotation=45, ha='right')
            plt.title('Number of Significant Voxels by Seed and Session')
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'significant_voxels_overview_{output_prefix}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Plot maximum t-statistics by seed
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=summary_df,
                x='Seed',
                y='Max_Absolute_T',
                hue='Session'
            )
            plt.xticks(rotation=45, ha='right')
            plt.title('Maximum Absolute T-Statistics by Seed and Session')
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'max_t_stats_overview_{output_prefix}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating overview plots: {e}")

    def save_analysis_summary(
        self,
        results: Dict[str, SessionResult],
        output_prefix: str
    ) -> None:
        """Save analysis summary to CSV"""
        try:
            summary_df = self.get_analysis_summary(results)
            summary_df.to_csv(
                self.output_dir / f'analysis_summary_{output_prefix}.csv',
                index=False
            )
        except Exception as e:
            logging.error(f"Error saving analysis summary: {e}")