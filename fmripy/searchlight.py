# searchlight_analysis.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import gc
from concurrent.futures import ThreadPoolExecutor

# Neuroimaging imports
import nibabel as nib
from nilearn import image
from nilearn.decoding import SearchLight
from nilearn.image import new_img_like, load_img
from nilearn.masking import compute_brain_mask, apply_mask
from nilearn.plotting import plot_stat_map

# Machine learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Scientific computing
from scipy.ndimage import label, generate_binary_structure, binary_erosion, binary_dilation
from statsmodels.stats.multitest import multipletests

# MPS support
import torch

# Check if MPS is available
HAS_MPS = torch.backends.mps.is_available()

from main_analysis import BaseAnalysis

@dataclass
class SearchlightResult:
    """Data class to store searchlight analysis results"""
    results_img: nib.Nifti1Image
    thresholded_img: nib.Nifti1Image
    cluster_img: nib.Nifti1Image
    cluster_coords: List[np.ndarray]
    n_clusters: int
    reject_mask: np.ndarray
    p_values: np.ndarray
    comparison_type: str
    session: str

class SearchlightAnalysis(BaseAnalysis):
    def __init__(
        self,
        derivatives_path: Union[str, Path],
        participants_file: Union[str, Path],
        output_dir: Union[str, Path],
        n_jobs: int = -1,
        use_mps: bool = True,
        memory: str = 'nilearn_cache'
    ):
        super().__init__(derivatives_path, participants_file, output_dir, n_jobs)
        self.use_mps = use_mps and HAS_MPS
        self.memory = memory
        
        if self.use_mps:
            logging.info("MPS support enabled")
            try:
                self.device = torch.device("mps")
                logging.info("Using Apple Metal device")
            except Exception as e:
                logging.warning(f"Could not initialize MPS device: {e}")
                self.use_mps = False
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def run_analysis(
        self,
        session: str,
        group1: str,
        group2: Optional[str] = None,
        radius: float = 4.0,
        n_permutations: int = 1000,
        min_cluster_size: int = 10
    ) -> Optional[SearchlightResult]:
        """Run optimized searchlight analysis"""
        try:
            # Get valid subjects
            subjects = self._get_subjects_for_analysis(session, group1, group2)
            if not subjects:
                return None
            
            # Load data with memory mapping
            X, y, mask_img = self.load_and_prepare_data(subjects, session, group1)
            if X is None:
                return None

            # MPS optimization
            if self.use_mps:
                logging.info("Running analysis with MPS acceleration...")
                try:
                    # Convert data to PyTorch tensors and move to MPS
                    X_tensor = torch.from_numpy(X).to(self.device)
                    y_tensor = torch.from_numpy(y).to(self.device)
                    
                    # Run searchlight using PyTorch for computations
                    scores = self._run_searchlight_mps(X_tensor, y_tensor, radius)
                    
                    # Move results back to CPU
                    scores = scores.cpu().numpy()
                    
                    # Clean up MPS memory
                    del X_tensor, y_tensor
                    torch.mps.empty_cache()
                    
                except Exception as e:
                    logging.warning(f"MPS processing failed, falling back to CPU: {e}")
                    scores = self._run_searchlight_cpu(X, y, radius)
            else:
                logging.info("Running analysis on CPU...")
                scores = self._run_searchlight_cpu(X, y, radius)
            
            # Create result images and process results
            results = self._process_results(scores, mask_img, min_cluster_size)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in searchlight analysis: {e}")
            return None

    def _run_searchlight_mps(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """Run searchlight analysis using MPS acceleration"""
        # Get dimensions
        n_samples, *spatial_dims = X.shape
        
        # Create searchlight sphere coordinates
        sphere_coords = self._get_sphere_coords(radius)
        
        # Initialize output
        scores = torch.zeros(spatial_dims, device=self.device)
        
        # Run cross-validation for each center voxel
        for i in range(spatial_dims[0]):
            for j in range(spatial_dims[1]):
                for k in range(spatial_dims[2]):
                    # Get sphere indices
                    sphere_mask = self._get_sphere_mask(
                        (i, j, k),
                        sphere_coords,
                        spatial_dims
                    )
                    
                    if sphere_mask.sum() > 0:
                        # Extract sphere data
                        sphere_data = X[:, sphere_mask]
                        
                        # Run cross-validation
                        score = self._cross_validate_mps(sphere_data, y)
                        scores[i, j, k] = score
        
        return scores

    def _cross_validate_mps(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Run cross-validation on MPS device"""
        cv_scores = []
        
        # Create cross-validation splits
        n_splits = self.searchlight_params['n_splits']
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        for train_idx, test_idx in kf.split(X.cpu().numpy(), y.cpu().numpy()):
            # Convert indices to tensors
            train_idx = torch.tensor(train_idx).to(self.device)
            test_idx = torch.tensor(test_idx).to(self.device)
            
            # Split data
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Train and evaluate
            score = self._train_evaluate_mps(X_train, y_train, X_test, y_test)
            cv_scores.append(score)
        
        return torch.tensor(cv_scores).mean()

    def _train_evaluate_mps(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> float:
        """Train and evaluate model using MPS"""
        # Simple logistic regression implementation using PyTorch
        model = torch.nn.Linear(X_train.shape[1], 1).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Train
        model.train()
        for epoch in range(100):  # Adjust number of epochs as needed
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output.squeeze(), y_train.float())
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            output = model(X_test)
            pred = (output.squeeze() > 0).float()
            accuracy = (pred == y_test).float().mean()
        
        return accuracy.item()

    def _run_searchlight_cpu(self, X: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
        """Run searchlight analysis on CPU"""
        searchlight = self._setup_searchlight(radius)
        return searchlight.fit(X, y)
    
    def _get_sphere_coords(self, radius: float) -> np.ndarray:
        """Create coordinates for a sphere of given radius"""
        # Create a grid of coordinates
        max_d = int(np.ceil(radius))
        x, y, z = np.ogrid[-max_d:max_d + 1,
                        -max_d:max_d + 1,
                        -max_d:max_d + 1]
        
        # Calculate distances from center
        distances = np.sqrt(x*x + y*y + z*z)
        
        # Get coordinates within radius
        sphere_coords = np.array(np.where(distances <= radius)).T
        
        # Center the coordinates
        sphere_coords -= max_d
        
        return sphere_coords

    def _get_sphere_mask(
        self,
        center: Tuple[int, int, int],
        sphere_coords: np.ndarray,
        spatial_dims: Tuple[int, ...]
    ) -> torch.Tensor:
        """Get mask for sphere at given center"""
        # Add center to coordinates
        coords = sphere_coords + np.array(center)
        
        # Create mask for valid coordinates
        valid = ((coords >= 0) & 
                (coords[:, 0] < spatial_dims[0]) &
                (coords[:, 1] < spatial_dims[1]) &
                (coords[:, 2] < spatial_dims[2])).all(axis=1)
        
        coords = coords[valid]
        
        # Create mask
        mask = torch.zeros(spatial_dims, dtype=torch.bool, device=self.device)
        if len(coords) > 0:
            mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        
        return mask

    def _setup_searchlight(self, radius: float) -> SearchLight:
        """Setup searchlight with optimized parameters"""
        return SearchLight(
            mask_img=self.searchlight_params['process_mask_img'],
            radius=radius,
            estimator=LogisticRegression(
                solver='liblinear',
                max_iter=1000
            ),
            n_jobs=self.n_jobs,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            verbose=1
        )

    def load_and_prepare_data(
        self,
        subjects: List[str],
        session: str,
        group1: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[nib.Nifti1Image]]:
        """Load and prepare data with memory mapping"""
        try:
            # Create memory mapped arrays
            temp_dir = Path(self.memory) / "temp_arrays"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize lists for data collection
            all_imgs = []
            labels = []
            
            # First pass: get dimensions
            first_sub = subjects[0]
            first_img = load_img(self._get_func_file_path(first_sub, session))
            img_shape = first_img.shape
            
            # Create memory mapped array
            X_mmap_path = temp_dir / "data.mmap"
            X_mmap = np.memmap(
                X_mmap_path,
                dtype='float32',
                mode='w+',
                shape=(len(subjects), *img_shape)
            )
            
            # Load data in parallel
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_subject = {
                    executor.submit(
                        self._load_subject_data,
                        subject, session, i, X_mmap
                    ): subject 
                    for i, subject in enumerate(subjects)
                }
                
                for future in future_to_subject:
                    subject = future_to_subject[future]
                    try:
                        future.result()
                        labels.append(1 if self.subject_groups[subject] == group1 else 0)
                    except Exception as e:
                        logging.error(f"Error loading {subject}: {e}")
            
            # Create and apply mask
            mask_img = compute_brain_mask(first_img)
            masked_data = apply_mask(X_mmap, mask_img)
            
            # Clean up
            try:
                X_mmap_path.unlink()
            except:
                pass
            
            return masked_data, np.array(labels), mask_img
            
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")
            return None, None, None

    def _process_results(
        self,
        scores: np.ndarray,
        mask_img: nib.Nifti1Image,
        min_cluster_size: int
    ) ->SearchlightResult:
        """Process searchlight results"""
        # Create results image
        results_img = new_img_like(mask_img, scores)
        
        # Threshold results
        thresh = np.percentile(scores[scores != 0], 95)
        thresholded = np.where(scores > thresh, scores, 0)
        thresholded_img = new_img_like(mask_img, thresholded)
        
        # Create clusters
        cluster_img, n_clusters, cluster_coords = self._create_clusters(
            thresholded_img,
            min_cluster_size
        )
        
        # Compute significance
        p_values = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        reject_mask = p_values < 0.05
        
        return SearchlightResult(
            results_img=results_img,
            thresholded_img=thresholded_img,
            cluster_img=cluster_img,
            cluster_coords=cluster_coords,
            n_clusters=n_clusters,
            reject_mask=reject_mask,
            p_values=p_values,
            comparison_type='between_groups',
            session=self.current_session
        )

    def _create_clusters(
        self,
        thresholded_img: nib.Nifti1Image,
        min_cluster_size: int
    ) -> Tuple[nib.Nifti1Image, int, List[np.ndarray]]:
        """Create clusters from thresholded image"""
        data = thresholded_img.get_fdata()
        binary_mask = data > 0
        structure = generate_binary_structure(3, 2)
        labeled_array, n_clusters = label(binary_mask, structure)
        
        mask = np.zeros_like(labeled_array)
        cluster_coords = []
        valid_clusters = 0
        
        for i in range(1, n_clusters + 1):
            cluster_mask = labeled_array == i
            if np.sum(cluster_mask) >= min_cluster_size:
                valid_clusters += 1
                mask[cluster_mask] = valid_clusters
                coords = np.mean(np.where(cluster_mask), axis=1)
                cluster_coords.append(coords)
        
        cluster_img = new_img_like(thresholded_img, mask)
        return cluster_img, valid_clusters, cluster_coords