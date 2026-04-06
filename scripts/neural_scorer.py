"""
Neural Scorer — Calibration MLP that maps cognitive dimensions to a quality score.

Pipeline: text → TRIBE v2 → ROI extraction → calibration MLP → scalar reward

The MLP is trained on (cognitive_dimensions, human_rating) pairs from the
calibration dataset (Experiment 3). Once trained, it can score any text
without human input.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from roi_extractor import ROIExtractor, CognitiveDimensions


class CalibrationMLP(nn.Module):
    """Tiny MLP: 5 cognitive dimensions → scalar quality score.

    Deliberately small (2K parameters) to prevent overfitting on
    small calibration datasets (100-1000 samples).
    """

    def __init__(self, n_dims: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class CalibrationResult:
    """Results from training the calibration MLP."""
    train_correlation: float
    val_correlation: float
    val_r2: float
    feature_importance: dict
    n_train: int
    n_val: int


class NeuralScorer:
    """Complete neural scoring pipeline.

    Usage:
        scorer = NeuralScorer()
        scorer.train(features, ratings)
        score = scorer.score_text(tribe_wrapper, "some text")
    """

    def __init__(self, n_dims: int = 5, device: str = "cpu"):
        self.n_dims = n_dims
        self.device = device
        self.mlp = CalibrationMLP(n_dims=n_dims).to(device)
        self.roi_extractor = ROIExtractor(n_rois=1000)
        self.is_trained = False
        self._feature_means = None
        self._feature_stds = None

    def train(
        self,
        features: np.ndarray,
        ratings: np.ndarray,
        n_folds: int = 5,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> CalibrationResult:
        """Train the calibration MLP with cross-validation.

        Args:
            features: (n_samples, n_dims) cognitive dimension scores
            ratings: (n_samples,) human quality ratings (e.g., 1-7 Likert)
            n_folds: Number of cross-validation folds
            epochs: Training epochs per fold
            lr: Learning rate
            verbose: Print training progress

        Returns:
            CalibrationResult with correlation metrics.
        """
        assert features.shape[1] == self.n_dims
        assert len(features) == len(ratings)

        # Standardize features
        self._feature_means = features.mean(axis=0)
        self._feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self._feature_means) / self._feature_stds

        # Standardize ratings
        self._rating_mean = ratings.mean()
        self._rating_std = ratings.std() + 1e-8
        ratings_norm = (ratings - self._rating_mean) / self._rating_std

        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        val_correlations = []
        val_predictions_all = []
        val_ratings_all = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(features_norm)):
            X_train = torch.tensor(features_norm[train_idx], dtype=torch.float32).to(self.device)
            y_train = torch.tensor(ratings_norm[train_idx], dtype=torch.float32).to(self.device)
            X_val = torch.tensor(features_norm[val_idx], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(ratings_norm[val_idx], dtype=torch.float32).to(self.device)

            # Fresh model per fold
            model = CalibrationMLP(n_dims=self.n_dims).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            # Train
            model.train()
            for epoch in range(epochs):
                pred = model(X_train)
                loss = nn.MSELoss()(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).cpu().numpy()
                val_true = y_val.cpu().numpy()

            r, _ = pearsonr(val_pred, val_true)
            val_correlations.append(r)
            val_predictions_all.extend(val_pred)
            val_ratings_all.extend(val_true)

            if verbose:
                print(f"  Fold {fold+1}/{n_folds}: val correlation = {r:.3f}")

        # Train final model on all data
        X_all = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
        y_all = torch.tensor(ratings_norm, dtype=torch.float32).to(self.device)

        self.mlp = CalibrationMLP(n_dims=self.n_dims).to(self.device)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr, weight_decay=1e-4)

        self.mlp.train()
        for epoch in range(epochs * 2):  # More epochs for final model
            pred = self.mlp(X_all)
            loss = nn.MSELoss()(pred, y_all)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.mlp.eval()
        self.is_trained = True

        # Compute feature importance (absolute weight from first layer)
        first_layer_weights = self.mlp.net[0].weight.data.cpu().numpy()
        importance = np.abs(first_layer_weights).mean(axis=0)
        importance = importance / importance.sum()

        dim_labels = CognitiveDimensions().labels
        feature_importance = {
            dim_labels[i]: float(importance[i]) for i in range(self.n_dims)
        }

        # Training correlation
        with torch.no_grad():
            train_pred = self.mlp(X_all).cpu().numpy()
        train_r, _ = pearsonr(train_pred, ratings_norm)

        # Overall validation metrics
        val_r = np.mean(val_correlations)
        val_r2 = r2_score(val_ratings_all, val_predictions_all)

        result = CalibrationResult(
            train_correlation=float(train_r),
            val_correlation=float(val_r),
            val_r2=float(val_r2),
            feature_importance=feature_importance,
            n_train=len(features),
            n_val=len(features) // n_folds,
        )

        if verbose:
            print(f"\n  Training correlation: {result.train_correlation:.3f}")
            print(f"  Validation correlation (mean): {result.val_correlation:.3f}")
            print(f"  Validation R²: {result.val_r2:.3f}")
            print(f"\n  Feature importance:")
            for name, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"    {name:25s} {imp:.3f} {bar}")

        return result

    def score_dims(self, dims: CognitiveDimensions) -> float:
        """Score a single set of cognitive dimensions.

        Args:
            dims: CognitiveDimensions from ROI extraction

        Returns:
            Scalar quality score (higher = better).
        """
        assert self.is_trained, "Call train() first"

        features = dims.to_array()
        features_norm = (features - self._feature_means) / self._feature_stds
        x = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score_norm = self.mlp(x).item()

        # Denormalize
        score = score_norm * self._rating_std + self._rating_mean
        return score

    def score_activation(self, activation: np.ndarray) -> float:
        """Score a raw brain activation map.

        Args:
            activation: (n_parcels,) predicted BOLD activation

        Returns:
            Scalar quality score.
        """
        dims = self.roi_extractor.extract(activation)
        return self.score_dims(dims)

    def score_text(self, tribe_wrapper, text: str) -> tuple[float, CognitiveDimensions]:
        """Full pipeline: text → TRIBE v2 → ROI → MLP → score.

        Args:
            tribe_wrapper: TribeWrapper instance
            text: Input text to score

        Returns:
            (score, dimensions) tuple
        """
        activation = tribe_wrapper.predict_text(text)

        # Handle case where TRIBE v2 returns different parcel count
        if len(activation) != self.roi_extractor.n_rois:
            # Re-initialize ROI extractor with correct size
            self.roi_extractor = ROIExtractor(n_rois=len(activation))

        dims = self.roi_extractor.extract(activation)
        score = self.score_dims(dims)
        return score, dims

    def compare_texts(
        self, tribe_wrapper, text_a: str, text_b: str
    ) -> dict:
        """Compare two texts and return which one is neurally preferred.

        This is the core function for generating DPO preference labels.

        Returns:
            dict with scores, dimensions, and preference label.
        """
        score_a, dims_a = self.score_text(tribe_wrapper, text_a)
        score_b, dims_b = self.score_text(tribe_wrapper, text_b)

        margin = abs(score_a - score_b)
        preferred = "a" if score_a > score_b else "b"

        return {
            "score_a": score_a,
            "score_b": score_b,
            "dims_a": dims_a.to_dict(),
            "dims_b": dims_b.to_dict(),
            "margin": margin,
            "preferred": preferred,
            "confident": margin > 0.3,  # Margin threshold for DPO
        }

    def save(self, path: str):
        """Save the trained scorer to disk."""
        torch.save({
            "mlp_state_dict": self.mlp.state_dict(),
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "rating_mean": self._rating_mean,
            "rating_std": self._rating_std,
            "n_dims": self.n_dims,
        }, path)
        print(f"Scorer saved to {path}")

    def load(self, path: str):
        """Load a trained scorer from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.n_dims = checkpoint["n_dims"]
        self.mlp = CalibrationMLP(n_dims=self.n_dims).to(self.device)
        self.mlp.load_state_dict(checkpoint["mlp_state_dict"])
        self.mlp.eval()
        self._feature_means = checkpoint["feature_means"]
        self._feature_stds = checkpoint["feature_stds"]
        self._rating_mean = checkpoint["rating_mean"]
        self._rating_std = checkpoint["rating_std"]
        self.is_trained = True
        print(f"Scorer loaded from {path}")


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing Neural Scorer with synthetic data...")
    print("(Replace with real TRIBE v2 outputs for actual experiments)\n")

    scorer = NeuralScorer()

    # Simulate 100 samples with 5 cognitive dimensions
    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, 5)

    # Create synthetic ratings that correlate with comprehension + attention
    ratings = (
        0.4 * features[:, 0]     # comprehension
        + 0.3 * features[:, 2]   # attention
        - 0.2 * features[:, 3]   # confusion (negative)
        + 0.1 * features[:, 4]   # dmn suppression
        + np.random.randn(n_samples) * 0.3  # noise
    )
    # Rescale to 1-7 Likert
    ratings = 4 + 1.5 * (ratings - ratings.mean()) / ratings.std()
    ratings = np.clip(ratings, 1, 7)

    print("Training calibration MLP...")
    result = scorer.train(features, ratings)

    print(f"\nModel has {scorer.mlp.n_parameters} parameters")

    # Test scoring
    test_dims = CognitiveDimensions(
        comprehension=0.8,
        memory_encoding=0.3,
        sustained_attention=0.6,
        confusion=0.1,
        dmn_suppression=0.5,
    )
    score = scorer.score_dims(test_dims)
    print(f"\nTest score for high-quality dims: {score:.2f}")

    test_dims_bad = CognitiveDimensions(
        comprehension=0.1,
        memory_encoding=0.1,
        sustained_attention=0.1,
        confusion=0.8,
        dmn_suppression=-0.3,
    )
    score_bad = scorer.score_dims(test_dims_bad)
    print(f"Test score for low-quality dims:  {score_bad:.2f}")
    print(f"Difference: {score - score_bad:.2f} (should be positive)")
