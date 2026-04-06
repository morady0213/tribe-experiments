"""
TRIBE v2 Wrapper — Clean interface for text-only brain prediction.

This wraps Meta's TRIBE v2 model to provide a simple API:
    wrapper = TribeWrapper()
    activation_map = wrapper.predict_text("Your text here")

The output is a numpy array of predicted BOLD fMRI activations
across cortical vertices (fsaverage5 surface mesh, ~20,484 vertices).
"""

import sys
import os
import json
import time
import tempfile
import argparse
import pathlib
import numpy as np
import torch

# Fix PosixPath on Windows: the TRIBE v2 config.yaml contains serialized
# PosixPath objects that can't be instantiated on Windows.
if sys.platform == "win32":
    _original_posix_new = pathlib.PosixPath.__new__
    def _patched_posix_new(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
    pathlib.PosixPath.__new__ = _patched_posix_new

# Add tribev2 repo to path
TRIBEV2_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tribev2")
if os.path.exists(TRIBEV2_DIR):
    sys.path.insert(0, TRIBEV2_DIR)


class TribeWrapper:
    """Clean wrapper around TRIBE v2 for text-only inference."""

    def __init__(
        self,
        weights_dir: str = None,
        device: str = "auto",
        text_only: bool = True,
        cache_folder: str = None,
    ):
        """
        Args:
            weights_dir: Path to downloaded TRIBE v2 weights.
                         Defaults to ../weights/tribev2/
            device: "auto", "cuda", "cpu"
            text_only: If True, only loads the text encoder (saves VRAM).
            cache_folder: Directory for caching extracted features.
                          Defaults to ../cache/
        """
        self.text_only = text_only
        self.device = self._resolve_device(device)
        self.weights_dir = weights_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "weights", "tribev2"
        )
        self.cache_folder = cache_folder or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cache"
        )
        os.makedirs(self.cache_folder, exist_ok=True)

        print(f"[TribeWrapper] Device: {self.device}")
        print(f"[TribeWrapper] Text-only mode: {self.text_only}")
        print(f"[TribeWrapper] Weights dir: {self.weights_dir}")

        self.model = None
        self.n_vertices = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load TRIBE v2 model via TribeModel.from_pretrained()."""
        print("[TribeWrapper] Loading model...")
        load_start = time.time()

        try:
            from tribev2.demo_utils import TribeModel

            self.model = TribeModel.from_pretrained(
                checkpoint_dir=self.weights_dir,
                cache_folder=self.cache_folder,
                device=self.device,
            )
            elapsed = time.time() - load_start
            print(f"[TribeWrapper] Loaded via TribeModel.from_pretrained() in {elapsed:.1f}s")

        except Exception as e:
            print(f"\n[TribeWrapper] Failed to load TRIBE v2: {e}")
            raise

    def predict_text(self, text: str) -> np.ndarray:
        """Predict brain activation from text input.

        Args:
            text: Input text (a sentence, paragraph, or full response).

        Returns:
            np.ndarray of shape (n_vertices,) — predicted BOLD activation
            per cortical vertex (fsaverage5), averaged over time segments.
        """
        # Write text to a temp file (TribeModel expects a file path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            text_path = f.name

        try:
            # Build events DataFrame from text (text -> TTS -> transcription -> events)
            events = self.model.get_events_dataframe(text_path=text_path)

            # Run inference: returns (n_segments, n_vertices), list_of_segments
            preds, segments = self.model.predict(events, verbose=False)

            # Average across time segments to get a single activation map
            if preds.ndim == 2:
                activation = preds.mean(axis=0)
            elif preds.ndim == 1:
                activation = preds
            else:
                activation = preds.mean(axis=tuple(range(preds.ndim - 1)))

            self.n_vertices = len(activation)
            return activation

        finally:
            os.unlink(text_path)

    def predict_text_temporal(self, text: str) -> tuple[np.ndarray, list]:
        """Predict brain activation with temporal resolution.

        Returns:
            preds: np.ndarray of shape (n_segments, n_vertices)
            segments: list of segment objects
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            text_path = f.name

        try:
            events = self.model.get_events_dataframe(text_path=text_path)
            preds, segments = self.model.predict(events, verbose=False)
            self.n_vertices = preds.shape[1] if preds.ndim == 2 else len(preds)
            return preds, segments
        finally:
            os.unlink(text_path)

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        """Predict brain activations for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape (n_texts, n_vertices).
        """
        results = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i+1}/{len(texts)}...")
            results.append(self.predict_text(text))
        return np.stack(results)

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        info = {
            "device": str(self.device),
            "text_only": self.text_only,
            "weights_dir": self.weights_dir,
            "n_vertices": self.n_vertices,
        }

        if self.device == "cuda":
            info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1e9, 2
            )
            info["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1e9, 2
            )

        return info


def run_smoke_test():
    """Experiment 0: Verify TRIBE v2 is working."""
    print("=" * 60)
    print("  EXPERIMENT 0: Smoke Test")
    print("=" * 60)
    print()

    # Load model
    wrapper = TribeWrapper(text_only=True)
    print()

    # Test input
    test_text = (
        "The mitochondria is the powerhouse of the cell. "
        "It converts nutrients into adenosine triphosphate, or ATP, "
        "which provides energy for cellular processes."
    )

    print(f"Input text: '{test_text[:80]}...'")
    print()

    # Run inference
    print("Running inference...")
    start = time.time()
    activation = wrapper.predict_text(test_text)
    elapsed = time.time() - start

    # Report results
    print()
    print("RESULTS:")
    print(f"  Output shape:     {activation.shape}")
    print(f"  N vertices:       {len(activation)}")
    print(f"  Value range:      [{activation.min():.3f}, {activation.max():.3f}]")
    print(f"  Mean:             {activation.mean():.4f}")
    print(f"  Std:              {activation.std():.4f}")
    print(f"  Inference time:   {elapsed:.3f}s")
    print()

    # Sanity checks
    checks_passed = True

    if len(activation) < 100:
        print("  WARNING: Very few vertices. Expected ~20,484.")
        checks_passed = False

    if activation.std() < 0.001:
        print("  WARNING: Very low variance. Output may be degenerate.")
        checks_passed = False

    if checks_passed:
        print("  All sanity checks passed!")

    # Save results
    info = wrapper.get_model_info()
    info["test_text"] = test_text
    info["output_shape"] = list(activation.shape)
    info["value_range"] = [float(activation.min()), float(activation.max())]
    info["inference_time_s"] = elapsed

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "experiment_0.json")
    with open(results_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    np.save(os.path.join(results_dir, "experiment_0_activation.npy"), activation)
    print(f"  Activation saved to: {results_dir}/experiment_0_activation.npy")
    print()
    print("Experiment 0 complete. Proceed to Experiment 1.")
    return activation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRIBE v2 Wrapper")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    args = parser.parse_args()

    if args.test:
        run_smoke_test()
    else:
        print("Usage: python tribe_wrapper.py --test")
        print("Or import TribeWrapper in your own scripts.")
