"""
harmonic_stabilizer.py
Minimal library for stabilizing recursive self-reflection in agents
via bounded harmonic updates on belief embeddings.

Usage:
    from harmonic_stabilizer import HarmonicAgent
    agent = HarmonicAgent(beliefs=[...])
    for _ in range(50):
        agent.reflect()
    agent.plot_drift()  # optional, requires matplotlib
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

try:
    # Lightweight model by default; users can swap to a larger one.
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception as e:
    _HAS_ST = False
    _IMPORT_ERR = e

try:
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception as e:
    _HAS_SK = False
    _IMPORT_ERR_SK = e


@dataclass
class HarmonicAgent:
    beliefs: Iterable[str]
    lambda_: float = 0.1
    omega: float = 1.0
    alpha: float = 0.05
    model_name: str = "all-MiniLM-L6-v2"
    seed: Optional[int] = 42
    _model: Optional["SentenceTransformer"] = field(init=False, default=None)
    _B0: Optional[np.ndarray] = field(init=False, default=None)
    _B: Optional[np.ndarray] = field(init=False, default=None)
    _drift_history: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if not _HAS_ST:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "    pip install sentence-transformers\n"
                f"Original import error: {_IMPORT_ERR}"
            )
        if not _HAS_SK:
            raise ImportError(
                "scikit-learn is required. Install with:\n"
                "    pip install scikit-learn\n"
                f"Original import error: {_IMPORT_ERR_SK}"
            )

        self._model = SentenceTransformer(self.model_name)
        if not self.beliefs:
            raise ValueError("`beliefs` must contain at least one string.")

        embeddings = self._model.encode(list(self.beliefs))
        self._B0 = embeddings.mean(axis=0)
        self._B = self._B0.copy()

    @property
    def drift_history(self) -> List[float]:
        return list(self._drift_history)

    def _current_drift(self) -> float:
        """Cosine distance between current embedding and initial centroid."""
        cs = cosine_similarity(self._B.reshape(1, -1), self._B0.reshape(1, -1))[0, 0]
        return float(1.0 - cs)

    def reflect(self) -> float:
        """
        One reflection step: apply bounded harmonic update to the belief vector.
        Returns the current drift value after the update.
        """
        t = len(self._drift_history) + 1
        # Synthetic correction vector (placeholder): gaussian noise as proxy
        # for a generic "self-assessment correction". In production, replace
        # with a domain-specific delta F(B_t).
        noise = np.random.randn(*self._B.shape) * 0.01

        g_t = math.exp(-self.alpha * t) * math.sin(self.omega * t)
        self._B = self._B + self.lambda_ * g_t * noise

        drift = self._current_drift()
        self._drift_history.append(drift)
        return drift

    def run(self, steps: int = 50) -> List[float]:
        """Run multiple reflection cycles and return the drift series."""
        for _ in range(steps):
            self.reflect()
        return self.drift_history

    def summarize(self) -> str:
        if not self._drift_history:
            return "No reflections yet. Call `run()` or `reflect()` first."
        arr = np.asarray(self._drift_history, dtype=float)
        return (
            f"Reflections: {len(arr)} | "
            f"Mean drift: {arr.mean():.6f} | "
            f"Variance: {arr.var():.6f} | "
            f"Min/Max drift: {arr.min():.6f}/{arr.max():.6f}"
        )

    def plot_drift(self, save_path: Optional[str] = None) -> None:
        """
        Plot the drift curve using matplotlib. Avoids picking a style or colors.
        If `save_path` is provided, saves to disk; otherwise shows interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            warnings.warn(
                "matplotlib is required for plotting. Install with:\n"
                "    pip install matplotlib\n"
                f"Original import error: {e}"
            )
            return

        if not self._drift_history:
            warnings.warn("No drift history to plot. Run `run()` first.")
            return

        plt.figure()
        plt.plot(range(1, len(self._drift_history) + 1), self._drift_history)
        plt.title("Recursive Belief Drift (bounded oscillations expected)")
        plt.xlabel("Reflection step")
        plt.ylabel("Cosine drift to initial belief")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
