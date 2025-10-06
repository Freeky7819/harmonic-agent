"""
harmonic_demo.py
Minimal runnable demo for Harmonic Stabilization against Recursive Belief Drift.

Run:
    python harmonic_demo.py --steps 50 --save-plot drift.png

Requirements:
    pip install sentence-transformers numpy scikit-learn matplotlib
"""

import argparse
from pathlib import Path

from harmonic_stabilizer import HarmonicAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Reflection cycles.")
    parser.add_argument("--lambda_", type=float, default=0.1, help="Update rate.")
    parser.add_argument("--omega", type=float, default=1.0, help="Oscillation frequency.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Exponential damping.")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model.")
    parser.add_argument("--save-plot", type=str, default="drift.png", help="Where to save the drift plot.")
    args = parser.parse_args()

    agent = HarmonicAgent(
        beliefs=["I value truth", "I prioritize safety", "I assist humans"],
        lambda_=args.lambda_,
        omega=args.omega,
        alpha=args.alpha,
        model_name=args.model,
    )

    agent.run(steps=args.steps)
    print(agent.summarize())

    # Save plot next to the script output path
    out_path = Path(args.save_plot)
    agent.plot_drift(save_path=str(out_path))
    print(f"Drift plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
