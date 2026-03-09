"""
Integration 2 — CI Autograder Tests

DO NOT MODIFY THIS FILE. It is used by the GitHub Actions autograder.
"""

import subprocess
import sys
from pathlib import Path


def test_train_file_exists():
    """train.py must exist at the repo root."""
    assert Path("train.py").exists(), "train.py not found"


def test_train_runs_end_to_end():
    """python train.py must run without errors."""
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"


def test_training_output_shows_loss():
    """Training output must print 'Epoch' and 'Loss' values."""
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=60
    )
    output = result.stdout.lower()
    assert "epoch" in output and "loss" in output, \
        "Training output must print 'Epoch' and 'Loss' values"


def test_loss_decreases():
    """Final reported loss must be lower than the first reported loss."""
    import re
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=60
    )
    losses = re.findall(r"[Ll]oss\s*[=:]\s*([\d.eE+\-]+)", result.stdout)
    assert len(losses) >= 2, \
        f"Expected >=2 loss values in output, found {len(losses)}: {result.stdout[:500]}"
    first_loss = float(losses[0])
    last_loss = float(losses[-1])
    assert last_loss < first_loss, \
        f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"


def test_predictions_file_exists():
    """predictions.csv must be created after training."""
    subprocess.run([sys.executable, "train.py"], capture_output=True, timeout=60)
    assert Path("predictions.csv").exists(), \
        "predictions.csv not found — make sure train.py saves it after training"


def test_predictions_has_correct_columns():
    """predictions.csv must have 'actual' and 'predicted' columns."""
    import pandas as pd
    subprocess.run([sys.executable, "train.py"], capture_output=True, timeout=60)
    df = pd.read_csv("predictions.csv")
    assert "actual" in df.columns, "'actual' column missing from predictions.csv"
    assert "predicted" in df.columns, "'predicted' column missing from predictions.csv"
