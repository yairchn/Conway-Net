"""Conway-Net: cellular-automaton lateral inhibition (LI=CA) for image classification."""
import os

# the repository root (the folder holding conwaynet/); the paper scripts read and write their files under it
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .config import load_config
from .data import load_dataset
from .lateral_inhibition import LateralInhibitionCA, lateral_inhibition, make_gate
from .models import build_model
from .train import train

__all__ = ["REPO_ROOT", "load_config", "load_dataset", "build_model", "train",
           "lateral_inhibition", "LateralInhibitionCA", "make_gate"]
