# src/train.py

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransfomer
from sklearn.impute from 