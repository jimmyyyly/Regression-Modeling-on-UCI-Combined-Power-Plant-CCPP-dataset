## CS577 Quiz Regression

This repo trains and evaluates several regression models on the UCI Combined Cycle Power Plant (CCPP) dataset using scikit-learn.

### Models
- **Linear Regression**
- **Polynomial Regression** (configurable degree)
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Regressor (RBF)**

### Dataset
- Fetched automatically via `ucimlrepo` (UCI ID: 294). No manual download needed.

### Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
python -m pip install -r requirements.txt
```

If you don't use virtual environments, ensure `pip` installs into your current Python environment.

### Usage
Run with defaults:
```bash
python main.py
```

Optional arguments:
```bash
python main.py \
  --test-size 0.2 \
  --random-state 42 \
  --poly-degree 2 \
  --rf-estimators 300 \
  --rf-min-samples-leaf 2 \
  --dt-min-samples-leaf 3
```

These options control the train/test split, polynomial degree, and key tree/forest hyperparameters.

### Output
The script prints MAE, MSE, RMSE, R2, and AdjR2 (when applicable) for each model on the held-out test set.

### Notes
- Requires internet access on first run to download the dataset metadata via `ucimlrepo`.
- Tested with:
  - `numpy 2.3.2`
  - `pandas 2.3.2`
  - `scikit-learn 1.7.2`
  - `ucimlrepo 0.0.7`


