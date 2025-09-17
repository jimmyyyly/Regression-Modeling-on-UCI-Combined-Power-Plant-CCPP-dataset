import argparse
import numpy as np

# (1) Import Libraries
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Helper function for evaluation for step 6
def evaluate(y_true, y_pred, name, p = None):
    # (6) Evaluates Model Performance : Prints MAE, MSE, RMSE, R^2, and Adjusted R^2 if p provided
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    line = f"{name:18s} MAE = {mae:3f} MSE = {mse:3f} RMSE = {rmse:3f} R2 = {r2:4f}"
    if p is not None:
        n = len(y_true)
        adjr2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        line += f" AdjR2={adjr2:.4f}"
    print(line)


def main():
    parser = argparse.ArgumentParser(description = "CS577 Regression Models on CCPP dataset")
    parser.add_argument("--test-size", type = float, default = 0.20, help = "Test split size fraction (default 0.20)")
    parser.add_argument("--random-state", type = int, default = 42, help = "Random seed for reproducibility (default 42)")
    parser.add_argument("--poly-degree", type = int, default = 2, help = "Polynomial degree for polynomial regression (default 2)")
    parser.add_argument("--rf-estimators", type = int, default = 300, help = "Number of trees for RandomForest (default 300)")
    parser.add_argument("--rf-min-samples-leaf", type = int, default = 2, help = "min_samples_leaf for RandomForest (default 2)")
    parser.add_argument("--dt-min-samples-leaf", type = int, default = 3, help = "min_samples_leaf for DecisionTree (default 3)")
    args = parser.parse_args()

    # (2) Import Dataset
    # Combined Cycle Power Plant dataset (ID 294)
    ccpp = fetch_ucirepo(id = 294)
    X = ccpp.data.features              # Features: T, AP, RH, V
    y = ccpp.data.targets.squeeze()     # Target: EP

    # (3) Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = args.test_size,
        random_state = args.random_state
    )

    print("Part A: RF, DT, Multiple Linear, Polynomial")

    # (4) Train Multiple Linear Regression Model
    lin = LinearRegression().fit(X_train, y_train)
    # (5) Predict Test Result
    yhat_lin = lin.predict(X_test)
    # (6) Evaluate
    evaluate(y_test, yhat_lin, "LinearRegression", p = X.shape[1])

    # (4) Train Polynomial Regression Model
    poly2 = Pipeline([
        ("poly", PolynomialFeatures(degree = args.poly_degree, include_bias = False)),
        ("lin", LinearRegression())
    ]).fit(X_train, y_train)
    # (5) Predict
    yhat_poly = poly2.predict(X_test)
    # (6) Evaluate
    p_poly = poly2.named_steps["poly"].n_output_features_
    evaluate(y_test, yhat_poly, f"Polynomial (deg = {args.poly_degree})", p = p_poly)

    # (4) Train Decision Tree Model
    dt = DecisionTreeRegressor(random_state = args.random_state, min_samples_leaf = args.dt_min_samples_leaf).fit(X_train, y_train)
    # (5) Predict
    yhat_dt = dt.predict(X_test)
    # (6) Evaluate
    evaluate(y_test, yhat_dt, "DecisionTree")

    # (4) Train Random Forest model
    rf = RandomForestRegressor(
        n_estimators = args.rf_estimators,
        min_samples_leaf = args.rf_min_samples_leaf,
        n_jobs = -1,
        random_state = args.random_state
    ).fit(X_train, y_train)
    # (5) Predict
    yhat_rf = rf.predict(X_test)
    # (6) Evaluate
    evaluate(y_test, yhat_rf, "RandomForest")

    # Part B
    print("Part B: Support Vector Regression")

    # (5) Train SVR model
    svr_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel = "rbf", C = 10.0, epsilon = 0.1, gamma = "scale"))
    ]).fit(X_train, y_train)
    # (6) Predict Test result set
    yhat_svr = svr_rbf.predict(X_test)
    # (7) Evaluate Model Performance
    evaluate(y_test, yhat_svr, "SVR (RBF)")


if __name__ == "__main__":
    main()
