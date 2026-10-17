# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.constants import *  # This imports all constants from constants.py

# Load dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))
# Extract feature columns (Temperature, pH, Glucose, Lactate)
features = df.columns[1:]
#Extract target column (Specific Growth Rate 1st column)
target = df.columns[0]
#convert to numpy
x = df.loc[:,features].to_numpy()
y = df.loc[:,target].to_numpy()
#split data into train and test sets
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

#__________
#Linear Regression (base model)
#__________
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
#predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
#performance metrics
train_rmse = root_mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Linear Regression Results")
print(f"Train | RMSE | {train_rmse:.4f} d^-1")
print(f"Test  | RMSE | {test_rmse:.4f} d^-1")
print()
print(f"Train | R^2  | {train_r2:.4f}")
print(f"Test  | R^2  | {test_r2:.4f}")
print()
#__________
#Polynomial Regression (Grid Search to find order)
#__________
poly_model = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
#test polynomial degrees from 1 to 6
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1,7)])
poly_gscv = GridSearchCV(poly_model, param_grid=param_grid_poly, cv=6, scoring="neg_root_mean_squared_error", return_train_score=True)
poly_gscv.fit(x_train, y_train)

#put results in a dataframe
os.makedirs("tables", exist_ok=True)
df_poly = pd.DataFrame(poly_gscv.cv_results_)
df_poly.to_csv("tables/polynomial_grid_search.csv", index=False)
columns_view = ["rank_test_score", "param_polynomialfeatures__degree","mean_test_score","std_test_score","mean_train_score","std_train_score"]
print("Polynomial Grid Search Results")
print(df_poly.loc[:, columns_view].to_string())
print()
print(f"Final Test RMSE: {-poly_gscv.score(x_test, y_test):.4f}d^-1") #made this positive so RMSE values easier to read/interpret
print()

#__________
#Cross Validate Bias_Variance Tradeoff
#__________
print("Bias_Variance Tradeoff")
print("Degree | Train RMSE | Validation RMSE | Train R^2 | Validation R^2")
for degree in range(1,7):
    bias_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    cv_results = cross_validate(bias_model, x_train,y_train,cv=4, scoring=("neg_root_mean_squared_error","r2"), return_train_score=True)

    train_rmse = -cv_results["train_neg_root_mean_squared_error"]
    train_r2 = cv_results["train_r2"]
    valid_rmse = -cv_results["test_neg_root_mean_squared_error"]
    valid_r2 = cv_results["test_r2"]
    print(f"Degree|{degree:6d}|"f"{np.mean(train_rmse):10.4f}|"f"{np.mean(valid_rmse):15.4f}|"f"{np.mean(train_r2):9.4f}|"f"{np.mean(valid_r2):13.4f}")
print()

#__________
#k-Nearest Neighbors (kNN) Regression
#__________
kNN_model = make_pipeline(StandardScaler(),KNeighborsRegressor())
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1,21)], kneighborsregressor__weights=["uniform","distance"]) #I chose 20 K neighbors based on research online
#make model using gridsearch
model_gscv_knn = GridSearchCV(kNN_model, param_grid=param_grid_knn, cv=6, scoring="neg_root_mean_squared_error", return_train_score=True)
model_gscv_knn.fit(x_train, y_train)

#put results in a dataframe
os.makedirs("tables", exist_ok=True)
df_knn=pd.DataFrame(model_gscv_knn.cv_results_)
df_knn.to_csv("tables/knn_grid_search.csv", index=False)
columns_view=["rank_test_score","param_kneighborsregressor__n_neighbors","param_kneighborsregressor__weights","mean_test_score","std_test_score","mean_train_score","std_train_score"]
print("kNN Regression GridSearch Results")
print(df_knn.loc[:, columns_view].to_string())
print()

print(f"Final Test RMSE: {-model_gscv_knn.score(x_test, y_test):.4f} d^-1") #made positive so score easier to read
print()

#========
#Polynomial Validation Curve
#========
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

degrees = df_poly["param_polynomialfeatures__degree"].to_numpy()
rmse_train = -df_poly["mean_train_score"].to_numpy()
rmse_valid = -df_poly["mean_test_score"].to_numpy()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
ax.plot(degrees, rmse_train, marker="o", label="Train RMSE")
ax.plot(degrees, rmse_valid, marker="d", linestyle="--", label="Validation RMSE")
ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
ax.set_ylabel("RMSE [d^-1]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(0, 1)
fig.savefig(os.path.join("figures", "figure_validation_curve_poly.png"))
plt.close(fig)

#========
#KNN Validation Curve
#========
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
for i, weight in enumerate(["uniform","distance"],start=1):
    mask_weight = (df_knn.loc[:,"param_kneighborsregressor__weights"]==weight)
    df_sub = df_knn.loc[mask_weight,:]
    df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'],ascending=True)
    n_neighbors = df_sub.loc[:,"param_kneighborsregressor__n_neighbors"].to_numpy()
    rmse_train = -df_sub.loc[:,"mean_train_score"].to_numpy()
    rmse_test = -df_sub.loc[:,"mean_test_score"].to_numpy()

    ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})", color=COLORS[i], linestyle="-", marker="o" if weight=="distance" else None)
    ax.plot(n_neighbors, rmse_test, label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle="--")

ax.set_xlabel("k", fontsize=fontsize_xy_label)
ax.set_ylabel("RMSE [d^-1]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
fig.savefig(os.path.join("figures", "figure_validation_curve_knn.png"))
plt.close(fig)

#__________
#Learning Curves (for both polynomial and kNN regression -note that cell growth dataset has 4 input features!)
#__________
print("Generating Learning curve plot")
os.makedirs("figures", exist_ok=True)
#Grid search
kwargs_gscv = dict(cv=6, scoring="neg_root_mean_squared_error", return_train_score=True )
#polynomial regression (degrees 1 to 6)
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1,7)])
#kNN regression (k values from 1-20 w/ both weighting (uniform and distance))
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1,21)], kneighborsregressor__weights=["uniform","distance"])
#random subsets
rng = np.random.default_rng(seed=1)
#store RMSE values at each sample size
lc_n_training_samples = []
lc_rmse_poly_train = []
lc_rmse_knn_train = []
lc_rmse_poly_test = []
lc_rmse_knn_test = []

#randomly select subset of training data. I use increments of 30 up to 240
sample_sizes = range(30, min(241, len(x_train)+1),30)
for n_samples in sample_sizes:
    idxs = rng.choice(len(x_train), size=n_samples, replace=False)
    x_sample = x_train[idxs]
    y_sample = y_train[idxs]

    # __________
    # Polynomial!
    # __________
    model_poly = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
    gscv_poly = GridSearchCV(model_poly, param_grid_poly,**kwargs_gscv)
    gscv_poly.fit(x_sample, y_sample)

    # __________
    # kNN!
    # __________
    model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
    gscv_knn = GridSearchCV(model_knn, param_grid=param_grid_knn, **kwargs_gscv)
    gscv_knn.fit(x_sample, y_sample)

    # __________
    # calc train and test RMSE
    # __________
    rmse_poly_train = root_mean_squared_error(y_sample, gscv_poly.predict(x_sample))
    rmse_knn_train = root_mean_squared_error(y_sample, gscv_knn.predict(x_sample))
    rmse_poly_test = root_mean_squared_error(y_test, gscv_poly.predict(x_test))
    rmse_knn_test = root_mean_squared_error(y_test, gscv_knn.predict(x_test))

    lc_n_training_samples.append(n_samples)
    lc_rmse_poly_train.append(rmse_poly_train)
    lc_rmse_knn_train.append(rmse_knn_train)
    lc_rmse_poly_test.append(rmse_poly_test)
    lc_rmse_knn_test.append(rmse_knn_test)

#plot and save learning curves
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200, layout="constrained")
ax.plot(lc_n_training_samples, lc_rmse_poly_train, marker ="o", label="RMSE Train (Polynomial)", color="teal", linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_poly_test, marker ="o", label="RMSE Test (Polynomial)", color="teal", linestyle="--")
ax.plot(lc_n_training_samples, lc_rmse_knn_train, marker ="s", label="RMSE Train (kNN)", color="magenta", linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_knn_test, marker ="s", label="RMSE Test (kNN)", color="magenta", linestyle="--")
ax.set_xlabel(f"Number of training samples")
ax.set_ylabel(f"RMSE [d^-1]")
ax.set_title(f"Learning Curves: Polynomial Regression vs KNN")
ax.legend()
ax.grid(alpha=0.3)

fig.savefig(os.path.join("figures", f"figure_learning_curve.png"))
plt.show()
plt.close(fig)

print(f"Training samples: {n_samples}")
print(f"Polynomial best degree: "f"{gscv_poly.best_params_['polynomialfeatures__degree']}")
print(f"Train RMSE: {rmse_poly_train:.4f} d^-1 | "f"Test RMSE: {rmse_poly_test:.4f} d^-1")
print(f"KNN best parameters: "f"k = {gscv_knn.best_params_['kneighborsregressor__n_neighbors']}, "f"weights = {gscv_knn.best_params_['kneighborsregressor__weights']}")
print(f"Train RMSE: {rmse_knn_train:.4f} d^-1 | "f"Test RMSE: {rmse_knn_test:.4f} d^-1")

#=====================
#Final Summary Table of Model Performance
#=====================
print()
print()
print ("=" * 85)
print(f"{'Model Type':<25} | {'Best Settings':<20} | {'Test RMSE (d^-1)':<18} | {'Test R^2':<10}")
print("-" * 85)
#1. Linear Regression (Base)
print(f"{'Linear Regression':<25} | {'Baseline':<20} | {test_rmse:<18.4f} | {test_r2:<10.4f}")
#2. Optimized Polynomial Regression
best_poly_model = poly_gscv.best_estimator_
poly_test_preds = best_poly_model.predict(x_test)
poly_test_rmse = root_mean_squared_error(y_test, poly_test_preds)
poly_test_r2 = r2_score(y_test, poly_test_preds)
poly_degree = poly_gscv.best_params_['polynomialfeatures__degree']
print(f"{'Polynomial Regression':<25} | {'Degree = ' + str(poly_degree):<20} | {poly_test_rmse:<18.4f} | {poly_test_r2:<10.4f}")
#3. Optimized kNN Regression
best_knn_model = model_gscv_knn.best_estimator_
knn_test_preds = best_knn_model.predict(x_test)
knn_test_rmse = root_mean_squared_error(y_test, knn_test_preds)
knn_test_r2 = r2_score(y_test, knn_test_preds)
knn_k = model_gscv_knn.best_params_['kneighborsregressor__n_neighbors']
knn_weight = model_gscv_knn.best_params_['kneighborsregressor__weights']
knn_settings = f"k={knn_k}, {knn_weight}"
print(f"{'kNN Regression':<25} | {knn_settings:<20} | {knn_test_rmse:<18.4f} | {knn_test_r2:<10.4f}")
print("=" * 85)
#CSV file
summary_table = pd.DataFrame({"Model Type": ["Linear Regression","Polynomial Regression","kNN Regression"],"Best Settings": ["Baseline",f"Degree = {poly_degree}",f"k = {knn_k}, {knn_weight}"],"Test RMSE (d^-1)": [test_rmse,poly_test_rmse,knn_test_rmse],"Test R^2": [test_r2,poly_test_r2,knn_test_r2]})
# Save the table in the tables folder
os.makedirs("tables", exist_ok=True)
summary_table.to_csv("tables/model_performance_summary.csv", index=False)
print("Summary table saved to tables/model_performance_summary.csv")
