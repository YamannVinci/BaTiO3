from Functions import *

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
df = load_data()
cat_cols, num_cols, out_col = col_type(df)
df[cat_cols] = encoder(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(object)
X = pd.concat([df[num_cols], df[cat_cols]], axis='columns')
X_s = feature_scaling(X)
Y = df['Wrec', 'Neta']
Y = Y.to_numpy()
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
kf_init = 10
# 100 folds total (10 x 10)
r2 = np.zeros([100, 6])
mse = np.zeros([100, 6])
rmse = np.zeros([100, 6])
mae = np.zeros([100, 6])
Y_test_all = []
y_xgb_all = []
time_start = time.time()
i = 0
fold_shuffle = np.random.randint(10, 100, 10)
for j in range(kf_init):
    # K-Fold split
    kf = KFold(
        n_splits=10,
        random_state=fold_shuffle[j],
        shuffle=True
    )
    for train_index, test_index in kf.split(X_s, Y):
        X_train, X_test = X_s[train_index], X_s[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_inv = np.exp(Y_test)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train,
            Y_train,
            test_size=0.2,
            random_state=42
        )
        Y_test_all.append(Y_inv)

        #1. LINEAR REGRESSION
        lr_model = LinearRegression()
        lr_model.fit(X_train, Y_train.ravel())
        y_pred_lr = lr_model.predict(X_test)
        y_lr_inv = np.exp(y_pred_lr).reshape(-1, 1)
        r2[i, 0] = r2_score(Y_inv, y_lr_inv)
        mse[i, 0] = mean_squared_error(Y_inv, y_lr_inv)
        rmse[i, 0] = RMSE(Y_inv, y_lr_inv)
        mae[i, 0] = mean_absolute_error(Y_inv, y_lr_inv)

       # 2. SUPPORT VECTOR REGRESSION
        svr_model = SVR(
            kernel='rbf',
            C=100,
            gamma=0.1,
            epsilon=0.01
        )
        svr_model.fit(X_train, Y_train.ravel())
        y_pred_svr = svr_model.predict(X_test)
        y_svr_inv = np.exp(y_pred_svr).reshape(-1, 1)
        r2[i, 1] = r2_score(Y_inv, y_svr_inv)
        mse[i, 1] = mean_squared_error(Y_inv, y_svr_inv)
        rmse[i, 1] = RMSE(Y_inv, y_svr_inv)
        mae[i, 1] = mean_absolute_error(Y_inv, y_svr_inv)

        # 3. DECISION TREE
        dt_model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        dt_model.fit(X_train, Y_train)
        y_pred_dt = dt_model.predict(X_test)
        y_dt_inv = np.exp(y_pred_dt).reshape(-1, 1)
        r2[i, 2] = r2_score(Y_inv, y_dt_inv)
        mse[i, 2] = mean_squared_error(Y_inv, y_dt_inv)
        rmse[i, 2] = RMSE(Y_inv, y_dt_inv)
        mae[i, 2] = mean_absolute_error(Y_inv, y_dt_inv)

        # 4. RANDOM FOREST
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, Y_train.ravel())
        y_pred_rf = rf_model.predict(X_test)
        y_rf_inv = np.exp(y_pred_rf).reshape(-1, 1)
        r2[i, 3] = r2_score(Y_inv, y_rf_inv)
        mse[i, 3] = mean_squared_error(Y_inv, y_rf_inv)
        rmse[i, 3] = RMSE(Y_inv, y_rf_inv)
        mae[i, 3] = mean_absolute_error(Y_inv, y_rf_inv)

        # 5. GRADIENT BOOSTING REGRESSION TREES
        gbrt_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        gbrt_model.fit(X_train, Y_train.ravel())
        y_pred_gbrt = gbrt_model.predict(X_test)
        y_gbrt_inv = np.exp(y_pred_gbrt).reshape(-1, 1)
        r2[i, 4] = r2_score(Y_inv, y_gbrt_inv)
        mse[i, 4] = mean_squared_error(Y_inv, y_gbrt_inv)
        rmse[i, 4] = RMSE(Y_inv, y_gbrt_inv)
        mae[i, 4] = mean_absolute_error(Y_inv, y_gbrt_inv)

        # 6. XGBOOST
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, Y_train.ravel())
        y_pred_xgb = xgb_model.predict(X_test)
        y_xgb_inv = np.exp(y_pred_xgb).reshape(-1, 1)
        y_xgb_all.append(y_xgb_inv)
        r2[i, 5] = r2_score(Y_inv, y_xgb_inv)
        mse[i, 5] = mean_squared_error(Y_inv, y_xgb_inv)
        rmse[i, 5] = RMSE(Y_inv, y_xgb_inv)
        mae[i, 5] = mean_absolute_error(Y_inv, y_xgb_inv)
        i += 1
time_end = time.time()
print(
    "Elapsed time: {} minutes and {:.0f} seconds".format(
        int((time_end - time_start) // 60),
        (time_end - time_start) % 60
    )
)
df_r2 = pd.DataFrame(r2).rename(columns={
    0: 'Linear Regression',
    1: 'SVR',
    2: 'Decision Tree',
    3: 'Random Forest',
    4: 'GBRT',
    5: 'XGBoost'
})
df_mse = pd.DataFrame(mse).rename(columns={
    0: 'Linear Regression',
    1: 'SVR',
    2: 'Decision Tree',
    3: 'Random Forest',
    4: 'GBRT',
    5: 'XGBoost'
})
df_rmse = pd.DataFrame(rmse).rename(columns={
    0: 'Linear Regression',
    1: 'SVR',
    2: 'Decision Tree',
    3: 'Random Forest',
    4: 'GBRT',
    5: 'XGBoost'
})
df_mae = pd.DataFrame(mae).rename(columns={
    0: 'Linear Regression',
    1: 'SVR',
    2: 'Decision Tree',
    3: 'Random Forest',
    4: 'GBRT',
    5: 'XGBoost'
})
df_r2.to_excel('100-fold_R2_All_Models.xlsx', index=False)
df_mse.to_excel('100-fold_MSE_All_Models.xlsx', index=False)
df_rmse.to_excel('100-fold_RMSE_All_Models.xlsx', index=False)
df_mae.to_excel('100-fold_MAE_All_Models.xlsx', index=False)
print("\nAverage R2 Scores:\n")
print(df_r2.mean())
print("\nAverage RMSE Values:\n")
print(df_rmse.mean())
print("\nAverage MAE Values:\n")
print(df_mae.mean())
print("\nFirst 20 rows of R2 results:\n")
print(df_r2.head(20))
