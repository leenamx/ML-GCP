import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostRegressor
from lifelines import CoxPHFitter
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold


data = pd.read_csv('filtered_data.csv')

X = data.drop(['patient_id', 'id_suffix', 'OS', 'OS_time'], axis=1)
Y = data['OS_time']

scaler = StandardScaler()
numerical_features = X.columns

RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

X_scaled = scaler.fit_transform(X)

# Define the cross-validation parameters
cv = StratifiedKFold(n_splits=5)

# -------------- Random Forest ------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Linear search range for parameters
rf_param_grid = {
    'n_estimators': [10, 30, 50, 100, 200, 300],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2']
}

rf_selected_features = []
rf_selected_features_cnt = {}
rf_mse = []

for train_index, test_index in RSKF.split(X, Y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Perform linear search for survival time OS_time
    grid_search = GridSearchCV(rf, rf_param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train)

    # Model using the best parameters
    best_svm_regressor = grid_search.best_estimator_

    # Evaluate the performance of the best model
    rf_train_pred = best_svm_regressor.predict(X_train)
    rf_train_mse = mean_squared_error(Y_train, rf_train_pred)

    rf_test_pred = best_svm_regressor.predict(X_test)
    rf_test_mse = mean_squared_error(Y_test, rf_test_pred)
    rf_mse.append((rf_test_mse, rf_train_mse))

    # Print feature importance
    importances = best_svm_regressor.feature_importances_
    features = X.columns
    importance_dict = dict(zip(features, importances))
    rf_sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    rf_feature_names = [item[0] for item in rf_sorted_importance[:20]]

rf_combined_data = list(zip(rf_mse, rf_selected_features))
rf_combined_data.sort(key=lambda x: x[0][0])

rf_top3_test_mse = [mse[0] for mse, _ in rf_combined_data[:3]]
rf_top3_train_mse = [mse[1] for mse, _ in rf_combined_data[:3]]

rf_top3_features_group = [features for _, features in rf_combined_data[:3]]
rf_top3_features_group = [feature for sublist in rf_top3_features_group for feature in sublist]


for feature_name in rf_top3_features_group:
    if feature_name in rf_selected_features_cnt.keys():
        rf_selected_features_cnt[feature_name] += 1
    else:
        rf_selected_features_cnt[feature_name] = 1

rf_sorted_top20_features = dict(sorted(rf_selected_features_cnt.items(), key=lambda item: item[1], reverse=True)[:20])


# -------------- Cox -----------------

cph = CoxPHFitter(penalizer=0.1)  # Penalizer for regularization to avoid overfitting; can be adjusted as needed

df = pd.read_csv('source_data_integrated_v2_drop_PGAM2.csv')  
tcga_numerical_features = df.drop(['patient_id', 'id_suffix', 'OS', 'OS_time'], axis=1).columns
tcga = df.drop(['patient_id', 'id_suffix'], axis=1)
tcga_scaled = scaler.fit_transform(df[tcga_numerical_features])
tcga_scaled = pd.DataFrame(tcga_scaled, columns=tcga_numerical_features)
tcga_scaled[['OS', 'OS_time']] = df[['OS', 'OS_time']]


cph.fit(tcga_scaled, duration_col='OS_time', event_col='OS', show_progress=True)

# Output feature importance
tcga_feature_importance = abs(cph.summary.loc[:, ['coef']]).sort_values('coef', ascending=False)

tcga_feature_name_abs = list(tcga_feature_importance['coef'][:20].index)

# -------------- XGBOOST ------------------

xgb_selected_features = []
xgb_selected_features_cnt = {}
xgb_mse = []

for train_index, test_index in RSKF.split(X, Y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
    dtrain = xgb.DMatrix(X_train, label=Y_train)

    xgb_params = {
        'max_depth': 3,  
        'eta': 0.3,      
        'subsample': 0.7, 
        'colsample_bytree': 0.8, 
    }

    num_boost_round = 100  # Total number of iterations
    cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=num_boost_round, nfold=5,
                        show_stdv=True, seed=42) 

    bst = xgb.train(xgb_params, dtrain, num_boost_round)

    xgb_train_pred = bst.predict(xgb.DMatrix(X_train))
    xgb_train_mse = mean_squared_error(Y_train, xgb_train_pred)

    xgb_test_pred = bst.predict(xgb.DMatrix(X_test))
    xgb_test_mse = mean_squared_error(Y_test, xgb_test_pred)
    xgb_mse.append((xgb_test_mse, xgb_train_mse))

    # Feature importance
    importance = bst.get_score(importance_type='weight')
    xgb_feature_importance = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    # xgb_feature_importance['Feature'] = X.columns
    xgb_feature_importance = xgb_feature_importance.sort_values(by='Importance', ascending=False)
    xgb_feature_names = list(xgb_feature_importance['Feature'][:20])
    xgb_selected_features.append(xgb_feature_names)


xgb_combined_data = list(zip(xgb_mse, xgb_selected_features))
xgb_combined_data.sort(key=lambda x: x[0][0])
xgb_top3_test_mse = [mse[0] for mse, _ in xgb_combined_data[:3]]
xgb_top3_train_mse = [mse[1] for mse, _ in xgb_combined_data[:3]]
xgb_top3_features_group = [features for _, features in xgb_combined_data[:3]]
xgb_top3_features_group = [feature for sublist in xgb_top3_features_group for feature in sublist]


for feature_name in xgb_top3_features_group:
    if feature_name in xgb_selected_features_cnt.keys():
        xgb_selected_features_cnt[feature_name] += 1
    else:
        xgb_selected_features_cnt[feature_name] = 1

xgb_sorted_top20_features = dict(sorted(xgb_selected_features_cnt.items(), key=lambda item: item[1], reverse=True)[:20])


# -------------- AdaBoost ------------------

ada_selected_features = []
ada_selected_features_cnt = {}
ada_mse = []

for train_index, test_index in RSKF.split(X, Y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
    ada_boost = AdaBoostRegressor()

    # Linear search range for parameters
    ada_param_grid = {
        'n_estimators': [5, 10, 20, 50, 100, 200],
        'loss': ['linear', 'square', 'exponential'],
        'learning_rate':[0.01, 0.05, 0.1, 0.5, 1, 2]
    }

    # Perform linear search for survival time OS_time
    grid_search = GridSearchCV(ada_boost, ada_param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    # Model using the best parameters
    ada_boost = grid_search.best_estimator_

    ada_train_pred = ada_boost.predict(X_train)
    ada_train_mse = mean_squared_error(Y_train, ada_train_pred)

    ada_test_pred = ada_boost.predict(X_test)
    ada_test_mse = mean_squared_error(Y_test, ada_test_pred)
    ada_mse.append((ada_test_mse, ada_train_mse))

    # Feature selection using the model's `coef_` attribute to evaluate feature importance
    ada_feature_importances = pd.Series(ada_boost.feature_importances_, index=X.columns)
    ada_top_features = ada_feature_importances.abs().sort_values(ascending=False).head(20)

    ada_top20_features_name = list(ada_top_features.index)
    ada_selected_features.append(ada_top20_features_name)

ada_combined_data = list(zip(ada_mse, ada_selected_features))
ada_combined_data.sort(key=lambda x: x[0][0])

ada_top3_test_mse = [mse[0] for mse, _ in ada_combined_data[:3]]
ada_top3_train_mse = [mse[1] for mse, _ in ada_combined_data[:3]]

ada_top3_features_group = [features for _, features in ada_combined_data[:3]]
ada_top3_features_group = [feature for sublist in ada_top3_features_group for feature in sublist]


for feature_name in ada_top3_features_group:
    if feature_name in ada_selected_features_cnt.keys():
        ada_selected_features_cnt[feature_name] += 1
    else:
        ada_selected_features_cnt[feature_name] = 1

ada_sorted_top20_features = dict(sorted(ada_selected_features_cnt.items(), key=lambda item: item[1], reverse=True)[:20])

# -------------- Support Vector Machine ------------------

svm_selected_features = []
svm_selected_features_cnt = {}
svm_mse = []

for train_index, test_index in RSKF.split(X, Y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
    svm_regressor = SVR(kernel='linear')

    # Linear search range for parameters
    svm_param_grid = {
        'C': [0.01, 0.05, 0.1, 5, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    }

    grid_search = GridSearchCV(svm_regressor, svm_param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    # Model using the best parameters
    best_svm_regressor = grid_search.best_estimator_

    # Evaluate the performance of the best model
    svm_train_pred = best_svm_regressor.predict(X_train)
    svm_train_mse = mean_squared_error(Y_train, svm_train_pred)

    svm_test_pred = best_svm_regressor.predict(X_test)
    svm_test_mse = mean_squared_error(Y_test, svm_test_pred)
    svm_mse.append((svm_test_mse, svm_train_mse))

    rfe = RFE(estimator=best_svm_regressor, n_features_to_select=20)
    X_train_rfe = rfe.fit_transform(X_train, Y_train)

    svm_selected_features_indices = rfe.get_support(indices=True)
    svm_top20_features_name = list(X.columns[svm_selected_features_indices])
    svm_selected_features.append(svm_top20_features_name)

svm_combined_data = list(zip(svm_mse, svm_selected_features))
svm_combined_data.sort(key=lambda x: x[0][0])

svm_top3_test_mse = [mse[0] for mse, _ in svm_combined_data[:3]]
svm_top3_train_mse = [mse[1] for mse, _ in svm_combined_data[:3]]

svm_top3_features_group = [features for _, features in svm_combined_data[:3]]
svm_top3_features_group = [feature for sublist in svm_top3_features_group for feature in sublist]


for feature_name in svm_top3_features_group:
    if feature_name in svm_selected_features_cnt.keys():
        svm_selected_features_cnt[feature_name] += 1
    else:
        svm_selected_features_cnt[feature_name] = 1

svm_sorted_top20_features = dict(sorted(svm_selected_features_cnt.items(), key=lambda item: item[1], reverse=True)[:20])