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


data = pd.read_csv('filtered_data.csv')

X = data.drop(['patient_id', 'id_suffix', 'OS', 'OS_time'], axis=1)
Y = data['OS_time']

scaler = StandardScaler()
numerical_features = X.columns

X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define the cross-validation parameters
cv = StratifiedKFold(n_splits=5)

# --------- Random Forest -----------

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Linear search range for parameters
rf_param_grid = {
    'n_estimators': [10, 30, 50, 100, 200, 300],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2']
}

# Perform linear search for survival time OS_time
grid_search = GridSearchCV(rf, rf_param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)

# Model using the best parameters
best_svm_regressor = grid_search.best_estimator_

# Evaluate the performance of the best model
best_svm_regressor.predict(X_train)

# Print feature importance
importances = best_svm_regressor.feature_importances_
features = X.columns
importance_dict = dict(zip(features, importances))
rf_sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
rf_feature_name = [item[0] for item in rf_sorted_importance[:20]]

# -------------- Cox -----------------

cph = CoxPHFitter(penalizer=0.1)  # Penalizer for regularization to avoid overfitting; can be adjusted as needed

df = pd.read_csv('source_data_integrated_v2_drop_PGAM2.csv')  
tcga_numerical_features = df.drop(['patient_id', 'id_suffix', 'OS', 'OS_time'], axis=1).columns
tcga = df.drop(['patient_id', 'id_suffix'], axis=1)
tcga_scaled = scaler.fit_transform(df[tcga_numerical_features])
tcga_scaled = pd.DataFrame(tcga_scaled, columns=tcga_numerical_features)
tcga_scaled[['OS', 'OS_time']] = df[['OS', 'OS_time']]

# Split dataset into training and testing sets
X_train, X_test = train_test_split(tcga_scaled, test_size=0.2, random_state=32)

cph.fit(X_train, duration_col='OS_time', event_col='OS', show_progress=True)

# Output feature importance
tcga_feature_importance = abs(cph.summary.loc[:, ['coef']]).sort_values('coef', ascending=False)

tcga_feature_name_abs = list(tcga_feature_importance['coef'][:20].index)

# -------------- XGBOOST ------------------

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

# Feature importance
importance = bst.get_score(importance_type='weight')
xgb_feature_importance = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
xgb_feature_importance['Feature'] = X.columns
xgb_feature_importance = xgb_feature_importance.sort_values(by='Importance', ascending=False)

# -------------- AdaBoost ------------------

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

# Feature selection using the model's `coef_` attribute to evaluate feature importance
ada_feature_importances = pd.Series(ada_boost.feature_importances_, index=X.columns)
ada_top_features = ada_feature_importances.abs().sort_values(ascending=False).head(20)

ada_top_10_features_name = list(ada_top_features.index)
ada_top_10_features_importance = list(ada_top_features.values)

# -------------- Support Vector Machine ------------------

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
best_scores_ost = cross_val_score(best_svm_regressor, X_train, Y_train, cv=cv, scoring='neg_mean_squared_error')

rfe = RFE(estimator=best_svm_regressor, n_features_to_select=20)
X_train_rfe = rfe.fit_transform(X_train, Y_train)

svm_selected_features_indices = rfe.get_support(indices=True)
svm_top_10_features_name = list(X.columns[svm_selected_features_indices])