import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
# DATASET
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# İLK BAKIŞ
print(df.head())

# SHAPE
print("\nShape:")
print(df.shape)

# INFO
print("\nInfo:")
print(df.info())

# MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# TARGET DAĞILIMI
print("\nChurn Distribution:")
#print(df['Churn'].value_counts()) #imbalance var ama çok kötü değil

df.info()
print(df['TotalCharges'].head(20))
print(df['TotalCharges'].unique()[:20])
print(df[df['TotalCharges'] == ' '])

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

print(df['TotalCharges'].isnull().sum())
df.dropna(inplace=True)
print(df.info())

print(df['TotalCharges'].dtype)
print(df[['MonthlyCharges', 'TotalCharges']].dtypes)

df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({
    'Yes': 1,
    'No': 0
})
print('#######' + str(df['Churn'].value_counts()))
print(df.info())

print(df.select_dtypes(include='object').columns)


df = pd.get_dummies(
    df,
    drop_first=True
)

print(df.head())
print(df.info())

X = df.drop('Churn', axis=1)
y = df['Churn']  ## churn target değişkeni


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, predictions))

auc = roc_auc_score(y_test, probabilities)
print("Random Forest ROC-AUC:", auc)

print('#########################')

# FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
# SORT
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)
print(importance_df.head(10))

print('*************************')



xgb_model = XGBClassifier(
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_probabilities = xgb_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, xgb_predictions))

xgb_auc = roc_auc_score(y_test, xgb_probabilities)

print("XGBoost ROC-AUC:", xgb_auc) 

print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&') 


# PARAMETRELER
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# MODEL
xgb_model = XGBClassifier(
    random_state=42
)

# RANDOM SEARCH
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

# TRAIN
random_search.fit(X_train, y_train)

# BEST MODEL
best_model = random_search.best_estimator_

# PREDICT
best_predictions = best_model.predict(X_test)

best_probabilities = best_model.predict_proba(X_test)[:,1]

# REPORT
print(classification_report(y_test, best_predictions))

# ROC-AUC
best_auc = roc_auc_score(
    y_test,
    best_probabilities
)

print("Best ROC-AUC:", best_auc)

# BEST PARAMS
print(random_search.best_params_)


print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# FINAL MODEL FEATURE IMPORTANCE
final_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})

final_importance_df = final_importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFinal Model Feature Importance:")
print(final_importance_df.head(10))

joblib.dump(best_model, "../back-end/churn_model.pkl")
joblib.dump(X.columns.tolist(), "../back-end/model_columns.pkl")
