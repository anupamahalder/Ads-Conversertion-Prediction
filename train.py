# -----------------------------
# 0. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv('Social_Network_Ads.csv')

# Drop unnecessary column
if 'User ID' in df.columns:
    df.drop('User ID', axis=1, inplace=True)

# Features & Target
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# -----------------------------
# 3. Feature Engineering
# -----------------------------
def create_features(X):
    X = X.copy()
    X['Age_Group'] = X['Age'] // 10
    X['Salary_Group'] = X['EstimatedSalary'] // 1000
    X['Age_Salary_Interaction'] = X['Age'] * X['EstimatedSalary']
    return X

feature_engineer = FunctionTransformer(create_features, validate=False)

num_cols = [
    'Age',
    'EstimatedSalary',
    'Age_Group',
    'Salary_Group',
    'Age_Salary_Interaction'
]

cat_cols = ['Gender']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

# -----------------------------
# 4. KNN Pipeline
# -----------------------------
knn_pipeline = Pipeline([
    ('feature_engineering', feature_engineer),
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier())
])

# -----------------------------
# 5. Baseline Training
# -----------------------------
knn_pipeline.fit(X_train, y_train)

y_pred = knn_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"KNN Test Accuracy (Baseline): {test_acc:.4f}")
print("-" * 50)

# -----------------------------
# 6. Cross-Validation
# -----------------------------
cv_scores = cross_val_score(
    knn_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("KNN Cross-Validation Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Std Dev CV Accuracy: {cv_scores.std():.4f}")
print("-" * 50)

# -----------------------------
# 7. Hyperparameter Tuning
# -----------------------------
param_grid = {
    'model__n_neighbors': [5, 7, 9],
    'model__weights': ['uniform', 'distance']
}

grid = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print(f"Best CV Score: {grid.best_score_:.4f}")

# -----------------------------
# 8. Final Evaluation
# -----------------------------
final_pred = best_model.predict(X_test)

print("\nFinal Test Accuracy:", accuracy_score(y_test, final_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

# -----------------------------
# 9. Save Final Model
# -----------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nFinal KNN model saved as model.pkl successfully!")