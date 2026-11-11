import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 10000

# Feature synthesis
income = np.random.normal(55000, 18000, n_samples).clip(15000, 150000)
debt = np.random.normal(18000, 10000, n_samples).clip(1000, 90000)
utilization = (debt / (income * 0.6)).clip(0.05, 1.2)
past_delinquencies = np.random.poisson(0.4, n_samples)

latent_risk = (
    0.35 * (debt / income)
    + 0.45 * utilization
    + 0.5 * past_delinquencies
    - 0.00002 * income
)
default_prob = 1 / (1 + np.exp(-(latent_risk - 1.2)))
target = (np.random.rand(n_samples) < default_prob).astype(int)

# Pure numerical feature array: [income, total_debt, dti, credit_utilization, past_delinquencies]
dti = debt / income
X = np.column_stack([income, debt, dti, utilization, past_delinquencies])
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Save artifact
joblib.dump(model, 'credit_model.joblib')
print("Model artifact exported to credit_model.joblib successfully.")
