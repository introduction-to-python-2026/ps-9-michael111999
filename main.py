import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('/content/parkinsons.csv')
display(df.head())
input_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

print("Input features (X) head:")
display(X.head())
print("\nOutput feature (y) head:")
display(y.head())

 

print("Scaled Input features (X_scaled) head:")
display(X_scaled.head())



print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Instantiate RandomForestClassifier with a random state for reproducibility
model = RandomForestClassifier(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = model.score(X_val, y_val)

print(f"Model Accuracy on Validation Set: {accuracy:.4f}")

if accuracy >= 0.8:
    print("Accuracy requirement met!")
else:
    print("Accuracy is below 0.8. Consider further model tuning or feature engineering.")
joblib.dump(model, 'my_model.joblib')
