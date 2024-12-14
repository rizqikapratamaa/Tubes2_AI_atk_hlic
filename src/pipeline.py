# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('your_dataset.csv')

# Data preprocessing
# Assuming 'target' is the column to predict and the rest are features
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the validation data
y_val_pred = pipeline.predict(X_val)

# Evaluate the model on the validation data
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy: {val_accuracy}")

# Load test dataset
test_data = pd.read_csv('your_test_dataset.csv')

# Make predictions on the test data using the same pipeline
test_predictions = pipeline.predict(test_data)

# Create a submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Assuming 'Id' is the identifier column
    'Prediction': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)