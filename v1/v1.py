# First version using Linear Regression, TF-IDF and simple transformations/feature selection
# This will be used for: Final PC Progress Prize 1

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.sparse import hstack

# Load the training and test datasets
train_df = pd.read_csv('small.csv')
test_df = pd.read_csv('test.csv')

# Handle missing values
# Fill missing numerical values with the median
train_df['year_review'].fillna(train_df['year_review'].median(), inplace=True)
test_df['year_review'].fillna(train_df['year_review'].median(), inplace=True)

# Fill missing categorical values with 'Unknown'
for col in ['firm', 'job_title']:
    train_df[col].fillna('Unknown', inplace=True)
    test_df[col].fillna('Unknown', inplace=True)

# Fill missing text fields with an empty string
for col in ['headline', 'pros', 'cons']:
    train_df[col].fillna('', inplace=True)
    test_df[col].fillna('', inplace=True)

# Encode categorical variables using OneHotEncoder
categorical_cols = ['firm', 'job_title']
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)

# Fit the encoder on the training data
onehot_encoder.fit(train_df[categorical_cols])

# Transform both training and test data
X_train_categorical = onehot_encoder.transform(train_df[categorical_cols])
X_test_categorical = onehot_encoder.transform(test_df[categorical_cols])

# Vectorize text features using TF-IDF
text_cols = ['headline', 'pros', 'cons']
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to manage computational load

# Combine text columns into a single column for vectorization
train_text = train_df[text_cols].apply(lambda x: ' '.join(x), axis=1)
test_text = test_df[text_cols].apply(lambda x: ' '.join(x), axis=1)

# Fit the vectorizer on the training text data
tfidf_vectorizer.fit(train_text)

# Transform both training and test text data
X_train_text = tfidf_vectorizer.transform(train_text)
X_test_text = tfidf_vectorizer.transform(test_text)

# Prepare numerical features
numerical_cols = ['year_review']

X_train_numerical = train_df[numerical_cols]
X_test_numerical = test_df[numerical_cols]

# Combine all features
from scipy.sparse import csr_matrix

X_train_combined = hstack([csr_matrix(X_train_numerical), X_train_categorical, X_train_text])
X_test_combined = hstack([csr_matrix(X_test_numerical), X_test_categorical, X_test_text])

# Prepare the target variable
y_train = train_df['rating']

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_combined, y_train)

# Compute R² on the training data
y_train_pred = model.predict(X_train_combined)
r_squared = r2_score(y_train, y_train_pred)
print(f"R² on the training data: {r_squared}")

# Generate predictions for the test set
y_pred = model.predict(X_test_combined)

# Ensure predictions are within the valid range (1 to 5)
y_pred = np.clip(y_pred, 1, 5)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({'prediction': y_pred})

# Save the predictions to a CSV file
predictions_df.to_csv('test_predictions.csv', index=False)

print("Predictions have been generated and saved to 'test_predictions.csv'.")
