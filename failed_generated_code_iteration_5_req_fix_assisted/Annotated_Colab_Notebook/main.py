# Import necessary libraries
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the ARFF file and convert to DataFrame
file_path = "dataset_adult.arff"
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Decode binary columns to strings (necessary for proper handling of categorical data)
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

# Display a preview of the dataset
print("Dataset sample:")
print(df.head())

# Step 2: Preprocess the dataset
# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop('income_>50K', axis=1)
y = df_encoded['income_>50K']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate model performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



