# Heart Disease Prediction Project (Refined with BMI, Outlier Removal, Pulse Pressure)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv('cardio_train.csv', sep=';')

# Step 3: Preprocessing
# Drop 'id' column
df = df.drop('id', axis=1)

# Convert 'age' from days to years
df['age'] = (df['age'] / 365).round(1)

# ✅ Add BMI Feature
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# ✅ Remove Outliers
df = df[(df['height'] >= 120) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]

# ✅ Add Pulse Pressure Feature
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Features and Target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions for Test Data
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# Step 9: Take User Input and Predict
print("\nEnter the following details to predict heart disease risk:")

user_data = {
    'age': float(input("Age (in years): ")),
    'gender': int(input("Gender (1: Male, 2: Female): ")),
    'height': float(input("Height (in cm): ")),
    'weight': float(input("Weight (in kg): ")),
    'ap_hi': int(input("Systolic Blood Pressure (ap_hi): ")),
    'ap_lo': int(input("Diastolic Blood Pressure (ap_lo): ")),
    'cholesterol': int(input("Cholesterol (1: Normal, 2: Above Normal, 3: Well Above Normal): ")),
    'gluc': int(input("Glucose (1: Normal, 2: Above Normal, 3: Well Above Normal): ")),
    'smoke': int(input("Do you smoke? (0: No, 1: Yes): ")),
    'alco': int(input("Do you consume alcohol? (0: No, 1: Yes): ")),
    'active': int(input("Are you physically active? (0: No, 1: Yes): "))
}

# ✅ Calculate BMI and Pulse Pressure for user input
user_data['bmi'] = user_data['weight'] / ((user_data['height'] / 100) ** 2)
user_data['pulse_pressure'] = user_data['ap_hi'] - user_data['ap_lo']

# Convert to DataFrame
user_df = pd.DataFrame([user_data])

# Scale the input
user_scaled = scaler.transform(user_df)

# Make prediction
prediction = model.predict(user_scaled)

# Show result
if prediction[0] == 1:
    print("\n⚠️  Warning: You have a higher risk of heart disease.")
else:
    print("\n✅ Good news: Low risk of heart disease.")

input("\nPress Enter to exit...")
