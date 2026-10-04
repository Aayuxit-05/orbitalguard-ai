import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("dataset.csv")

# Features used by the ML model
features = [
    "sampling", "duration", "len", "mean", "var", "std",
    "kurtosis", "skew", "n_peaks", "smooth10_n_peaks",
    "smooth20_n_peaks", "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var", "gaps_squared",
    "len_weighted", "var_div_duration", "var_div_len"
]

X = data[features]
y = data["anomaly"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt

# Get feature importance from the Random Forest model
importance = model.feature_importances_

# Create a table of features and their importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Plot the top 10 most important features
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features for Satellite Anomaly Detection")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# Visualize normal vs anomalous telemetry

import matplotlib.pyplot as plt

counts = data["anomaly"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Normal", "Anomaly"], counts.values)

plt.xlabel("Telemetry Status")
plt.ylabel("Number of Segments")
plt.title("Normal vs Anomalous Satellite Telemetry")

plt.tight_layout()
plt.show()
# Interactive prediction demo

print("\n--- Satellite Telemetry Prediction Demo ---")

row_number = int(input(
    f"Enter a segment row number (0 to {len(data)-1}): "
))

# Get the selected row
selected_row = data.iloc[[row_number]]

# Get its features
input_data = selected_row[features]

# Make prediction
prediction = model.predict(input_data)[0]

# Get anomaly probability
probability = model.predict_proba(input_data)[0][1]
probability_percent = probability * 100

if prediction == 1:
    print("\n🚨 ANOMALY DETECTED!")

    if probability_percent >= 80:
        risk = "🔴 HIGH"
    elif probability_percent >= 50:
        risk = "🟡 MEDIUM"
    else:
        risk = "🟢 LOW"

    print(f"Anomaly probability: {probability_percent:.1f}%")
    print(f"Risk Level: {risk}")

else:
    print("\n🟢 TELEMETRY IS NORMAL")
    print(f"Anomaly probability: {probability_percent:.1f}%")
    print("Risk Level: 🟢 LOW")

actual = selected_row["anomaly"].iloc[0]

# Show the actual label for comparison
actual = selected_row["anomaly"].iloc[0]

if actual == 1:
    print("Actual label: ANOMALY")
else:
    print("Actual label: NORMAL")
    # Model Prediction Performance

cm = confusion_matrix(y_test, y_pred)

correct_normal = cm[0, 0]
false_alarm = cm[0, 1]
missed_anomaly = cm[1, 0]
correct_anomaly = cm[1, 1]

labels = [
    "Correct Normal",
    "False Alarm",
    "Missed Anomaly",
    "Correct Anomaly"
]

values = [
    correct_normal,
    false_alarm,
    missed_anomaly,
    correct_anomaly
]

# Calm scientific colors
colors = [
    "#6B2737",
    "#8E4A5A",
    "#A66A76",
    "#B98B94"
]


plt.figure(figsize=(9, 6))

bars = plt.bar(labels, values, color=colors)

plt.title(
    "Satellite Anomaly Detection Performance",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Prediction Result")
plt.ylabel("Number of Telemetry Segments")

# Add values above bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 5,
        str(value),
        ha="center",
        fontsize=11
    )

plt.xticks(rotation=10)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
# Satellite Telemetry Summary

total_segments = len(data)
normal_segments = (data["anomaly"] == 0).sum()
anomaly_segments = (data["anomaly"] == 1).sum()

anomaly_percentage = (anomaly_segments / total_segments) * 100

print("\n--- Satellite Telemetry Summary ---")
print("Total segments:", total_segments)
print("Normal segments:", normal_segments)
print("Anomalous segments:", anomaly_segments)
print("Anomaly percentage:", round(anomaly_percentage, 2), "%")