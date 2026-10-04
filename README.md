# Satellite Telemetry Anomaly Detection Using Machine Learning

## About the Project

This project uses Machine Learning to identify unusual patterns in satellite telemetry data.

Satellite telemetry contains measurements that help describe the behavior and condition of a satellite. The model analyzes these measurements and classifies each telemetry segment as either **Normal** or **Anomalous**.

The project uses a **Random Forest Classifier** built with Python and scikit-learn.

## Dataset

The project uses the publicly available **OPSSAT-AD dataset**, which is based on telemetry data from the ESA OPS-SAT mission.

The dataset contains telemetry segments with labels indicating whether they are normal or anomalous.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Random Forest Machine Learning

## How the Model Works

The project follows these main steps:

1. Load the satellite telemetry dataset.
2. Select relevant telemetry features.
3. Split the data into training and testing sets.
4. Train a Random Forest classifier.
5. Predict whether telemetry segments are normal or anomalous.
6. Evaluate the model using accuracy, classification report, and confusion matrix.
7. Analyze which features are most important to the model.
8. Test individual telemetry segments using an interactive prediction demo.

## Model Performance

The Random Forest model achieved approximately **95% accuracy** on the test data.

The confusion matrix was:

```text
[[334, 4],
 [17, 70]]
```

This shows the number of correctly and incorrectly classified normal and anomalous telemetry segments.

## Feature Importance

The project also analyzes which telemetry features contribute most to the model's decisions.

Some of the important features include:

* `n_peaks`
* `smooth10_n_peaks`
* `diff_peaks`
* `diff2_peaks`
* `len`
* `gaps_squared`
* `diff2_var`
* `kurtosis`
* `skew`
* `duration`

A feature-importance graph is generated using Matplotlib.

## Interactive Prediction

The project includes an interactive demo where the user can enter a telemetry row number.

The model then predicts:

* 🟢 **Telemetry is Normal**
* 🚨 **Anomaly Detected**

The actual dataset label is also displayed so the prediction can be compared with the real label.

## Limitations

* The model is trained on a specific public dataset and may not perform the same way on every satellite.
* An anomaly prediction does not identify the exact cause of a satellite problem.
* More real-time and diverse satellite telemetry data would be required for a production-level system.

## Future Scope

This project can be expanded into a more advanced satellite monitoring system by:

* Adding real-time telemetry monitoring.
* Using larger and more diverse datasets.
* Comparing different Machine Learning models.
* Adding automatic alerts for detected anomalies.
* Building a dashboard for visualizing satellite telemetry.
* Exploring Deep Learning methods for more complex telemetry patterns.

## Project Status

**Completed beginner Machine Learning prototype.**

This project was developed as a first-year B.Tech Artificial Intelligence and Machine Learning project to understand how Machine Learning can be applied to real-world satellite telemetry data.
