# Predictive-Maintenance-using-Machine-Learning
A machine-learning project that predicts different types of industrial machine failures using operating conditions such as temperature, rotational speed, torque, tool wear, and product type.

## Problem Statement

Machine reliability is critical in industrial environments because unexpected equipment failures can interrupt production, affect product quality, increase maintenance requirements, and cause production delays.

My interest in this problem comes from my previous exposure to industrial environments through internships and professional experience at **Artistic Milliners, Engro, and Millennium Engineering**, where I worked around different manufacturing and processing operations. These experiences highlighted the importance of machine reliability and motivated me to explore how data and machine learning could support predictive maintenance.

## Project Objective

The objective of this project is to develop a classification model that predicts the **type of machine failure** based on its operating conditions.

The model predicts six possible outcomes:

* No Failure
* Heat Dissipation Failure
* Power Failure
* Overstrain Failure
* Tool Wear Failure
* Random Failures

## Approach

The project follows a simple machine-learning pipeline:

1. Explored and analyzed the machine-maintenance dataset.
2. Selected relevant machine operating parameters.
3. Encoded the categorical `Type` feature.
4. Encoded the failure types as numerical target labels.
5. Split the data into training and testing sets using an **80/20 stratified split**.
6. Trained a **Random Forest Classifier**.
7. Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Analyzed feature importance to understand which operating parameters influenced the model most.
9. Saved the trained model for deployment.
10. Developed a **Streamlit interface** for interactive predictions.

## Results

The Random Forest model achieved:

**Accuracy: 98.4%**

The most influential features according to the model were:

| Feature             | Importance |
| ------------------- | ---------: |
| Torque              |     31.35% |
| Rotational Speed    |     23.18% |
| Tool Wear           |     16.40% |
| Air Temperature     |     13.53% |
| Process Temperature |     12.85% |
| Product Type        |      2.69% |

Because the dataset is highly imbalanced, the model performs significantly better on the dominant **No Failure** class than on the rare failure classes. This highlights an important limitation of the dataset and shows why accuracy alone is not sufficient for evaluating predictive-maintenance models.

## Requirements

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## Future Improvements

Possible improvements include collecting more examples of rare failure types, experimenting with additional classification techniques, and tuning the model to improve detection of minority failure classes.

## Author
Amna Mujahid
