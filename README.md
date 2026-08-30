# Predictive-Maintenance-using-Machine-Learning

A machine-learning project that predicts different types of industrial machine failures using operating conditions such as temperature, rotational speed, torque, tool wear, and product type.

## Live Demo

🔗 **https://predictive-maintenance-using-machine-learning-kxfgllc2g4vzuky3.streamlit.app/**

The Streamlit application allows users to enter machine operating conditions and receive a predicted failure type along with prediction probabilities.

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

The project follows a machine-learning pipeline from data exploration to model deployment:

1. Explored and analyzed the machine-maintenance dataset.
2. Examined the distribution of failure types and identified significant **class imbalance** in the dataset.
3. Selected relevant machine operating parameters:
   * Product Type
   * Air Temperature
   * Process Temperature
   * Rotational Speed
   * Torque
   * Tool Wear
4. Used **PyCaret** to compare multiple classification algorithms and identify the best-performing model.
5. Encoded the categorical `Type` feature using **Ordinal Encoding**.
6. Encoded the failure types using **Label Encoding**.
7. Split the data into training and testing sets using an **80/20 stratified split**.
8. Based on the PyCaret model comparison, **Random Forest Classifier** was selected for further implementation.
9. Trained the Random Forest model using **Scikit-learn**.
10. Evaluated the model using **accuracy, precision, recall, F1-score, and a confusion matrix**.
11. Analyzed feature importance to understand which machine parameters contributed most to the model's predictions.
12. Saved the trained model and preprocessing encoders using **Joblib**.
13. Developed a **Streamlit web application** for interactive machine-failure predictions.

## Results

The Random Forest model achieved:

**Accuracy: 98.4%**

The most influential features according to the trained Random Forest model were:

| Feature             | Importance |
| ------------------- | ---------: |
| Torque              |     31.35% |
| Rotational Speed    |     23.18% |
| Tool Wear           |     16.40% |
| Air Temperature     |     13.53% |
| Process Temperature |     12.85% |
| Product Type        |      2.69% |

**Torque** was the most influential feature, followed by rotational speed and tool wear.

Because the dataset is highly imbalanced, the model performs significantly better on the dominant **No Failure** class than on the rare failure classes. The rare **Random Failures** and **Tool Wear Failures** are particularly difficult for the model to detect because they have very few examples in the dataset.

This highlights an important limitation of the project and demonstrates why **accuracy alone is not sufficient** for evaluating predictive-maintenance models.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyCaret
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## Future Improvements

Possible improvements include:

* Collecting more examples of rare failure types.
* Applying class-balancing techniques.
* Experimenting with additional machine-learning algorithms.
* Improving recall for minority failure classes.
* Further tuning the Random Forest model.
* Incorporating additional machine sensor data for more accurate failure detection.

## Author

Amna Mujahid
