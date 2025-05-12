# DIABETES-PREDICTION-WITH-PCA
INTRODUCTION
This project focuses on building a predictive system for the early identification of diabetes
using machine learning methods, supported by Principal Component Analysis (PCA) for
feature reduction. As diabetes continues to pose a major global health concern, the importance
of early diagnosis cannot be overstated. Through this work, we aim to develop reliable
classification models that assess an individual's likelihood of having diabetes based on a range
of clinical and demographic factors.
The study utilises the "diabetes_data.csv" dataset, which contains structured medical
information for 1000 individuals. Machine learning models such as Logistic Regression,
Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbours (KNN) were
trained and evaluated. A deep learning Neural Network was also implemented to capture
complex relationships within the data. PCA was integrated into the pipeline to enhance model
performance by minimising dimensionality while retaining key information.

**Project Aim**
The primary goal of this project is to create an accurate and efficient diabetes prediction
framework by leveraging machine learning and deep learning techniques. Key objectives
include:
• Utilising Principal Component Analysis (PCA) to streamline input features while
maintaining essential variance.
• Developing and comparing multiple predictive models to accurately classify
individuals based on health-related attributes.
• Evaluating models based on critical metrics such as accuracy, precision, recall, F1
score, and ROC-AUC to select the most effective approach.
• Demonstrating how machine learning can contribute to early detection strategies,
improve patient management, and support public health initiatives

**DATASET DESCRIPTION**
Source: The dataset is sourced from Kaggle – Diabetes Prediction Dataset, specifically from
the repository titled "100000 Diabetes Clinical Dataset". It contains synthetic medical data
points for the binary classification of diabetes presence. A subset of this dataset, comprising
1000 patient records, was utilised to develop and test predictive models for early-stage diabetes
detection.
The dataset includes a range of demographic, lifestyle, and clinical variables that are commonly
associated with diabetes risk. Each row in the dataset corresponds to a single patient and
contains the following features:
 Number of Records: 1000 patient records.
 Columns (Features):
1. gender – Gender of the individual (Male, Female, Other)
2. age – Age of the person in years
3. hypertension – Binary indicator (0 = No, 1 = Yes)
4. heart disease – Binary indicator (0 = No, 1 = Yes)
5. smoking history – Categorical smoking status (never, current, former, not current,
ever, No Info)
6. Bmi – Body Mass Index
7. HbA1c_level – Hemoglobin A1c percentage
8. blood_glucose_level – Glucose level (in mg/dL)
9. diabetes – Target label (0 = non-diabetic, 1 = Diabetic)
The dataset provides a balanced set of features that are essential for building and evaluating
machine learning models capable of predicting diabetes effectively. Its synthetic nature ensures
privacy while maintaining realistic distributions like actual clinical data

**MACHINE LEARNING MODELLING ANALYSIS**
In this project, several machine learning and deep learning techniques were applied to predict
the presence of diabetes based on clinical features. A summary of each method, along with its
role in the analysis, is detailed below.
Data Preparation Techniques:
Train-Test Split:
The dataset was divided into training and testing sets using 80:20 split ratio for train and test
datasets was divided, where 80 is the training set and 20 is the testing set. This is done using
Scikit-learn’s train_test_split() function. To ensure reproducibility, a fixed random_state is
used. The training set used to train the models and the testing set measured how well they
generalized to new data.
Machine Learning Algorithms:
**1. Logistic Regression:** Logistic Regression is used as a classifier to predict diabetes
outcomes. This simple, fast, and effective binary classification algorithm models the
probability of a sample belonging to a specific class, such as diabetic or non-diabetic.
In this project, logistic regression helped in providing a Straightforward and Interpretable
model, Serves as a benchmark to compare the performance of more classifiers like Random
Forest and SVM. Works well with linearly separable data and scaled features and it is
implemented using Scikit-learn's LogisticRegression(). After applying PCA and feature
scaling it has been trained on the pre-processed dataset.
**2. Random Forest Classifier:** Random Forest is an ensemble learning method used to
classify whether a patient is diabetic based on medical features like glucose level, BMI,
and insulin. Since our dataset contains potentially correlated and noisy features, Random
Forest helps by handling feature interactions, Less sensitivity to outliers and noise, Higher
accuracy, and Maintaining Performance. Also, Random Forest tells us which
features(symptoms) were most important.
**3. Support Vector Machine (SVM):** SVM is used in Diabetes prediction and designing and
training the model, which helps in classifying diabetes cases by finding the optimal hyperplane
that separates the two classes (for Example, it can classify diabetic and non-diabetic).
Especially after PCA, SVM works well with high-dimensional data. When classes are not
easily separable, SVM will be more effective in datasets. In our project, SVM helped in
improving prediction accuracy by focusing on data points most critical to the classification
(Support Vectors)
**3. Decision Tree Regressor:** As a part of the analysis step, Decision Tree Regressor is used
to predict a continuous probability like output rather than a strict class label, It helped us to
estimate how keenly a patient’s features align with diabetic tendencies in a more granular
way than binary classification. By fitting the regressor on the reduced feature after applying
PCA, it provided Interpretable Outputs and Non-linear decision boundaries. Decision tree
regressor supported deeper exploration of the data’s structure.  
**4. K-Nearest Neighbours (KNN):** KNN was used in our project to classify patients based on
similarity to others in the dataset. Data is well-suited for a distance-based method like KNN
after applying PCA and Feature scaling. KNN helps in identifying diabetic cases by
comparing each patient to the most similar ones. KNN offered competitive accuracy and
worked well with reducing feature space. Compare KNN's simpler distance-based strategy
to Logistic Regression, Random Forest, and SVM's more complex models.
**5. Deep Learning Model-Neural Network:** Neural Network was developed to capture
complex, non-linear patterns present in the diabetes dataset and to achieve better prediction
accuracy. The model achieved an accuracy of 86.92%, demonstrating strong performance
in classifying diabetic and non-diabetic cases. Techniques like dropout and early stopping
were incorporated during training to prevent overfitting and enhance the model’s reliability.
These methods helped the Neural Network maintain consistent precision and recall of
around 87% across both classes. Overall, the Neural Network improved the model's ability
to recognise subtle patterns that traditional machine learning algorithms might overlook.
Sequential Model: Layers are stacked one after the other
First Dense Layer: 16 neurons and ReLU activation, which is good for hidden layers as it helps in
learning complex patterns.
Dropout(0.2) = to prevent overfitting 20% of the neurons are randomly turned off
during training
Second Dense layer: works with 8 neurons and ReLU
For more regularisation, we used Dropout again.
Final Dense layer: 1 neuron, sigmoid activation (because diabetes is a binary
classification — diabetic or not).
Loss function: binary_crossentropy (default for binary classification).
Optimiser: Adam (adaptive optimiser — great for deep learning).

**MODEL EVALUATION**
In this project, model evaluation was carried out using multiple standard metrics to assess the
performance of different algorithms on unseen data. The primary metrics considered were
Accuracy, Precision, Recall, and F1-score, derived from the Confusion Matrix.
Principal Component Analysis(PCA):
PCA was utilised for dimensionality reduction, aiming to reduce the number of input features
while retaining most of the original data's variance.
(a) Dimensionality Reduction – To reduce the number of input features while preserving
most of the variance in the data, Lower-dimensional data is often easier and faster to process,
especially with algorithms like KNN and SVM.
(b) Improving Model Performance – from Overfitting and Curse of Dimensionality.
(c) Visualization – PCA reduces data to 2 or 3 principal components, making it easier to
visualize clusters or separation between classes and understand data structure before applying
classification algorithms.
This strategy will help interpret model behaviour or explain results to non-technical
audiences. Classic Machine Learning Algorithm.
Evaluation Metrics:
The following metrics were used to evaluate the performance of each model:
• Accuracy measures the proportion of correctly predicted instances (both positive and
negative) over the total number of instances. It gives an overall effectiveness of the
model, but can be misleading if the dataset is imbalanced.
Accuracy = 𝑻𝑷+𝑻𝑵
𝑻𝑷+𝑻𝑵+𝑭𝑷+𝑭𝑵
• Precision indicates the proportion of correctly predicted positive observations to the
total predicted positive observations.
Precision = 𝑻𝑷
𝑻𝑷+𝑭𝑷
• Recall measures the proportion of actual positives correctly identified by the model.
Recall = 𝑻𝑷
𝑻𝑷+𝑭𝑵
• F1-Score calculates the harmonic mean of precision and recall, balancing both
concerns.
F1-Score = 𝟐𝑻𝑷
𝟐𝑻𝑷+𝑭𝑷+𝑭𝑵
• ROC-AUC Score:
The Receiver Operating Characteristic - Area Under Curve (ROC-AUC) measures the
ability of the classifier to distinguish between classes. A higher AUC value indicates better model performance across various threshold settings. An AUC score of 1.0
represents a perfect model, while 0.5 suggests no discrimination ability.
After training the machine learning models on the diabetes dataset, a comprehensive
evaluation was carried out to assess their effectiveness and reliability. The evaluation process
employed a set of standard metrics and validation techniques to ensure a rigorous comparison
across different models.

**Confusion Matrix:**
Visualization plays a critical role in our project, and we incorporated matplotlib-based
confusion matrix plots for each model to better interpret and compare their performance.
These visualizations allow us to observe misclassification patterns across different classifiers.
To gain deeper insights into each model’s behaviour, confusion matrices were generated for
every classifier trained, including Random Forest, SVM, KNN, and Neural Network.
A confusion matrix outlines the following counts:
• True Positives (TP): Correctly identified diabetic cases.
• True Negatives (TN): Correctly identified non-diabetic cases.
• False Positives (FP): Non-diabetic individuals incorrectly classified as diabetic (Type
I error).
• False Negatives (FN): Diabetic individuals incorrectly classified as non-diabetic
(Type II error).
Analysing the confusion matrix provides detailed information on the types of errors
committed by each model, thereby guiding improvements and helping balance trade-offs
between sensitivity (recall) and specificity.
To visually compare the model performances, we used a custom Python script with the
following key features:
• Subplots were created in a 2×2 grid layout for neat comparison.
• All models were systematically defined and evaluated.
• A loop was employed to plot the confusion matrices for each classifier.
• The visual output facilitated quick identification of the best-performing model based
on minimal misclassifications.
This structured visualization approach made it significantly easier to evaluate and select the
most effective model for diabetes prediction.
