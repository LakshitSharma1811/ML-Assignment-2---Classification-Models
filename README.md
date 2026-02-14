Problem Statement:
The goal of this project is to implement and compare the performance of six different machine learning classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, Random Forest, and XGBoost. The comparison is based on standard evaluation metrics such as Accuracy, Precision, Recall, F1 Score, MCC, and AUC Score. The best performing model will be identified for deployment.

Dataset Description:

Dataset Name: Breast Cancer Wisconsin (Diagnostic) Data Set

Source: UCI Machine Learning Repository (via Scikit-Learn)

Description: The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

Features: 30 numeric features (e.g., radius, texture, perimeter, area, smoothness).

Instances: 569 instances.

Target: Binary classification (Malignant vs. Benign).

Models Used & Comparison Table:

ML Model Name	Accuracy	Precision	Recall	F1	MCC	AUC
Logistic Regression	0.9737	0.9737	0.9737	0.9736	0.9439	0.9974
Decision Tree	0.9474	0.9474	0.9474	0.9474	0.888	0.944
KNN	0.9474	0.9474	0.9474	0.9474	0.888	0.982
Naive Bayes	0.9649	0.9652	0.9649	0.9647	0.9253	0.9974
Random Forest	0.9649	0.9652	0.9649	0.9647	0.9253	0.9953
XGBoost	0.9561	0.9561	0.9561	0.956	0.9064	0.9908

Observations:

ML Model Name	Observation about model performance
Logistic Regression	Demonstrated very high accuracy and AUC, indicating the data is linearly separable.
Decision Tree	Slightly lower performance compared to ensemble methods, likely due to overfitting on training data.
KNN	Performed well, but inference time increases with dataset size. Sensitive to feature scaling (handled in code).
Naive Bayes	Good baseline model, fast training, but slightly less accurate than complex models.
Random Forest	Robust performance with high F1 score, handling non-linear relationships well.
XGBoost	Competitive with Random Forest; often achieved the highest AUC score in testing.




