# AI & ML Internship - Task 4

## Classification with Logistic Regression



## 1. Objective



The objective of this task is to build a binary classification model using Logistic Regression.



The project covers:



- Binary classification

- Train-test splitting

- Feature standardization

- Logistic Regression

- Confusion Matrix

- Precision

- Recall

- ROC-AUC

- ROC Curve

- Sigmoid Function

- Classification threshold tuning



---



## 2. Dataset



### Breast Cancer Wisconsin Dataset



The Breast Cancer Wisconsin dataset provided by Scikit-learn was used for this task.



Dataset details:



- Samples: 569

- Features: 30

- Classes: 2

- Malignant samples: 212

- Benign samples: 357



The dataset was saved locally as:





data/breast\_cancer\_wisconsin.csv



## 3. Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- Matplotlib



## 4. Project Structure



aiml-task-4-logistic-regression/

│

├── data/

│   ├── breast\_cancer\_wisconsin.csv

│   └── threshold\_results.csv

│

├── visualizations/

│   ├── confusion\_matrix.png

│   ├── roc\_curve.png

│   ├── sigmoid\_curve.png

│   └── threshold\_analysis.png

│

├── logistic\_regression.py

├── requirements.txt

└── README.md



## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Breast Cancer Wisconsin dataset.

2. Separated features and target variable.

3. Split the dataset into training and testing sets.

4. Used 80% of the data for training and 20% for testing.

5. Used stratified splitting to preserve the class distribution.

6. Standardized the features using StandardScaler.



Train-Test Split

Training samples: 455

Testing samples: 114



## 6. Logistic Regression Model

A Logistic Regression classifier was trained using Scikit-learn.



model = LogisticRegression(

&#x20;   max\_iter=1000,

&#x20;   random\_state=42

)



The trained model was used to predict the class labels and class probabilities on the test dataset.



## 7. Model Evaluation

The model was evaluated using:

- Accuracy

- Precision

- Recall

- ROC-AUC

- Confusion Matrix



Results



| Metric | Score |

|---|---:|

| Accuracy | 98.25% |

| Precision | 98.61% |

| Recall | 98.61% |

| ROC-AUC | 99.54% |



## 8. Confusion Matrix

The confusion matrix obtained from the test set was:

[[41  1]

&#x20;[ 1 71]]



This means:

- 41 malignant samples were correctly classified.

- 71 benign samples were correctly classified.

- 1 malignant sample was classified as benign.

- 1 benign sample was classified as malignant.



Visualization:

visualizations/confusion\_matrix.png



## 9. ROC Curve and ROC-AUC

The ROC curve was generated using the predicted probabilities from the Logistic Regression model.



The model achieved:

ROC-AUC = 0.9954



A higher ROC-AUC indicates that the model has strong ability to distinguish between the two classes.



Visualization:

visualizations/roc\_curve.png



## 10. Sigmoid Function

Logistic Regression uses the sigmoid function to convert the model's output into a probability between 0 and 1.



The sigmoid curve was visualized in:

visualizations/sigmoid\_curve.png



The probability can then be converted into a class prediction using a classification threshold.



## 11. Classification Threshold Tuning

The default classification threshold of 0.50 was compared with several other thresholds.

| Threshold | Precision | Recall |

|---:|---:|---:|

| 0.30 | 0.9730 | 1.0000 |

| 0.40 | 0.9861 | 0.9861 |

| 0.50 | 0.9861 | 0.9861 |

| 0.60 | 0.9855 | 0.9444 |

| 0.70 | 0.9853 | 0.9306 |



Observation

At a threshold of 0.30, recall reached 100%, while precision decreased slightly.

At a threshold of 0.50, precision and recall were both 98.61%.

At higher thresholds, precision remained high while recall decreased.

This demonstrates how changing the classification threshold affects the trade-off between precision and recall.



Visualization:

visualizations/threshold\_analysis.png



## 12. Key Concepts Learned

Binary Classification-

Binary classification predicts one of two possible classes.



Logistic Regression-

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to a class.



Precision-

Precision measures how many of the samples predicted as positive were actually positive.



Recall-

Recall measures how many of the actual positive samples were correctly identified.



Confusion Matrix-

A confusion matrix summarizes correct and incorrect predictions across the actual and predicted classes.



ROC-AUC-

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.



Classification Threshold-

The threshold determines the probability above which a sample is assigned to the positive class.



## 13. Files Generated

Python Script

logistic_regression.py



Contains the complete data loading, preprocessing, model training, evaluation, threshold tuning, and visualization code.

Dataset

data/breast_cancer_wisconsin.csv



Threshold Results

data/threshold\_results.csv



Visualizations

visualizations/confusion\_matrix.png

visualizations/roc\_curve.png

visualizations/sigmoid\_curve.png

visualizations/threshold\_analysis.png



## 14. How to Run the Project

Clone the repository and navigate to the project directory.

Install the required libraries:

pip install -r requirements.txt



Run the Python script:

python logistic\_regression.py



The dataset, threshold results, and visualizations will be generated automatically.



## 15. Conclusion

A Logistic Regression binary classification model was successfully developed using the Breast Cancer Wisconsin dataset.

The model achieved 98.25% accuracy and a ROC-AUC score of 99.54% on the test dataset.

The project also demonstrated how classification thresholds influence precision and recall, along with visualization of the confusion matrix, ROC curve, sigmoid function, and threshold analysis.



Author

Tulsi R. Dounekar





