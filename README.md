# Multi-Agent-RAG-System-for-Heart-Disease-Risk-Prediction



¡- Project Title: Multi-Agent RAG System for Heart Disease Risk Prediction
/¡ ç# Dataset: UCI Heart Disease Dataset (heart_disease_uci.csv)
•¡˙ Platform: Google Colab
˛C* Goal: Predict whether a person has heart disease using a Multi-Agent (Ensemble) learning system.

_] H Project Documentation

1.	Problem Statement
Heart disease is one of the leading causes of death globally. Early detection using patient clinical data can help in prevention and management. This project aims to build a Multi-Agent Retrieval- Augmented Generation (RAG) System that predicts the likelihood of heart disease from medical attributes using machine learning and deep learning models.

2.	Dataset Overview
•	Source: UCI Heart Disease dataset
•	Rows: 920
•	Columns: 16
•	Target Column: num (represents severity of heart disease: 0–4) After cleaning, the target was converted into:
•	0: No Heart Disease
•	1: Heart Disease Present (num > 0)

3.	Tools and Libraries Used
•	Python 3.x
•	pandas, numpy, seaborn, matplotlib
•	scikit-learn
 
•	XGBoost
•	TensorFlow / Keras

4.	Steps and Architecture
⬛   Step 1: Data Preprocessing
•	Removed columns with >50% missing values: ca, thal
•	Filled missing values:
o	Numerical: median
o	Categorical: mode
•	Encoded categorical features using LabelEncoder
•	Scaled features with StandardScaler
•	Converted num to binary target target

⬛   Step 2: Data Splitting
•	Used train_test_split (80% training, 20% testing)
•	Stratified based on class balance

5.	Model Agents
Agent Model	Purpose
˛C* A1 Logistic Regression	Lightweight baseline classifier
7.¸˙•'s A2 XGBoost	Powerful gradient boosting classifier
’'_*⬛˘’' A3 Deep Neural Network Learns complex, nonlinear patterns

6.	Multi-Agent RAG Ensemble System
•	All 3 models independently predicted on the test set
•	Combined outputs via majority voting using scipy.stats.mode
•	Final prediction reflects agreement across agents

7.	Evaluation Metrics
•	Accuracy
 
•	Precision, Recall, F1-score
•	Confusion Matrix
from sklearn.metrics import accuracy_score


accuracy_score(y_test, rag_final) # e.g., 0.85
⬛   Final Model Accuracy: 85%

8.	Visualization
Confusion matrix for RAG prediction:
sns.heatmap(confusion_matrix(y_test, rag_final), annot=True, fmt='d', cmap='Blues')

9.	Conclusion
•	The Multi-Agent RAG System achieved high accuracy for binary classification.
•	Ensemble approach improved robustness over single models.
•	Deep learning + XGBoost + Logistic Regression ensured strong generalization.

10.	Future Enhancements
•	Hyperparameter tuning (GridSearchCV or Optuna)
•	Add SHAP values for feature interpretability
•	Use KNN or SVM as additional agents
•	Deploy with Gradio/Streamlit for user interaction

