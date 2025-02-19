---
layout: page
title: Proposal
permalink: /proposal
---

## Proposal Sections & Checklist

### 1. Introduction/Background
Provide an introduction of your topic and literature review of related work. Briefly explain your dataset and its features, and provide a link to the dataset if possible.

- **Literature Review**  
- **Dataset Description**  
- **Dataset Link**

### 2. Problem Definition
Identify a problem and motivate the need for a solution.

- **Problem**  
- **Motivation**  

### 3. Methods
Present proposed solutions including specific data processing methods and machine learning algorithms, and elaborate on why you think each will be effective. It is recommended to identify specific functions/classes in existing packages and libraries (i.e. scikit-learn) rather than coding the algorithms from scratch.

- **3+ Data Preprocessing Methods**  
- **3+ ML Algorithms/Models**  
- Supervised **or** Unsupervised Learning Methods Identified (Supervised is recommended)

## **Data Preprocessing Methods (5–6)**

1. **Handling Missing Values**
   - **Method**: `SimpleImputer` (from `sklearn.impute`)
   - **What It Does**: Replaces missing values with a constant, mean, median, or mode.
   - **Why Effective**: Preserves as much data as possible, preventing large-scale data loss or bias that might arise from simply removing rows/columns.
   - **Example**:
     ```python
     from sklearn.impute import SimpleImputer
     imputer = SimpleImputer(strategy='mean')
     X_imputed = imputer.fit_transform(X)
     ```

2. **Outlier Detection and Removal**
   - **Method**: Manual thresholding or using algorithms like `IsolationForest` (from `sklearn.ensemble`) or `LocalOutlierFactor` (from `sklearn.neighbors`)
   - **What It Does**: Identifies points that deviate significantly from the rest (potential anomalies), which can hamper model training.
   - **Why Effective**: Reduces the negative impact of extreme values on model parameters and improves generalization.
   - **Example** (`IsolationForest`):
     ```python
     from sklearn.ensemble import IsolationForest
     iso_forest = IsolationForest(random_state=42)
     outlier_labels = iso_forest.fit_predict(X)
     # Retain only inlier data
     X_inliers = X[outlier_labels == 1]
     ```

3. **Feature Scaling**
   - **Method**: `StandardScaler` or `MinMaxScaler` (from `sklearn.preprocessing`)
   - **What It Does**: Normalizes numerical features either to have zero mean and unit variance (StandardScaler) or to a fixed range like [0, 1] (MinMaxScaler).
   - **Why Effective**: Helps many algorithms (like SVM, KNN, neural networks) converge faster and perform better since they assume data is on a similar scale.
   - **Example**:
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     ```

4. **Categorical Encoding**
   - **Method**: `OneHotEncoder` or `OrdinalEncoder` (from `sklearn.preprocessing`)
   - **What It Does**: Converts categorical string variables into numerical arrays. 
   - **Why Effective**: Most ML algorithms require numeric inputs. One-hot encoding preserves “no ordering” assumptions among categories.
   - **Example**:
     ```python
     from sklearn.preprocessing import OneHotEncoder
     encoder = OneHotEncoder(handle_unknown='ignore')
     X_encoded = encoder.fit_transform(X_categorical)
     ```

5. **Feature Selection**
   - **Method**: `SelectKBest` or `RFE` (Recursive Feature Elimination) (from `sklearn.feature_selection`)
   - **What It Does**: Selects a subset of the most relevant features based on statistical tests (SelectKBest) or by iteratively removing the weakest features (RFE).
   - **Why Effective**: Reduces dimensionality, improves model performance, and shortens training time by removing noise or redundant features.
   - **Example** (`SelectKBest`):
     ```python
     from sklearn.feature_selection import SelectKBest, f_classif
     selector = SelectKBest(score_func=f_classif, k=10)
     X_selected = selector.fit_transform(X, y)
     ```

6. **Dimensionality Reduction**
   - **Method**: `PCA` (from `sklearn.decomposition`)
   - **What It Does**: Principal Component Analysis transforms features into a lower-dimensional space by capturing directions of maximum variance.
   - **Why Effective**: Improves computational efficiency, can help reduce overfitting, and might lead to better model performance in high-dimensional data.
   - **Example**:
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=2)
     X_pca = pca.fit_transform(X_scaled)
     ```

---

## **Machine Learning Algorithms (5–6)**

### **Supervised Methods**

1. **Logistic Regression**
   - **Class**: `LogisticRegression` (from `sklearn.linear_model`)
   - **Use Cases**: Binary or multi-class classification (e.g., spam detection, disease prediction).
   - **Why Effective**: 
     - Interpretable coefficients  
     - Typically a strong baseline for many classification tasks  
     - Fast to train and often robust with smaller or moderately sized datasets
   - **Code Example**:
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

2. **Random Forest**
   - **Class**: `RandomForestClassifier` or `RandomForestRegressor` (from `sklearn.ensemble`)
   - **Use Cases**: Classification and regression, often for tabular data with diverse feature types.
   - **Why Effective**:
     - Combines multiple decision trees to reduce variance (overfitting)  
     - Robust to outliers and can handle various data distributions  
     - Provides feature importance, aiding interpretability
   - **Code Example**:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     rf = RandomForestClassifier(n_estimators=100, random_state=42)
     rf.fit(X_train, y_train)
     y_pred = rf.predict(X_test)
     ```

3. **Gradient Boosting**
   - **Class**: `GradientBoostingClassifier` (from `sklearn.ensemble`)  
     or external libraries: `XGBClassifier` (XGBoost), `LGBMClassifier` (LightGBM)
   - **Use Cases**: Classification (and regression) in a variety of domains, often state-of-the-art on tabular data.
   - **Why Effective**:
     - Iteratively corrects errors of previous weak learners  
     - Good balance of bias-variance trade-off  
     - Hyperparameter tuning can yield excellent performance
   - **Code Example (scikit-learn)**:
     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
     gbc.fit(X_train, y_train)
     y_pred = gbc.predict(X_test)
     ```

4. **Support Vector Machines (SVM)**
   - **Class**: `SVC` (for classification) or `SVR` (for regression) (from `sklearn.svm`)
   - **Use Cases**: High-dimensional spaces, text classification, image classification (with kernels).
   - **Why Effective**:
     - Capable of complex boundaries via kernel functions  
     - Often performs well on smaller datasets with well-separated classes  
     - Robust to outliers with appropriate kernel and regularization
   - **Code Example**:
     ```python
     from sklearn.svm import SVC
     svm_model = SVC(kernel='rbf', C=1.0)
     svm_model.fit(X_train, y_train)
     y_pred = svm_model.predict(X_test)
     ```

5. **K-Nearest Neighbors (KNN)**
   - **Class**: `KNeighborsClassifier` or `KNeighborsRegressor` (from `sklearn.neighbors`)
   - **Use Cases**: Simple classification or regression tasks where interpretability and ease of use are desired.
   - **Why Effective**:
     - Instance-based (no explicit training phase, aside from storing examples)  
     - Highly interpretable: predictions are based on similarity to neighbors  
     - Works well for small to medium-sized datasets with well-defined clusters
   - **Code Example**:
     ```python
     from sklearn.neighbors import KNeighborsClassifier
     knn = KNeighborsClassifier(n_neighbors=5)
     knn.fit(X_train, y_train)
     y_pred = knn.predict(X_test)
     ```

### **Unsupervised Methods**

6. **K-Means Clustering**
   - **Class**: `KMeans` (from `sklearn.cluster`)
   - **Use Cases**: Grouping unlabeled data into clusters, e.g., customer segmentation, exploratory data analysis.
   - **Why Effective**:
     - Simple and widely used partitioning method  
     - Useful baseline for clustering; straightforward to implement and interpret  
     - Works best when clusters are roughly spherical in feature space
   - **Code Example**:
     ```python
     from sklearn.cluster import KMeans
     kmeans = KMeans(n_clusters=3, random_state=42)
     kmeans.fit(X)
     labels = kmeans.predict(X)
     ```

7. **DBSCAN** *(Alternative Unsupervised Method)*  
   - **Class**: `DBSCAN` (from `sklearn.cluster`)
   - **Use Cases**: Detecting clusters of arbitrary shape and outliers in unlabeled data.
   - **Why Effective**:
     - Automatically determines number of clusters based on density  
     - Identifies outliers as points not belonging to any cluster  
     - Good choice for data with clusters of varying shapes and noise
   - **Code Example**:
     ```python
     from sklearn.cluster import DBSCAN
     dbscan = DBSCAN(eps=0.5, min_samples=5)
     labels_dbscan = dbscan.fit_predict(X)
     ```

---

## **Why These Are Effective**

1. **Data Preprocessing**  
   - **Imputation & Outlier Handling**: Preserves data integrity and reduces extreme distortions.  
   - **Scaling & Encoding**: Many algorithms rely on numerical stability or distance metrics, which require scaling/categorical transformation.  
   - **Feature Selection & Dimensionality Reduction**: Helps focus on the most informative features, mitigating the “curse of dimensionality” and improving efficiency.

2. **Machine Learning Algorithms**  
   - **Wide Coverage of Tasks**:  
     - **Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN** (all supervised) for classification/regression tasks with labeled data.  
     - **K-Means, DBSCAN** (unsupervised) for clustering or anomaly detection when labels are not available.
   - **Performance and Interpretability**:  
     - **Logistic Regression** → interpretable; a good baseline.  
     - **Random Forest** → robust, handles feature diversity.  
     - **Gradient Boosting** → often best-in-class for tabular data.  
     - **SVM** → powerful for high-dimensional data.  
     - **KNN** → simple, instance-based approach.  
     - **K-Means, DBSCAN** → common clustering methods for exploratory data analysis.

---

### 4. (Potential) Results and Discussion
Identify several quantitative metrics you plan to use for the project (i.e. ML Metrics). Present goals in terms of these metrics, and state any expected results.

- **3+ Quantitative Metrics**  
- **Project Goals** (recommended to include sustainability and ethical considerations)  
- **Expected Results**

### 5. References
Cite relevant papers and articles utilizing the IEEE format. All references in this section must have a matching in-text citation in the body of your proposal text.

- **3+ References** (preferably peer reviewed)  
- **1+ In-Text Citation Per Reference**

---   
- **Gantt Chart**: list each member’s planned responsibilities for the entirety of the project. Feel free to use the Fall and Spring semester sample Gantt Chart.  

```gantt
    dateFormat  YYYY-MM-DD
    title Example Project Gantt Chart
    
    section Phase 1: Problem Definition
    Literature Review      :task1, 2025-02-01, 2025-02-07
    Dataset Research       :task2, after task1, 5d
    
    section Phase 2: Data Collection & Preprocessing
    Data Collection        :task3, 2025-02-13, 2025-02-20
    Data Cleaning          :task4, after task3, 5d
    Feature Engineering    :task5, after task4, 5d
    
    section Phase 3: Modeling
    Model Selection        :task6, 2025-03-01, 2025-03-07
    Model Training         :task7, after task6, 7d
    
    section Phase 4: Evaluation & Report
    Model Evaluation       :task8, 2025-03-15, 2025-03-21
    Proposal/Report Prep   :task9, after task8, 5d
    Video Presentation     :task10, after task9, 3d
```

- **Contribution Table**: list all group members’ names and explicit contributions in preparing the proposal using the format below.

     | Name    | Proposal Contributions |
     |---------|------------------------|
     | Naman Goyal | Contributions       |
     | Aryeman Singh | Contributions     |
     | Sameer Arora  | Contributions  |
     | Aryika Kumar | Contributions |
     | Sanjay Lokkit Babu Narayanan | Contributions |


2. **Video Presentation**  
   - 

3. **GitHub Repository**  
   - 

