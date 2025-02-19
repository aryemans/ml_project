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
     - **K-Means, DBSCAN, GMM** → common clustering methods for exploratory data analysis.

---

### 4. (Potential) Results and Discussion
Identify several quantitative metrics you plan to use for the project (i.e. ML Metrics). Present goals in terms of these metrics, and state any expected results.

- **3+ Quantitative Metrics**  
---

## **1. Classification Metrics**

### 1.1 Accuracy
- **Definition**: Proportion of correct predictions among all predictions.
- **Usefulness**: Straightforward measure, but can be misleading if classes are highly imbalanced.
- **Vague Goal**: Aim for a reasonably high accuracy (e.g., 70–95% range) as an initial benchmark.

### 1.2 Balanced Accuracy
- **Definition**: Average of recall obtained on each class, mitigating bias in imbalanced datasets.
- **Usefulness**: More reliable than simple Accuracy when class sizes differ significantly.
- **Vague Goal**: Substantially higher than random chance (often > 0.70).

### 1.3 Precision & Recall
- **Precision**: Out of predicted positives, how many are truly positive?  
  \[
    \text{Precision} = \frac{TP}{TP + FP}
  \]
- **Recall (Sensitivity)**: Out of actual positives, how many did we correctly predict?  
  \[
    \text{Recall} = \frac{TP}{TP + FN}
  \]
- **Usefulness**:  
  - **Precision** is crucial when false positives are costly.  
  - **Recall** is important when false negatives are costly.
- **Vague Goal**: Balance these or focus on whichever aligns with project needs (e.g., > 0.80 if you want few false positives).

### 1.4 F1 Score (Macro, Micro, Weighted)
- **Definition**: Harmonic mean of Precision and Recall.  
  - **Macro-F1**: Averages F1 across all classes equally.  
  - **Micro-F1**: Weighs classes by their size (sum of TP across classes, etc.).  
  - **Weighted-F1**: Similar to Macro-F1 but weights by class frequency.
- **Usefulness**: Good single measure to balance Precision and Recall, especially for imbalanced or multi-class tasks.
- **Vague Goal**: 0.70–0.80+ can be considered strong, depending on complexity.

### 1.5 ROC-AUC (Receiver Operating Characteristic – Area Under Curve)
- **Definition**: Measures how well the model ranks positive instances higher than negative ones across thresholds; 1.0 = perfect, 0.5 = random.
- **Usefulness**: Threshold-independent metric, widely used for binary classification (can be extended to multi-class via one-vs-rest).
- **Vague Goal**: Above 0.80 often indicates good discrimination.

### 1.6 Precision-Recall AUC
- **Definition**: Area under the Precision-Recall curve; more informative than ROC-AUC if classes are highly imbalanced.
- **Usefulness**: Focuses on positive class performance under various thresholds, crucial for skewed data.
- **Vague Goal**: Significantly above the baseline (the proportion of positives); e.g., 0.40–0.70+ depending on difficulty.

### 1.7 Matthews Correlation Coefficient (MCC)
- **Definition**: Correlation coefficient between observed and predicted class labels (ranges from -1 to +1).
- **Usefulness**: Considers all four confusion matrix elements (TP, TN, FP, FN); robust for imbalanced data.
- **Vague Goal**: An MCC above 0 indicates better-than-random; 0.5+ is fairly strong.

### 1.8 Cohen’s Kappa
- **Definition**: Measures the agreement between predicted and actual labels adjusted for chance.
- **Usefulness**: Another way to account for imbalances or random agreement.
- **Vague Goal**: Values > 0.60 often considered “good,” but context-dependent.

### 1.9 Jaccard Score (Intersection over Union)
- **Definition**: For binary classification, \(\frac{TP}{TP+FP+FN}\). In multi-class or multi-label, it’s averaged over classes/labels.
- **Usefulness**: Alternative measure similar to F1 for multi-label classification, measuring the overlap between predicted and actual sets.
- **Vague Goal**: Aim for a Jaccard index significantly above random guess; e.g., > 0.70 could be strong.

### 1.10 Logarithmic Loss (Log Loss / Cross-Entropy)
- **Definition**: Penalizes confident but incorrect probabilistic predictions more heavily.
- **Usefulness**: Excellent for **probabilistic** classification performance.
- **Vague Goal**: Lower is better; specific targets depend on data distribution.

---

## **2. Regression Metrics**

### 2.1 R-squared (\(R^2\))
- **Definition**: Proportion of variance in the dependent variable explained by the model.
- **Usefulness**: High-level view of how well features explain outcome variance.
- **Vague Goal**: 0.70–0.90+ in many contexts indicates a decent fit, but domain-dependent.

### 2.2 Adjusted R-squared
- **Definition**: Like \(R^2\) but penalizes for adding features that don’t improve the model enough.
- **Usefulness**: Helps avoid overfitting, ensuring that adding variables is truly beneficial.
- **Vague Goal**: Should be as close to \(R^2\) as possible; big gaps might mean overfitting.

### 2.3 Mean Squared Error (MSE)
- **Definition**: Average of squared differences between predicted and actual values.
- **Usefulness**: Penalizes large errors more severely, widely used for many regression problems.
- **Vague Goal**: Minimize MSE; the acceptable range depends on the magnitude of target values.

### 2.4 Root Mean Squared Error (RMSE)
- **Definition**: Square root of MSE, in the same units as the target variable.
- **Usefulness**: Easier interpretability (same scale as targets).
- **Vague Goal**: Keep this below a certain threshold relevant to the domain (e.g., if the average target is 100, an RMSE of 10 might be good).

### 2.5 Mean Absolute Error (MAE)
- **Definition**: Average absolute difference between predicted and actual values.
- **Usefulness**: More robust to outliers than MSE, easier to interpret.
- **Vague Goal**: If the domain average is 100, an MAE of 8–12 might be considered moderate or good, depending on context.

### 2.6 Mean Absolute Percentage Error (MAPE)
- **Definition**: Average of \(\left|\frac{y_i - \hat{y}_i}{y_i}\right|\times 100\%\).
- **Usefulness**: Measures error as a percentage of actual values, making it scale-independent.
- **Vague Goal**: MAPE < 10–20% often indicates decent performance; domain-specific tolerance varies.

---

## **3. Clustering & Unsupervised Metrics**

### 3.1 Silhouette Coefficient
- **Definition**: Measures how similar an object is to its own cluster compared to other clusters (range -1 to 1).
- **Usefulness**: Higher values indicate well-separated, cohesive clusters.
- **Vague Goal**: > 0.5 might indicate decent separation, but it depends on the problem.

### 3.2 Calinski-Harabasz Index (Variance Ratio Criterion)
- **Definition**: Ratio of between-clusters dispersion to within-cluster dispersion.
- **Usefulness**: Higher values imply more distinct clusters.
- **Vague Goal**: No absolute threshold, but compare relative scores across different numbers of clusters or clustering methods.

### 3.3 Davies-Bouldin Index
- **Definition**: Average “similarity” (ratio of within-cluster distance to separation from other clusters). Lower is better.
- **Usefulness**: Provides a single numeric measure of cluster quality.
- **Vague Goal**: Lower indicates more compact and well-separated clusters (e.g., < 1 is often good in some contexts).

### 3.4 Adjusted Rand Index (ARI)
- **Definition**: Measures similarity between predicted clusters and true labels (if any exist). Adjusted for random chance.
- **Usefulness**: Good if you have ground-truth labels for a clustering task and want a measure of agreement.
- **Vague Goal**: Values near 1.0 mean perfect agreement; > 0.5 often suggests moderate agreement.

### 3.5 Normalized Mutual Information (NMI)
- **Definition**: Mutual information between true labels and cluster assignments, normalized to [0, 1].
- **Usefulness**: Another measure of how well clusters align with ground truth, ignoring label permutation.
- **Vague Goal**: A score closer to 1 indicates better alignment.

### 3.6 Homogeneity, Completeness, V-measure
- **Definitions**:  
  - **Homogeneity**: Each cluster has data points primarily of a single class.  
  - **Completeness**: Each class is contained mostly within a single cluster.  
  - **V-measure**: The harmonic mean of homogeneity and completeness.
- **Usefulness**: These give a multi-faceted view of cluster labeling quality when real labels are known.
- **Vague Goal**: Closer to 1 = better clustering quality.

---
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

