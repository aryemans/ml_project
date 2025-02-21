---
layout: page
title: Proposal
permalink: /proposal
---

## Proposal Sections & Checklist

### 1. Introduction/Background

Within Natural Language Processing (NLP), text classification is key challenge that is being heavily researched by many teams across the field. Text classification has applications in various domains, including machine learning research and their associated papers. Prior research into research paper classification has involved approaches from traditional models, such as Support Vector Machines, due to their effiency and interpretability. The models, when trained on well-defined features of text data, can offer robust classification of structured texts, such as those of research papers.

Within text classification, developments have also occurred in the preprocessing steps for the text, where techniques such as topic modeling, term frequency - inverse document frequeuncy, and bag-of-words have arisen to aid in extracting features from the text. This study intends to analyze the effectiveness of traditional machine learning models in classifying machine learning research abstracts by the research area and the machine learning methodologies used within the paper. The study will evaluate different algorithms to determine which models yield the highest accuracy.

The dataset will consist of machine learning research abstracts, labeled by machine learning methodologis and the research field. The preprocessing pipelinee will include x......x to extract features such as x.....x. 

The dataset will be created by pulling peer-reviewed papers and their associated abstracts from the following site: https://paperswithcode.com/. 

### 2. Problem Definition
 
As machine learning research expands and the number of papers published grows, it is pertinent for researchers to efficiently gather other papers which use similar machine learning methods within the same research domain. Keyword matching forms the basis of today's search methods, yet this may not accurately account for methodological similarities between papers in the same domain. Any ineffiencies within similarity search for papers can introduce bottlenecks to comparing the results of research.

A structured approach to reseach abstract classification by both research topic and machine learning methodology would enable researchers to find other relevant papers quickly. By evaluating traditional models, the study can provide an interpretable and efficient solution to categorize research and streamline the process of comparing research results.

### 3. Methods

## Data Pre-Processing Methods
To effectively classify research paper abstracts, robust data preprocessing is essential. Tokenization and stopword removal help refine the text by breaking it into words and removing common terms that do not contribute to classification. TF-IDF transformation converts abstracts into numerical representations by weighing words based on their frequency and importance in the dataset. This is particularly useful for models like SVM. For deeper contextual understanding, word embeddings such as BERT provide dense vector representations that capture relationships between words in research abstracts. Additionally, lemmatization and stemming standardize words to their base forms, ensuring consistency in the dataset. These preprocessing steps improve the quality of the input data, making machine learning models more efficient and accurate.

## Machine Learning Algorithms
For classification, Support Vector Machines (SVMs) are used as a supervised learning model to assign abstracts to predefined categories. When labeled data is scarce, K-Means Clustering groups abstracts into clusters based on textual similarities, aiding in topic discovery. This can also be used for semi-supervised learning, where pseudo-labels from clusters expand the training set for SVM. Additionally, Latent Dirichlet Allocation (LDA) extracts underlying research topics, helping interpret abstract clusters. By integrating these models, we combine supervised classification with unsupervised topic discovery, enhancing both structured classification and exploratory analysis of research papers.

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

