---
layout: page
title: Midpoint Checkpoint
permalink: /midpoint
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data. This study focuses on classifying machine learning research abstracts based on the machine learning (ML) model(s)/algorithm(s) used in a research paper. Prior work highlights effective models such as **Support Vector Machines**, **Decision Trees**, and **Naïve Bayes**, which, when trained on well-defined textual features provided from pre-processing methods can provide robust classification [1]–[4]. 

To evaluate classification accuracy, this study will test machine learning models on **Introduction** and **Conclusion** sections of research papers sourced from [**Papers with Code**](https://paperswithcode.com/) via their free, public API. These sections often summarize the key contributions and methodology of a paper, making them a suitable dataset for classification. The objective is to determine the most effective approach for categorizing research papers based on content rather than simple keyword matching.

## Problem Definition
As the world of machine learning focused research expands, gathering relevant literature for review within a specific domain gets increasingly difficult. Modern keyword matching techniques do not accurately capture deeper semantic meaning and context. As a result, a structured approach to classify machine learning research papers and provide researchers with quick access to relevant papers and efficiently analyze their domain. 

## Methods

### Data Preprocessing Methods Implemented
Our data collection contains two key steps—obtaining our `y-actual`, our labels that signify which ML algorithm/model is used by querying the **PapersWithCode** API, collecting each paper's methodology information, and translating them into vector formats. The other half of our data collection pertains to obtaining the text for the introduction and conclusion of each paper, from their PDF's, provided by the **PapersWithCode** API. This data enables our model to form the `y-predicted` labels, contextually labelling each paper with the models used in that paper. 

In the files `PapersWithCodeAPICalling.ipynb` and `PDFProcessing.ipynb` under the `Data Extraction` directory, we complete these two main data collection steps and undergo the initial steps of our our **Data Transformation** and **Data Cleaning**. Here, we standardize the labels for each class / category possible, throw out papers that have malformed scans of the "Introduction "and "Conclusion" and ensure strictly the text from these two section is used (excluding "References," "Methods," etc.). We save all of this into a CSV which is fully stored in our `Datasets` directory, and as we use the CSV, we ensure that we split our 725 papers for 80% training and 20% testing. 

We conduct further data preprocessing in our `Feature Extraction` directory, under `Feature Extraciton.ipynb`. To begin, we leverage **NLTK**'s **word_tokenize**, **WordNetLemmatizer**, and **stopwords** module for text normalization and tokenization to lowercase the words and remove punctuation, stopwords, and reference numbers of the format "[1]", "[2]", etc. After that, we call **PyTorch**'s transformers library to use **BertTokenizer**, **BertModel**, and **SentenceTransformer** to generate dense vector representations of each block of text and capture contextual meaning, based on the `all-mpnet-base-v2` embedding model. This enabled us to conduct multi-labelling of each paper with 14 general labels that portrays the ML models/algorithms used in that paper, such as "CNN-based," or "Logistic-Regression". 

Next, we used a pretrained transformer-based language model, **PyTorch**'s **Sentence-BERT (all-mpnet-base-v2)**, to extract **768-dimensional sentence embeddings** for each paper's combined introduction & conclusion. These embeddings capture deep semantic meaning and were chosen due to their strong performance in semantic textual similarity tasks [3], [4]. 
### Machine Learning Algorithms/Models Implemented

#### Used for Exploratory Data Analysis

We also conducted exploratory data analysis in our `Feature Extraction` directory, under `Feature Extraciton.ipynb`. To explore the structure of our data without labels, we applied unsupervised learning techniques to the BERT embeddings: UMAP (Uniform Manifold Approximation and Projection) and HDBSCAN (Hierarchical Density-Based Spatial Clustering). 

UMAP reduced high-dimensional BERT embeddings (768-d) to 2D for visualization and helps preserve both global and local structure better than t-SNE or PCA (which we also tested out in `Feature Extraction.ipynb`). 

HDBSCAN discovered dense clusters of similar papers in the 2D UMAP space. This technique works well with irregular cluster shapes, handles noise robustly, and does not require pre-setting the number of clusters (so we may understand the natural grouping of our corpus)

Together, these models provided meaningful unsupervised insights into natural groupings in our corpus, validating semantic distinctions captured by BERT embeddings.

#### Used for Paper Classification
We implemented a Logistic Regression classifier in a One-vs-Rest (OvR) setting to perform multi-label classification over 14 potential categories.

The details of the model include using **sklearn**'s **LogisticRegression** module with the following hyper-parameters: `C = 10.0` (a relatively small penalty term, allowing the model to fit the training data more closely but runs the risk of overfitting) (, `class_weight = 'balanced'` (to handle label imbalance), and `max_iter = 1000`.

Each paper's introduction and conclusion is encoded into a 768-dimensional BERT embedding that captures rich semantic features of the text—such as the research method, goal, and key findings. Logistic Regression then uses these numerical embeddings as input features to learn decision boundaries for each label. This allows the model to effectively distinguish between different machine learning categories based on meaning, rather than just surface-level word counts.

This method was selected for its simplicity, interpretability, and effectiveness when paired with high-quality embeddings like BERT. for its simplicity, interpretability, and effectiveness when paired with high-quality embeddings like BERT.

### OLD **Classification Models**
**Once high-dimensional representations of text are generated, the following machine learning models will be tested for classification.**

**Naïve Bayes (NB), implemented via MultinomialNB in scikit-learn, is a probabilistic classifier that assumes feature independence, enabling efficient training and scalability [3], [4].** 

**Support Vector Machines (SVM), using SVC from scikit-learn, construct hyperplanes for binary classification and leverage kernel functions to handle high-dimensional spaces effectively [1], [4].** 

**Random Forest, implemented with RandomForestClassifier in scikit-learn, aggregates multiple decision trees for classification but requires careful tuning to balance computation time and overfitting risk [1], [4].**

## OLD**Results and Discussion**
#### **Metrics**
**To assess model performance, we will use accuracy, precision, recall, and F1-score. Accuracy measures the proportion of correct predictions among total predictions. Precision evaluates how many of the predicted positive classifications were actually correct. Recall measures the proportion of actual positives correctly classified. F1-score balances precision and recall, particularly useful when handling imbalanced datasets.**

#### **Project Goals**
**We hope to identify which classification models are the most accurate and have the highest F-1 score. The project also considers sustainability and ethical considerations, aiming to find a computationally efficient model that minimizes mis-categorization and overrepresentation of dominant research fields.** 

#### **Expected Results**
**We expect SVM to perform the best in accuracy and F1 Score due to its suitability for high dimensional text classification [3]. Naïve Bayes will offer strong results for smaller text but might struggle with complexity and will likely be the most resource-efficient [4]. Random Forest may overfit but offers interpretability and potential for high accuracy [1].**


## Results and Discussion
### Quantitative Metrics

| Metric               | Value      |
| -------------------- | ---------- |
| Exact Match Accuracy | **0.3931** |
| Hamming Loss         | **0.0679** |
| F1 Score (Micro)     | **0.5975** |
| F1 Score (Macro)     | **0.6395** |
These scores indicate a moderate overall performance. 39% of test samples had all predicted labels exactly correct and only 6.8% of the label entries were wrong. Our F1 Micro assumes we treat each prediction equally and understand the overall ability to make correct label predictions—about 60% of (label, sample) pairs were precise and complete. F1 macro is especially important for imbalanced datasets and treats each class equally. This metric showed strong results at ~64%. This suggests that our model handles rare classes reasonably well. 
### Visualization
#### EDA Visualizations

![Label Frequency](/assets/label-frequency.png)
#### Model Visualizations

### Analysis of 1+ Algorithm(s)/Model(s)
### Next Steps

## References
[1] A. Gasparetto, M. Marcuzzo, A. Zangari, and A. Albarelli, “A survey on text classification algorithms: From text to predictions,” _Information_, vol. 13, no. 2, Feb. 2022. doi:10.3390/info13020083

[2] C. C. Aggarwal and C. Zhai, “A survey of text classification algorithms,” _Mining Text Data_, pp. 163–222, 2012. doi:10.1007/978-1-4614-3223-4_6

[3] I. Dawar, N. Kumar, S. Negi, S. Pathan, and S. Layek, “Text categorization using supervised machine learning techniques,” _2023 Sixth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU)_, Mar. 2023. doi:10.1109/wids-psu57071.2023.00046

[4] K. Shyrokykh, M. Girnyk, and L. Dellmuth, “Short text classification with machine learning in the Social Sciences: The case of climate change on Twitter,” _PLOS ONE_, vol. 18, no. 9, Sep. 2023. doi:10.1371/journal.pone.0290762

---   
## Other
### [Full Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/akumar906_gatech_edu/EXJc6ihn5flFu38MAn05b_4BXmzhr109P-YNltiuoURhIg?e=5WeXAZ)

### Contribution Table
Based on the template on the class website, here is the contribution table:

| Name                         | Proposal Contributions                        |
| :--------------------------- | :-------------------------------------------- |
| Aryeman Singh                | GitHub Pages, Problem Definition, Motivation  |
| Sameer Arora                 | Introduction, Methods, Discussion, References |
| Naman Goyal                  | Video Recording, Video Creation, Gantt Chart  |
| Lokkit Sanjay Babu Narayanan | Methods, Presentation, Potential Dataset      |
| Aryika Kumar                 | Gantt Chart, Results, Presentation            |


### [GitHub Repository](https://github.gatech.edu/asingh899/ml_project_43/)

### We would like opt-in to be considered for the “Outstanding Project” award.
