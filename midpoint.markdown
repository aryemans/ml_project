---
layout: page
title: Midpoint Checkpoint
permalink: /midpoint
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data. This study focuses on classifying machine learning research abstracts based on the machine learning (ML) model(s)/algorithm(s) used in a research paper. Prior work highlights effective models such as **Logistic Regression**, **Naïve Bayes**, and **Random Forest** which, when trained on well-defined textual features provided from pre-processing methods can provide robust classification [1]–[4]. 

To evaluate classification accuracy, this study will test machine learning models on **Introduction** and **Conclusion** sections of research papers sourced from [**Papers with Code**](https://paperswithcode.com/) via their free, public API. These sections often summarize the key contributions and methodology of a paper, making them a suitable dataset for classification. The objective is to determine the most effective approach for categorizing research papers based on content rather than simple keyword matching.

## Problem Definition
As the world of machine learning focused research expands, gathering relevant literature for review within a specific domain gets increasingly difficult. Modern keyword matching techniques do not accurately capture deeper semantic meaning and context. As a result, our structured approach to classify machine learning research papers provides researchers with quick access to relevant papers and efficiently analyze their domain. 

## Methods

### Data Preprocessing Methods Implemented
Our data collection contains two key steps—obtaining our `y-actual`, our labels that signify which ML algorithm/model is used by querying the **PapersWithCode** API, collecting each paper's methodology information, and translating them into vector formats. The other half of our data collection pertains to obtaining the text for the introduction and conclusion of each paper, from their PDF's, provided by the **PapersWithCode** API. This data enables our model to form the `y-predicted` labels, contextually labelling each paper with the models used in that paper. 

In the files `PapersWithCodeAPICalling.ipynb` and `PDFProcessing.ipynb` under the `Data Extraction` directory, we complete these two main data collection steps and undergo the initial steps of our our **Data Transformation** and **Data Cleaning**. Here, we standardize the labels for each class / category possible, throw out papers that have malformed scans of the "Introduction "and "Conclusion" and ensure strictly the text from these two section is used (excluding "References," "Methods," etc.). We save all of this into a CSV which is fully stored in our `Datasets` directory, and as we use the CSV, we ensure that we split our 725 papers for 80% training and 20% testing. 

We conduct further data preprocessing in our `Feature Extraction` directory, under `Feature Extraction.ipynb`. To begin, we leverage **NLTK**'s **word_tokenize**, **WordNetLemmatizer**, and **stopwords** module for text normalization and tokenization to lowercase the words and remove punctuation, stopwords, and reference numbers of the format "[1]", "[2]", etc. After that, we call **PyTorch**'s transformers library to use **BertTokenizer**, **BertModel**, and **SentenceTransformer** to generate dense vector representations of each block of text and capture contextual meaning, based on the `all-mpnet-base-v2` embedding model. This enabled us to conduct multi-labelling of each paper with 14 general labels that portrays the ML models/algorithms used in that paper, such as "CNN-based," or "Logistic-Regression". 

Next, we used a pretrained transformer-based language model, **PyTorch**'s **Sentence-BERT (all-mpnet-base-v2)**, to extract **768-dimensional sentence embeddings** for each paper's combined introduction & conclusion. These embeddings capture deep semantic meaning and were chosen due to their strong performance in semantic textual similarity tasks [3], [4]. 
### Machine Learning Algorithms/Models Implemented

#### Used for Exploratory Data Analysis

We also conducted exploratory data analysis in our `Feature Extraction` directory, under `Feature Extraction.ipynb`. To explore the structure of our data without labels, we applied unsupervised learning techniques to the BERT embeddings: UMAP (Uniform Manifold Approximation and Projection) and HDBSCAN (Hierarchical Density-Based Spatial Clustering). 

UMAP reduced high-dimensional BERT embeddings (768-d) to 2D for visualization and helps preserve both global and local structure better than t-SNE or PCA (which we also tested out in `Feature Extraction.ipynb`). 

HDBSCAN discovered dense clusters of similar papers in the 2D UMAP space. This technique works well with irregular cluster shapes, handles noise robustly, and does not require pre-setting the number of clusters (so we may understand the natural grouping of our corpus)

Together, these models provided meaningful unsupervised insights into natural groupings in our corpus, validating semantic distinctions captured by BERT embeddings.

#### Used for Paper Classification
We implemented a Logistic Regression classifier in a One-vs-Rest (OvR) setting to perform multi-label classification over 14 potential categories.

The details of the model include using **sklearn**'s **LogisticRegression** module with the following hyper-parameters: `C = 10.0` (a relatively small penalty term, allowing the model to fit the training data more closely but runs the risk of overfitting) (, `class_weight = 'balanced'` (to handle label imbalance), and `max_iter = 1000`.

Each paper's introduction and conclusion is encoded into a 768-dimensional BERT embedding that captures rich semantic features of the text—such as the research method, goal, and key findings. Logistic Regression then uses these numerical embeddings as input features to learn decision boundaries for each label. This allows the model to effectively distinguish between different machine learning categories based on meaning, rather than just surface-level word counts.

This method was selected for its simplicity, interpretability, and effectiveness when paired with high-quality embeddings like BERT. for its simplicity, interpretability, and effectiveness when paired with high-quality embeddings like BERT.
## Results and Discussion
### Quantitative Metrics

| Metric               | Value      |
| -------------------- | ---------- |
| Exact Match Accuracy | **0.3931** |
| Hamming Loss         | **0.0679** |
| F1 Score (Micro)     | **0.5975** |
| F1 Score (Macro)     | **0.6395** |

These scores indicate a moderate overall performance. 39% of test samples had all predicted labels exactly correct and only 6.8% of the label entries were wrong. Our F1 Micro assumes we treat each prediction equally and understand the overall ability to make correct label predictions—about 60% of (label, sample) pairs were precise and complete. F1 macro is especially important for imbalanced datasets and treats each class equally. This metric showed strong results at ~64%. This suggests that our model handles rare classes reasonably well. 
### EDA Visualizations
#### Label Frequency

![Label Frequency](/assets/label-frequency.png)
Conducting this visualization opened our eyes to the possibility of bias in the model due to imbalanced data, which is why when we trained our Logistic Regression model, we used the `class_weight` hyperparameter to balance the weight of each class and we threw our labels, like ensemble-based which had a frequency of 0 in our training set.

#### HDBSCAN Clustering on UMAP Embeddings
![Label Frequency](/assets/hdbscan-umap.png)
This 2D visualization consisted of dimensionality reduction (UMAP) and conduct density-based clustering (HDBSCAN). We leveraged this visualization to understand the label space and ascertain insight into our corpus. The resulting clustering suggests that papers with similar semantic content are naturally grouped together in the embedding space, validating the idea that BERT embeddings capture deeper topical signals from the text. 

![Label Frequency](/assets/comparison.png)
We compared this visualized to t-SNE and PCA to find which preserved a relevant global and local structure, so as to not distort our clusterings by other linear or non-linear dimensionality reduction techniques. 
### Model Visualizations
#### Per-Class F1 vs. Threshold
![Label Frequency](/assets/per-class-f1-score.png)
Visualizing the Per-Class F1 vs. Threshold illuminates the optimal threshold for each class that maximized F1. We use this visualization to evaluate individual class performance for each label individually. In our plot, some classes (like "Transformer-based") peak around 0.6–0.7, while others (like "Object Detection") perform best at much lower thresholds. This tells us that a single threshold across all classes would underperform—class-specific tuning is crucial and will be included in our next steps. 
#### Impact on Global Micro F1
![Label Frequency](/assets/impact-micro-F1.png)
In conjunction with the previous visualization, this plot indicates which classes have a disproportionate affect on the overall model performance (the previous visualization illuminates the thresholds we may tune—this visualization portrays the impact of tuning each one of those thresholds). This impact may potentially be due to a class' frequency or correlation with other labels. 

#### Macro F1 vs Per-Class Threshold
![Label Frequency](/assets/macro-f1-per-class.png)
Since macro F1 weighs all classes equally, this visualization reveals how rare or poorly performing labels influence fairness across the label space. We observed that tuning thresholds for low-frequency classes (like "Object Detection") causes sharper shifts in macro F1 than common ones — helpful for diagnosing imbalance.

#### ROC Curves per Class
![Label Frequency](/assets/roc.png)
Helps assess class separability. AUC values close to 1.0 (like for "Q-Learning") show strong classifier confidence. In contrast, flatter curves indicate ambiguous or overlapping feature distributions. This helps identify which labels the model finds easiest or hardest to distinguish based on BERT embeddings.
### Analysis of Algorithm(s)/Model(s)

Our model exhibited several strengths that contributed to its moderate success. By leveraging high-quality semantic representations from BERT, the model was able to understand the meaning and context of each paper, going beyond simple keyword matching. The use of class-specific threshold optimization allowed it to better handle imbalanced class distributions and F1 performance. Additionally, the interpretability of logistic regression helped us analyze and understand which dimensions in the embedding space contributed to each prediction and enabled us to further tweak our model's pipeline. 

However, the model also faced some limitations. The dataset contained significant class imbalance, with some categories such as 'Object Detection' having very few examples, making them difficult to learn reliably. Logistic regression, being a linear model, may not capture complex, nonlinear relationships present in the BERT embeddings. Furthermore, the BERT embeddings used were frozen and not fine-tuned on our specific dataset, which limits their ability to adapt to domain-specific subtleties. Finally, since the One-vs-Rest approach treats each label independently, it does not take advantage of correlations and co-occurrence patterns among labels that could potentially enhance prediction accuracy.
### Next Steps
For this model, we may proceed with  fine-tuning BERT directly for our classification task at-hand.

In regards to other models, we want to explore the possibility of leveraging **Naive Bayes** to compare its efficacy to that of the **Logistic Regression** model. In an effort to make Naive Bayes compatible with our BERT embeddings, in order to capture semantic meaning in our models' classifications, we will extract and engineer features that make our embedding data compatible with Naive Bayes.

Other avenues we will be exploring is similarly leveraging **Random Forest** in a way that is compatible with our BERT embeddings, so as to capture semantic meaning and continue in our pursuit to learn more about models that can label text-based data. 

A key aspect of our building of these models include the cognizance of the multi-labelling task we have at-hand. Incorporating software to handle these cases will look different for each model—for logistic regression, we had to experiment with threshold (which we will similarly have to do for Naive Bayes), and for random forest, we will have to analyze an extra classifier for each class that posits the question of "*reject* or *accept*". 

Lastly, we will be conducting a rigorous analysis where we will be examining the shortcomings of  and the particular strengths of each model. We will also conduct our own testing to illuminate the presence of an semantic understanding of text in each model, rather than just fancy keyword-matching, through a variety of techniques (like artificial data poisoning). Crucially, we will also validate our analysis with more metrics, due to the unique task of multi-labelling that we conduct with our research.

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

| Name                         | Proposal Contributions                                                          |
| :--------------------------- | :------------------------------------------------------------------------------ |
| Aryeman Singh                | Data Extraction & Pre-processing, Model Training, Gantt Chart, GitHub Pages     |
| Sameer Arora                 | Data Extraction, Led Report Writing, GitHub Pages                               |
| Naman Goyal                  | Helped with Report Writing, Led Feature Extraction, Model Training, Gantt Chart |
| Lokkit Sanjay Babu Narayanan | Feature Extraction, Model Training                                              |
| Aryika Kumar                 | Feature Extraction, Model Training                                              |


### [GitHub Repository](https://github.gatech.edu/asingh899/ml_project_43/)

### We would like opt-in to be considered for the “Outstanding Project” award.
