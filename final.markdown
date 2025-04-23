---
layout: page
title: Final Report
permalink: /final
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data. This study focuses on classifying machine learning research abstracts based on the machine learning (ML) model(s)/algorithm(s) used in a research paper. Prior work highlights effective models such as **Logistic Regression**, **Naïve Bayes**, and **Random Forest** which, when trained on well-defined textual features provided from pre-processing methods can provide robust classification [1]–[4]. 

To evaluate classification accuracy, this study will test machine learning models on **Abstract** and **Results** sections of research papers sourced from [**Papers with Code**](https://paperswithcode.com/) via their free, public API (we previously used used the **Introduction** and **Conlusion** sections for NLP analysis, but based on feedback from Professor Roozbahani and Richard Koulen, as well as our own analysis into the best sections of the paper to use for classification). These sections often summarize the key contributions and methodology of a paper, making them a suitable dataset for classification. The objective is to determine the most effective approach for categorizing research papers based on content rather than simple keyword matching.

## Problem Definition
As the world of machine learning focused research expands, gathering relevant literature for review within a specific domain gets increasingly difficult. Modern keyword matching techniques do not accurately capture deeper semantic meaning and context. As a result, our structured approach to classify machine learning research papers provides researchers with quick access to relevant papers and efficiently analyze their domain. 

## Methods

### Data Preprocessing Methods Implemented
Our data collection contains two key steps—obtaining our `y-actual`, our labels that signify which ML algorithm/model is used by querying the **PapersWithCode** API, collecting each paper's methodology information, and translating them into vector formats. The other half of our data collection pertains to obtaining the text for the abstract and results sections of each paper, from their PDF's, provided by the **PapersWithCode** API. This data enables our model to form the `y-predicted` labels, contextually labelling each paper with the models used in that paper. 

In the files `PapersWithCodeAPICalling.ipynb` and `PDFProcessing.ipynb` under the `Data Extraction` directory, we complete these two main data collection steps and undergo the initial steps of our our **Data Transformation** and **Data Cleaning**. Here, we standardize the labels for each class / category possible, throw out papers that have malformed scans of the "Abstract "and "Results" and ensure strictly the text from these two section is used (excluding "References," "Methods," etc.). We save all of this into a CSV which is fully stored in our `Datasets` directory, and as we use the CSV, we ensure that we split our 725 papers for 80% training and 20% testing. 

We conduct further data preprocessing in our `Feature Extraction` directory, under `Feature Extraction.ipynb`. To begin, we leverage **NLTK**'s **word_tokenize**, **WordNetLemmatizer**, and **stopwords** module for text normalization and tokenization to lowercase the words and remove punctuation, stopwords, and reference numbers of the format "[1]", "[2]", etc. After that, we call **PyTorch**'s transformers library to use **BertTokenizer**, **BertModel**, and **SentenceTransformer** to generate dense vector representations of each block of text and capture contextual meaning, based on the `all-mpnet-base-v2` embedding model. This enabled us to conduct multi-labelling of each paper with 14 general labels that portrays the ML models/algorithms used in that paper, such as "CNN-based," or "Logistic-Regression". 

Next, we used a pretrained transformer-based language model, **PyTorch**'s **Sentence-BERT (all-mpnet-base-v2)**, to extract **768-dimensional sentence embeddings** for each paper's combined abstract & results. These embeddings capture deep semantic meaning and were chosen due to their strong performance in semantic textual similarity tasks [3], [4]. 
### Machine Learning Algorithms/Models Implemented

#### Used for Exploratory Data Analysis

We also conducted exploratory data analysis in our `Feature Extraction` directory, under `Feature Extraction.ipynb`. To explore the structure of our data without labels, we applied unsupervised learning techniques to the BERT embeddings: UMAP (Uniform Manifold Approximation and Projection) and HDBSCAN (Hierarchical Density-Based Spatial Clustering). 

UMAP reduced high-dimensional BERT embeddings (768-d) to 2D for visualization and helps preserve both global and local structure better than t-SNE or PCA (which we also tested out in `Feature Extraction.ipynb`). 

HDBSCAN discovered dense clusters of similar papers in the 2D UMAP space. This technique works well with irregular cluster shapes, handles noise robustly, and does not require pre-setting the number of clusters (so we may understand the natural grouping of our corpus)

Together, these models provided meaningful unsupervised insights into natural groupings in our corpus, validating semantic distinctions captured by BERT embeddings.

#### Logistic Regression
##### **Logistic Regression with TF-IDF Features**
We implemented a multi-label Logistic Regression classifier using **sklearn**’s `LogisticRegression` module, wrapped in a `OneVsRestClassifier` to handle the multi-label nature of the task.

The input features were constructed by applying **TF-IDF vectorization** to the combined abstract and result section of each research paper. The vectorizer was configured with the following hyperparameter—`max_features = 3000`.

This produces sparse, high-dimensional feature vectors that capture the importance of words and phrases across the corpus.

The Logistic Regression classifier was initialized with:  
- `C = 10.0` (encouraging closer fits to the training data)  
- `max_iter = 1000` (to ensure convergence)  
- `class_weight = 'balanced'` (to correct for label imbalance)

After training, the classifier’s predicted probabilities were passed through a **threshold optimization routine** to improve multi-label decision making. Each label’s threshold was individually tuned using grid search over the range [0.1, 0.9], selecting the value that maximized the average of F1-micro, F1-macro, and F1-weighted scores.

This approach balances simplicity and interpretability with solid performance using sparse lexical features.
##### **Logistic Regression with BERT Embeddings**

We also implemented a Logistic Regression classifier using **768-dimensional BERT embeddings** as input features. These embeddings, precomputed from each paper’s abstract and result, capture rich semantic representations that go beyond surface-level word statistics.

Similar to the TF-IDF pipeline, we used **sklearn**’s `LogisticRegression` in a `OneVsRestClassifier` setup, configured with:  
- `C = 10.0`  
- `max_iter = 1000`  
- `class_weight = 'balanced'`  

Prior to training, we removed "dead" labels (i.e., classes with no positive samples) to ensure numerical stability.

Following training, a **per-class threshold optimization** was performed. Each class’s threshold was adjusted over a fine-grained grid [0.1, 0.9], optimizing for a combined F1 score across multiple averaging strategies (macro, micro, weighted). This process increased performance by tailoring the classification sensitivity for each label.

This model benefits from the deep contextual signals encoded in BERT embeddings while maintaining the interpretability and scalability of linear classifiers.
#### Naive Bayes with TF-IDF Features

We implemented a Naive Bayes classifier using a **One-vs-Rest (OvR)** strategy to handle multi-label classification across 16 research categories. After receiving guidance on providing a baseline of our BERT-based models to show potential enhancement or shortcoming of leveraging semantic understanding via BERT, we decided to implement a Naive Bayes, TF-IDF-based model. 

For input features, we applied **TF-IDF vectorization** to each paper’s abstract and results section after preprocessing. Preprocessing included tokenization, stopword removal, and lemmatization using **NLTK**, followed by transformation using **sklearn**’s `TfidfVectorizer` with the following configuration: `max_features = 175` (to not over-complicate the simple model) and `ngram_range = (1, 1)`.

This setup captures unigram term importance while reducing the feature space for efficiency—so as to make it as close to a simple keyword-based classifier as possible.

The classifier used was **sklearn**’s `MultinomialNB` wrapped in a `OneVsRestClassifier`, with: `alpha = 0.8` (to apply smoothing and prevent zero probabilities).

The model was trained on the TF-IDF features and evaluated using F1-micro and F1-macro metrics. We further refined classification performance using a **threshold optimization** routine. For each class, the probability threshold was tuned in the range `[0.1, 0.9]` to maximize the average of F1-macro scores. This allowed the model to account for class imbalance and better control the decision boundary for each label.

This Naive Bayes approach provides a simple yet effective baseline for multi-label text classification and demonstrates reasonable performance when paired with carefully engineered keyword-based features and threshold tuning.
#### Support Vector Machine (SVM)
##### **TF-IDF + LinearSVC (SVM)**
We implemented a multi-label Support Vector Machine classifier using **sklearn**’s **LinearSVC** in a One-vs-Rest (OvR) configuration. To enhance probabilistic interpretability, each binary classifier was calibrated using **CalibratedClassifierCV** with sigmoid scaling.

The model pipeline starts by converting each paper's abstract and results into a sparse, high-dimensional TF-IDF vector with the following hyperparameters:  
- `max_features = 5000`  
- `ngram_range = (1, 2)`  
- `sublinear_tf = True`  

This setup captures unigram and bigram phrases while reducing the influence of high-frequency terms.

A base LinearSVC classifier with `C = 1.0` and `max_iter = 10000` was wrapped in an OvR strategy to allow independent binary classifiers per label. Each classifier’s output was then passed through a sigmoid calibration layer (`cv='prefit'`) to produce probability scores.

To handle the challenge of multi-label thresholding, we implemented a joint optimization routine using **scipy.optimize.minimize** (method: L-BFGS-B), tuning per-class probability thresholds to maximize a combined F1-micro and F1-macro score. This calibration enables the model to balance precision and recall across imbalanced labels.

This method leverages the interpretability and speed of linear SVMs with the expressive power of TF-IDF features, offering an efficient and scalable approach for text-based multi-label classification.
##### **BERT Embeddings + LinearSVC (SVM)**
We also trained an SVM using dense 768-dimensional BERT embeddings for each paper. These embeddings encapsulate rich contextual semantics from the abstract and result sections, providing a deep representation of research content.

We used **sklearn**’s **LinearSVC** for binary classification per label in a One-vs-Rest scheme, with key hyperparameters set to `C = 1.0` and `max_iter = 10000`. To account for class imbalance, we computed `class_weight='balanced'` individually for each label using **sklearn.utils.class_weight.compute_class_weight**, which was passed into the corresponding LinearSVC instance.

As with the TF-IDF model, each binary classifier was calibrated using **CalibratedClassifierCV** (`method='sigmoid'`, `cv='prefit'`) to produce reliable probability estimates.

Finally, threshold optimization was performed across all 16 labels using the same L-BFGS-B procedure as above, jointly maximizing F1-micro and F1-macro scores.

The combination of BERT’s semantic richness with a linear classifier and optimized per-label thresholds yielded significantly improved multi-label classification performance—particularly in terms of F1 scores—compared to sparse input representations.

#### Multilayer Perceptron
COMPLETE ONCE WE FIND THE MLP JUPYTER NOTEBOOK

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

#### ROC Curves per Class (Naïve Bayes)
The ROC curves for Naïve Bayes highlight its ability to distinguish individual classes. Most classes achieve strong separability with AUC scores above 0.85, such as Class 11 (AUC = 0.96) and Class 6 (AUC = 0.95), indicating high classifier confidence for those labels. However, Class 7 presents a significantly flatter curve (AUC = 0.48), suggesting challenges in distinguishing that label from others. These curves reveal that while Naïve Bayes is effective for certain classes, it may struggle on labels with overlapping features or lower representation in the training set.

#### Macro F1 vs Per-Class Threshold (Naïve Bayes)
The Macro F1 plot evaluates fairness across all classes by weighing each class equally. For Naïve Bayes, we see notable gains at lower thresholds (~0.1–0.3), with the highest Macro F1 occurring around threshold 0.2 for several classes like Class 5 and Class 2. After this range, the score drops steeply, indicating that rare or lower-performing classes contribute less when thresholds are too strict. This underscores the value of per-class tuning to ensure balanced performance, especially in the presence of label imbalance.

#### Impact on Global Micro F1 (Naïve Bayes)
This plot shows how adjusting the classification threshold for each class affects the global Micro F1 score. Notably, Naïve Bayes achieves its peak Micro F1 performance at lower thresholds (~0.1–0.2), especially for classes like Class 5 and Class 10. As thresholds increase, performance consistently drops for nearly all classes, emphasizing the model’s preference for more lenient classification boundaries. This trend suggests that optimizing performance under Naïve Bayes requires favoring recall, particularly for more dominant or frequent classes.

#### ROC Curves per Class (BERT + LinearSVC)
The ROC curves indicate modest separability across most classes, with AUC scores hovering between 0.65 and 0.76 for the majority of labels. For instance, Class 1 (AUC = 0.76) and Class 11 (AUC = 0.75) show relatively strong discrimination, while others like Class 6 (AUC = 0.59) and Class 12 (AUC = 0.57) reflect weaker classifier confidence. Unlike more probabilistic models, the decision boundaries here may be constrained by the hard-margin nature of SVMs, suggesting a need for richer feature representations or additional calibration to improve separation.

#### Macro F1 vs Per-Class Threshold (BERT + LinearSVC)
This plot shows erratic variation in macro F1 across thresholds, which may reflect class imbalance and SVC's sensitivity to threshold adjustments. Peaks occur at varied thresholds for different classes, and sharp fluctuations suggest the model reacts unpredictably to minor changes in class-specific thresholds. These patterns imply that without per-class threshold tuning, the model may over- or under-predict specific classes, especially less frequent ones. Tuning thresholds per label could stabilize performance and offer a more balanced classification strategy.

#### Impact on Global Micro F1 (BERT + LinearSVC)
Similar to the macro F1 curve, the micro F1 plot for BERT + LinearSVC is highly variable across thresholds. Despite a few peaks around 0.2–0.4 for classes like Class 3 and Class 11, the lack of a clear, sustained improvement region reveals that the model may not be consistently leveraging class weights effectively. The unstable performance may stem from class-wise decision boundary sensitivity or the challenge of mapping SVM decision scores to probabilities. Smoothing predictions or integrating Platt scaling could help mitigate this effect.

#### ROC Curves per Class (Deep MLP)
The ROC curves for the Deep MLP model indicate strong class separability, with nearly all AUC values exceeding 0.90. Notably, Class 2 and Class 10 achieve perfect discrimination (AUC = 1.00), while Class 0 and Class 15 also perform extremely well (AUC = 0.99). However, a few classes like Class 9 (AUC = 0.66) and Class 13 (AUC = 0.84) show less confident separation, likely due to overlapping feature distributions or lower representation. Overall, the Deep MLP exhibits robust class-wise confidence, suggesting high representational capacity and learning ability.

#### Macro F1 vs Per-Class Threshold (Deep MLP)
The macro F1 plot demonstrates that performance remains fairly stable across different thresholds, especially between 0.2 and 0.7. Classes such as Class 3 and Class 4 show slightly elevated macro F1 scores above 0.52, but the differences are relatively minor. This stability suggests the Deep MLP handles class imbalance better than more brittle models, and while threshold tuning still matters, it is less volatile than in models like Naïve Bayes or LinearSVC.

#### Impact on Global Micro F1 (Deep MLP)
The micro F1 plot reinforces the stability observed in macro F1. Most classes remain between 0.68 and 0.70 across the full threshold range, with minimal fluctuation. This smooth behavior suggests the Deep MLP is well-calibrated on the frequent classes and does not overly rely on threshold sensitivity to achieve high performance. The model’s generalization is likely driven by its ability to learn nonlinear relationships from the BERT embeddings used as input.

#### Per-Class F1 vs. Threshold (Deep MLP)
This bar plot highlights substantial variation in F1 performance across classes. Class 10 stands out with an F1 score of 1.00, while Class 9 and Class 6 lag behind, scoring below 0.5. These disparities may result from uneven class frequencies or inherent differences in text features among research paper categories. Notably, most classes cluster around the 0.6–0.9 range, reflecting strong per-class precision-recall balance. Future work could investigate boosting performance for underperforming classes via oversampling or data augmentation.

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
### [Full Gantt Chart](https://gtvault-my.sharepoint.com/:x:/r/personal/akumar906_gatech_edu/_layouts/15/Doc.aspx?sourcedoc=%7B28EA5C72-E567-45F9-BB7F-0C027D396FFE%7D&file=ML%20GANTT%20CHART.xlsx&fromShare=true&action=default&mobileredirect=true)

### Contribution Table
Based on the template on the class website, here is the contribution table:

| Name                         | Proposal Contributions                                                          |
| :--------------------------- | :------------------------------------------------------------------------------ |
| Aryeman Singh                | Data Extraction & Pre-processing, Model Training, Gantt Chart, GitHub Pages     |
| Sameer Arora                 | Data Extraction & Pre-processing, Led Report Writing, GitHub Pages                               |
| Naman Goyal                  | Helped with Report Writing, Led Feature Extraction, Model Training, Gantt Chart |
| Lokkit Sanjay Babu Narayanan | Feature Extraction, Model Training                                              |
| Aryika Kumar                 | Feature Extraction, Model Training                                              |


### [GitHub Repository](https://github.gatech.edu/asingh899/ml_project_43/)

### We would like opt-in to be considered for the “Outstanding Project” award.
