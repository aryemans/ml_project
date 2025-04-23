---
layout: page
title: Final Report
permalink: /final
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data. This study focuses on classifying machine learning research abstracts based on the machine learning (ML) model(s)/algorithm(s) used in a research paper. Prior work highlights effective models such as **Logistic Regression**, **Naive Bayes**, and **Random Forest** which, when trained on well-defined textual features provided from pre-processing methods can provide robust classification [1]–[4]. 

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
#### Naive Bayes with TF-IDF Features

We implemented a Naive Bayes classifier using a **One-vs-Rest (OvR)** strategy to handle multi-label classification across 16 research categories. After receiving guidance on providing a baseline of our BERT-based models to show potential enhancement or shortcoming of leveraging semantic understanding via BERT, we decided to implement a Naive Bayes, TF-IDF-based model. 

For input features, we applied **TF-IDF vectorization** to each paper’s abstract and results section after preprocessing. Preprocessing included tokenization, stopword removal, and lemmatization using **NLTK**, followed by transformation using **sklearn**’s `TfidfVectorizer` with the following configuration: `max_features = 175` (to not over-complicate the simple model) and `ngram_range = (1, 1)`.

This setup captures unigram term importance while reducing the feature space for efficiency—so as to make it as close to a simple keyword-based classifier as possible.

The classifier used was **sklearn**’s `MultinomialNB` wrapped in a `OneVsRestClassifier`, with: `alpha = 0.8` (to apply smoothing and prevent zero probabilities).

The model was trained on the TF-IDF features and evaluated using F1-micro and F1-macro metrics. We further refined classification performance using a **threshold optimization** routine. For each class, the probability threshold was tuned in the range `[0.1, 0.9]` to maximize the average of F1-macro scores. This allowed the model to account for class imbalance and better control the decision boundary for each label.

This Naive Bayes approach provides a simple yet effective baseline for multi-label text classification and demonstrates reasonable performance when paired with carefully engineered keyword-based features and threshold tuning.
#### Deep Multilayer Perceptron with BERT Embeddings

We implemented a Deep Multilayer Perceptron (MLP) classifier using PyTorch to perform multi-label classification over 16 potential categories. The model receives 768-dimensional BERT embeddings as input, which capture deep semantic features from each paper’s abstract and result section.

The architecture consists of a feedforward neural network with the following layer structure:
- Input layer: 768 units  
- Hidden layers: [768 → 512 → 256] with **Layer Normalization**, **ReLU** activations, and **Dropout (0.2)** between each layer  
- Output layer: 16 units (one per class), with raw logits passed through a **sigmoid** during inference to produce probabilities

The model was trained using:
- **Loss function**: `BCEWithLogitsLoss` (binary cross-entropy for multi-label classification)
- **Optimizer**: `Adam` with learning rate `lr = 0.001`
- **Epochs**: 50
- **Batch size**: 32

Training was conducted using the BERT feature vectors, and evaluation was performed on a held-out test set. Predictions were binarized using a global threshold of 0.2 to optimize F1 scores.

This MLP model demonstrates strong performance in capturing complex, non-linear relationships in high-dimensional semantic space, making it an effective deep learning alternative to linear classifiers we previously implemented.
## Results and Discussion
### Quantitative Metrics
#### Logistic Regression (TF-IDF + BERT)

| Metric               | TF-IDF Logistic Regression | BERT Logistic Regression |
| -------------------- | -------------------------- | ------------------------ |
| Exact Match Accuracy | **0.4855**                 | **0.5942**               |
| Hamming Loss         | **0.0457**                 | **0.0371**               |
| F1 Score (Micro)     | **0.6645**                 | **0.7285**               |
| F1 Score (Macro)     | **0.4795**                 | **0.5546**               |

These scores indicate a notable improvement in model performance after filtering. The BERT-based Logistic Regression model achieved an **Exact Match Accuracy** of **59.4%**, meaning that nearly 60% of the test samples had all predicted labels exactly correct. The **Hamming Loss**—which measures the fraction of incorrect labels—was reduced to just **3.7%**, indicating high reliability across individual label predictions.

The **F1 Score (Micro)** for BERT was **72.9%**, showing strong overall precision and recall when treating each (sample, label) pair equally. The **F1 Score (Macro)** reached **55.5%**, suggesting decent performance even across less frequent categories, though there’s still room for improvement in handling class imbalance.

In comparison, the TF-IDF-based model performed moderately, with an **Exact Match Accuracy** of **48.6%** and a **Hamming Loss** of **4.6%**. Its **Micro F1** of **66.5%** and **Macro F1** of **47.9%** reflect good baseline performance but lag behind the BERT-based model, especially in capturing nuanced or rare labels.

Overall, the BERT-based Logistic Regression model consistently outperforms the TF-IDF baseline across all metrics, particularly in holistic correctness (Exact Match Accuracy) and balanced class representation (Macro F1).

#### SVM (TF-IDF + BERT)

| Metric           | TF-IDF SVM | BERT SVM |
| -------------------- | -------------- | ------------ |
| Exact Match Accuracy | **0.2246**     | **0.3913**   |
| Hamming Loss         | **0.0553**     | **0.0489**   |
| F1 Score (Micro)     | **0.3441**     | **0.5537**   |
| F1 Score (Macro)     | **0.2331**     | **0.4062**   |

The SVM-based classification results demonstrate a clear performance gain when using BERT embeddings over TF-IDF features. The **BERT SVM** model attained an **Exact Match Accuracy** of **39.1%**, meaning nearly 4 out of 10 samples had all labels predicted correctly. This is a significant improvement over the **TF-IDF SVM**, which achieved only **22.5%** on the same metric, suggesting that BERT embeddings offer better semantic understanding for multi-label classification.

In terms of **Hamming Loss**, the BERT model also performed slightly better, with only **4.9%** of label predictions being incorrect—compared to **5.5%** for TF-IDF. This suggests a more consistent label-wise prediction accuracy with BERT.

The **F1 Score (Micro)** for BERT was **55.4%**, indicating solid overall performance in identifying correct label assignments across all samples. In contrast, the TF-IDF model struggled here with a lower Micro F1 of **34.4%**, showing difficulty in making precise predictions on a per-label basis.

Lastly, **F1 Score (Macro)**—which evaluates the model’s ability to handle class imbalance by treating each class equally—further illustrates the strength of the BERT model, with **40.6%** versus **23.3%** from the TF-IDF baseline. This metric shows that the BERT SVM model is better at predicting not just common labels but also the rarer ones.

The **BERT-based SVM** provides a substantial improvement across all evaluation metrics, highlighting the benefit of leveraging contextual semantic embeddings over sparse term frequency features in multi-label classification tasks.

#### Naive Bayes (TF-IDF)

| Metric           | TF-IDF Naive Bayes |
| -------------------- | ---------------------- |
| Exact Match Accuracy | **0.3551**             |
| Hamming Loss         | **0.0562**             |
| F1 Score (Micro)     | **0.6101**             |
| F1 Score (Macro)     | **0.3845**             |

The TF-IDF-based Naive Bayes classifier demonstrates moderate performance on the multi-label classification task. The **Exact Match Accuracy** of **35.5%** indicates that just over a third of the test samples had all labels predicted correctly. While this score is lower than some more complex models, it reflects a reasonable outcome given the simplicity and efficiency of Naive Bayes.

The **Hamming Loss** was **5.6%**, meaning that a relatively small portion of individual label predictions were incorrect. This suggests a decent level of consistency across label-wise decisions, even if complete multi-label accuracy per sample remains limited.

The **F1 Score (Micro)** was **61.0%**, showing that the model performs well in aggregate across all (sample, label) pairs. This score balances precision and recall across the entire dataset and confirms the model’s general reliability in predicting frequently occurring labels.

However, the **F1 Score (Macro)** was **38.5%**, which reflects lower performance on rare or underrepresented classes. Since macro F1 treats all classes equally, this score reveals the model’s struggle with imbalanced datasets and its limited ability to identify minority labels effectively.

While the TF-IDF-based Naive Bayes classifier may not match the performance of deep or embedding-based models, it still provides a lightweight and interpretable baseline with competitive results for common labels—representing how a key-word based classifier may in this task.

#### Deep Multilayer Perceptron (BERT)

| Metric           | BERT Deep MLP |
| -------------------- | ----------------- |
| Exact Match Accuracy | **0.5797**        |
| Hamming Loss         | **0.0448**        |
| F1 Score (Micro)     | **0.6754**        |
| F1 Score (Macro)     | **0.7639**        |

These results show strong performance from the BERT-based Deep Multilayer Perceptron (MLP) model in the multi-label classification task. The model achieved an **Exact Match Accuracy** of **57.97%**, indicating that nearly 6 out of 10 samples had all labels predicted correctly. This reflects the model’s ability to understand and predict the full set of labels with high precision.

The **Hamming Loss** was **4.48%**, suggesting that only a small fraction of label assignments were incorrect. This low loss reflects the model’s consistent performance across individual label decisions.

The **F1 Score (Micro)** was **67.54%**, which measures the model’s precision and recall across all label/sample pairs. This shows that, in aggregate, the model makes accurate label predictions and is especially strong in handling the dominant classes in the dataset.

The standout metric here is the **F1 Score (Macro)** of **76.39%**, which treats each class equally regardless of frequency. This result is particularly impressive, as it implies the model performs very well even on rare or underrepresented labels—a common challenge in multi-label classification.

Ultimately, the BERT-based Deep MLP model demonstrates excellent generalization and balanced label prediction. Its performance across all metrics makes it a robust choice for scenarios where both common and rare classes must be captured reliably and is the strongest model we trained.
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
#### Logistic Regression
#### ROC Curves per Class (Logistic Regression)
![Label Frequency](/assets/roc.png)
Helps assess class separability. AUC values close to 1.0 (like for "Q-Learning") show strong classifier confidence. In contrast, flatter curves indicate ambiguous or overlapping feature distributions. This helps identify which labels the model finds easiest or hardest to distinguish based on BERT embeddings.
#### Macro F1 vs Per-Class Threshold (Logistic Regression)
![Label Frequency](/assets/macro-f1-per-class.png)
Since macro F1 weighs all classes equally, this visualization reveals how rare or poorly performing labels influence fairness across the label space. We observed that tuning thresholds for low-frequency classes (like "Object Detection") causes sharper shifts in macro F1 than common ones — helpful for diagnosing imbalance.
#### Impact on Global Micro F1 (Logistic Regression)
![Label Frequency](/assets/impact-micro-F1.png)
This plot indicates which classes have a disproportionate affect on the overall model performance (the previous visualization illuminates the thresholds we may tune—this visualization portrays the impact of tuning each one of those thresholds). This impact may potentially be due to a class' frequency or correlation with other labels. 


#### Naive Bayes
#### ROC Curves per Class (Naive Bayes)
![Plot](/assets/nb-roc.png)
The ROC curves for Naive Bayes highlight its ability to distinguish individual classes. Most classes achieve strong separability with AUC scores above 0.85, such as "SVM (AUC = 0.96)" and Logistic "Regression (AUC = 0.95)', indicating high classifier confidence for those labels. However, "Object Detection" presents a significantly flatter curve (AUC = 0.48), suggesting challenges in distinguishing that label from others. These curves reveal that while Naive Bayes is effective for certain classes, it may struggle on labels with overlapping features or lower representation in the training set.
#### Macro F1 vs Per-Class Threshold (Naive Bayes)
![Plot](/assets/nb-macro.png)
The Macro F1 plot evaluates fairness across all classes by weighing each class equally. For Naive Bayes, we see notable gains at lower thresholds (~0.1–0.3), with the highest Macro F1 occurring around threshold 0.2 for several classes like Autoencoder-based and GAN-based. After this range, the score drops steeply, indicating that rare or lower-performing classes contribute less when thresholds are too strict. This underscores the value of per-class tuning to ensure balanced performance, especially in the presence of label imbalance.
#### Impact on Global Micro F1 (Naive Bayes)
![Plot](/assets/nb-impact.png)
This plot shows how adjusting the classification threshold for each class affects the global Micro F1 score. Notably, Naive Bayes achieves its peak Micro F1 performance at lower thresholds (~0.1–0.2), especially for classes like Logistic Regression and Autoencoder-based. As thresholds increase, performance consistently drops for nearly all classes, emphasizing the model’s preference for more lenient classification boundaries. This trend suggests that optimizing performance under Naive Bayes requires favoring recall, particularly for more dominant or frequent classes.

#### Support Vector Machine (SVM)
#### ROC Curves per Class (BERT SVM)
![Plot](/assets/svm-roc.png)
The ROC curves indicate modest separability across most classes, with AUC scores hovering between 0.65 and 0.76 for the majority of labels. For instance, "CNN-based (AUC = 0.76)" and "SVM (AUC = 0.75)" show relatively strong discrimination, while others like "Logistic Regression (AUC = 0.59)" and "Transformer-based (AUC = 0.57)" reflect weaker classifier confidence. Unlike more probabilistic models, the decision boundaries here may be constrained by the hard-margin nature of SVMs, suggesting a need for richer feature representations or additional calibration to improve separation (fine-tuning may be a technique to accomplish this).
#### Macro F1 vs Per-Class Threshold (BERT SVM)
![Plot](/assets/svm-macro.png)
This plot shows erratic variation in macro F1 across thresholds, which may reflect class imbalance and SVC's sensitivity to threshold adjustments. Peaks occur at varied thresholds for different classes, and sharp fluctuations suggest the model reacts unpredictably to minor changes in class-specific thresholds. These patterns imply that without per-class threshold tuning, the model may over- or under-predict specific classes, especially less frequent ones. 
#### Impact on Global Micro F1 (BERT SVM)
![Plot](/assets/svm-impact.png)
Similar to the macro F1 curve, the micro F1 plot for BERT SVM is highly variable across thresholds. Despite a few peaks around 0.2–0.4 for classes like Graph-based and Object Detection, the lack of a clear, sustained improvement region reveals that the model may not be consistently leveraging class weights effectively. The unstable performance may stem from class-wise decision boundary sensitivity or the challenge of mapping SVM decision scores to probabilities. 

#### Deep MLP
#### ROC Curves per Class (Deep MLP)
![Plot](/assets/mlp-roc.png)
The ROC curves for the Deep MLP model indicate strong class separability, with nearly all AUC values exceeding 0.90. Notably, Class 2 and Class 10 achieve perfect discrimination (AUC = 1.00), while Autoencoder-based and Gan-based also perform extremely well (AUC = 0.99). However, a few classes, such as SVM (AUC = 0.83) show less confident separation, likely due to overlapping feature distributions or lower representation. Overall, the Deep MLP exhibits robust class-wise confidence, suggesting high representational capacity and learning ability.
#### Macro F1 vs Per-Class Threshold (Deep MLP)
![Plot](/assets/mlp-macro.png)
The macro F1 plot demonstrates that performance remains fairly stable across different thresholds, especially between 0.2 and 0.7. Classes such as Gaussian Process and Graph-based show slightly elevated macro F1 scores above 0.52, but the differences are relatively minor. This stability suggests the Deep MLP handles class imbalance better than more brittle models, and while threshold tuning still matters, it is less volatile than in models like Naive Bayes or SVM.
#### Impact on Global Micro F1 (Deep MLP)
![Plot](/assets/mlp-impact.png)
The micro F1 plot reinforces the stability observed in macro F1. Most classes remain between 0.68 and 0.70 across the full threshold range, with minimal fluctuation. This smooth behavior suggests the Deep MLP is well-calibrated on the frequent classes and does not overly rely on threshold sensitivity to achieve high performance. The model’s generalization is likely driven by its ability to learn nonlinear relationships from the BERT embeddings used as input.
### Analysis of Algorithm(s)/Model(s)
#### Logistic Regression
Logistic Regression demonstrated consistent and interpretable performance, particularly when paired with BERT embeddings. The TF-IDF variant achieved moderate scores (Exact Match Accuracy: 48.6%, Micro F1: 66.5%), reflecting its reliance on surface-level token frequency. While sufficient for common labels, this model struggled to generalize across imbalanced classes, as evidenced by its lower Macro F1 score (47.9%).

In contrast, the BERT-based Logistic Regression model significantly outperformed its sparse counterpart. With an Exact Match Accuracy of 59.4% and a Micro F1 of 72.9%, it leveraged the deep semantic signals captured by contextual embeddings to better identify both frequent and rare labels. The relatively high Macro F1 (55.5%) further indicates improved fairness across classes. This result reinforces the value of semantic representation, even within linear classifiers.

The combination of calibrated thresholds and semantic input enabled this model to strike a balance between interpretability and expressiveness, making it one of the strongest linear baselines.
#### Support Vector Machine (SVM)
The SVM models exhibited mixed results, with substantial improvements observed when transitioning from TF-IDF to BERT embeddings. The TF-IDF-based SVM yielded the weakest overall performance (Exact Match Accuracy: 22.5%, Micro F1: 34.4%), revealing the limitations of sparse, high-dimensional representations and rigid decision boundaries in handling nuanced textual data.

By contrast, the BERT-augmented SVM showed notable improvements across all metrics—Exact Match Accuracy rose to 39.1%, and Micro F1 climbed to 55.4%. However, despite the richer input, the model’s performance remained volatile. Both Macro F1 (40.6%) and threshold sensitivity visualizations suggested unstable generalization across infrequent labels. These shortcomings likely stem from the SVM’s hard-margin nature, which does not naturally lend itself to probabilistic calibration or flexible multi-label separation.

Therefore, while the BERT SVM benefits from better features, it falls short of the consistency observed in the MLP or Logistic Regression due to its architectural rigidity.
#### Naive Bayes
The Naive Bayes model served as an efficient and interpretable baseline. With TF-IDF input restricted to unigrams and a limited feature space (max_features=175), the model achieved modest results (Micro F1: 61.0%, Macro F1: 38.5%). These outcomes illustrate its strength in capturing dominant label patterns while struggling with sparse classes, as reflected in its comparatively low Macro F1.

The ROC curve analysis revealed high separability for well-defined classes (e.g., “SVM,” “Logistic Regression”), but poor confidence on labels with overlapping or ambiguous terms (e.g., “Object Detection”). Its performance was also highly sensitive to threshold tuning, further affirming its limited robustness in a complex, imbalanced multi-label setting.

Despite its simplicity, the model highlighted the challenge of relying purely on lexical indicators (like keywords), reinforcing the need for semantic augmentation in future iterations.

#### Deep Multilayer Perceptron
The BERT-based Deep MLP was the most performant model across all metrics. With an Exact Match Accuracy of 57.97%, Micro F1 of 67.54%, and a standout Macro F1 of 76.39%, this model demonstrated the strongest capacity to generalize across both frequent and rare classes.

The architecture’s depth, coupled with nonlinear activations and dropout regularization, enabled it to model complex interactions within the high-dimensional BERT embeddings. The remarkably high Macro F1 score signifies its strength in treating all classes equitably, a vital trait in imbalanced multi-label tasks.

Moreover, visualizations of F1 stability across thresholds emphasized the MLP’s calibration and resilience. Its ability to maintain consistent performance without aggressive tuning sets it apart as the most adaptable and generalizable model in the study.

### Comparison of Models
![Model Comparison Plot](/assets/model-comparison.png)
The performance comparison across models underscores the central takeaway of this study: contextual embeddings derived from transformer-based models substantially outperform traditional keyword-based approaches in the multi-label classification of ML research papers. As shown in our model comparison chart, BERT-based models demonstrate clear superiority across all four evaluation metrics—**Exact Match Accuracy**, **Hamming Loss**, **F1 Score (Micro)**, and **F1 Score (Macro)**—compared to their TF-IDF-based counterparts.

Among the tested models, the **Deep Multilayer Perceptron (MLP)** leveraging **BERT embeddings** achieved the best overall results. It attained the highest Exact Match Accuracy (0.5797), the lowest Hamming Loss (0.0448), and the strongest F1 scores—particularly a Macro F1 of 0.7639, indicating its ability to generalize across both common and rare classes. This strong performance reflects the model’s ability to learn complex, nonlinear relationships from semantically rich 768-dimensional BERT embeddings extracted from each paper's abstract and results sections.

The **Logistic Regression (BERT)** model also performed well, ranking second across all metrics. Despite its linear nature, it benefited significantly from the semantic depth captured by the BERT embeddings (depicted in the "HDBSCAN Clustering on UMAP Embeddings" visualization above), achieving an Exact Match Accuracy of 0.5942 and a Micro F1 score of 0.7285. Its strong performance reinforces the idea that high-quality semantic feature representations can make even simple models highly effective.

In contrast, models using TF-IDF features underperformed. The **SVM (TF-IDF)** model achieved the lowest overall scores, including a Micro F1 of 0.3441 and Macro F1 of just 0.2331, highlighting its struggle to generalize due to the sparsity and lack of contextual information in TF-IDF representations. Even the **Naive Bayes (TF-IDF)** model, though simpler, outperformed SVM on most metrics—suggesting that its probabilistic assumptions were somewhat better suited to the task, but still limited by the expressiveness of its input features.

These results validate the use of **semantic embedding techniques** in scientific document classification. By incorporating BERT-based embeddings and structuring the classification task around the abstract and results sections, our models were able to better capture the methodological essence of research papers. These findings emphasize the importance of leveraging contextualized representations and more expressive model architectures in automated literature analysis tasks.

#### Data Poisoning (Comparison of TF-IDF vs. BERT)


### Next Steps
A key limitation of our current pipeline is the lack of **fine-tuning** on the BERT model. While the pre-trained embeddings provided strong performance, they were generated without adaptation to our specific task. As a result, semantically similar ML terms (e.g., "Transformer-based" vs. "CNN-based") may lie close in BERT’s vector space, making them harder to distinguish in a classification setting. Fine-tuning the transformer on our labeled dataset would likely improve label separation and classification accuracy by aligning the embedding space more directly with our target labels.

However, full fine-tuning would require significant time and computational resources, including training infrastructure, layer freezing strategies, and careful hyperparameter tuning—making it infeasible within our project scope.

In future work, we recommend incorporating task-specific fine-tuning of BERT, or lightweight alternatives like adapter layers or prompt tuning, to further leverage the power of contextual embeddings. Additionally, expanding the dataset would help mitigate class imbalance and improve generalization—particularly for rare or low-frequency labels, which consistently underperformed across models.

Together, fine-tuning and increased data would allow for better label discrimination, improved calibration, and stronger performance in multi-label classification of scientific text.

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

| Name                         | Proposal Contributions                                                                                                  |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| Aryeman Singh                | Data Extraction & Pre-processing, Model Training, Gantt Chart, GitHub Pages                                             |
| Sameer Arora                 | Data Extraction & Pre-processing, Led Report Writing, Led Results Analysis, Led Comparison of the Models, GitHub Pages  |
| Naman Goyal                  | Helped with Report Writing, Led Feature Extraction, Model Training, Gantt Chart Organization, Exploratory Data Analysis |
| Lokkit Sanjay Babu Narayanan | Feature Extraction, Model Training, Visualization Production / Results Analysis                                         |
| Aryika Kumar                 | Feature Extraction, Model Training, Visualization Production / Results Analysis                                         |


### [GitHub Repository](https://github.gatech.edu/asingh899/ml_project_43/)

### We would like opt-in to be considered for the “Outstanding Project” award.
