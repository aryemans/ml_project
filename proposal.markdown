---
layout: page
title: Proposal
permalink: /proposal
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data. This study focuses on classifying machine learning research abstracts based on research area and methodology. Prior work highlights effective models such as **Support Vector Machines**, **Decision Trees**, and **Naïve Bayes**, which, when trained on well-defined textual features provided from pre-processing methods such as TF-IDF and bag-of-words can provide robust classification [1]–[4]. 

To evaluate classification accuracy, this study will test machine learning models on **Introduction** and **Conclusion** sections of research papers sourced from [**Papers with Code**](https://paperswithcode.com/). These sections often summarize the key contributions and methodology of a paper, making them a suitable dataset for classification. The objective is to determine the most effective approach for categorizing research papers based on content rather than simple keyword matching.

## Problem Definition
As the world of machine learning focused research expands, gathering relevant literature for review within a specific domain gets increasingly difficult. Modern keyword matching techniques do not accurately capture deeper semantic meaning and context. As a result, a structured approach to classify machine learning research papers and provide researchers with quick access to relevant papers and efficiently analyze their domain. 

## Methods
### Data-Pre-Processing Methods
We will be labeling a set of research papers from [**Papers with Code**](https://paperswithcode.com/) and splitting our data into 80% for training and 20% for metric evaluation. 

To improve classification accuracy, the following preprocessing steps will be applied using **NLTK** and **scikit-learn**. 

Text normalization and tokenization refine the text by lowercasing words, expanding contractions, and splitting text into meaningful units. Lemmatization reduces words to their base forms, and stopword removal eliminates non-informative words [2]. This will be done using NLTK's **word_tokenize()**, **WordNetLemmatizer**, and **stopwords** module. 

Feature extraction techniques such as **bag-of-words** and **TF-IDF** with **sci-kit learn's TfidfVectorizer** help quantify the importance of words. While term frequency gives greater weight to common words, IDF penalizes frequently occurring terms, highlighting specific terminology relevant to research papers [1]. 

To improve semantic understanding, **BERT-based embeddings** via **Hugging Face's transformers library** generate dense vector representations, capturing contextual meaning beyond word frequency [1]. 

### Classification Models
Once high-dimensional representations of text are generated, the following machine learning models will be tested for classification.

Naïve Bayes (NB), implemented via **MultinomialNB** in **scikit-learn**, is a probabilistic classifier that assumes feature independence, enabling efficient training and scalability [3], [4]. 

Support Vector Machines (SVM), using **SVC** from **scikit-learn**, construct hyperplanes for binary classification and leverage kernel functions to handle high-dimensional spaces effectively [1], [4]. 

Random Forest, implemented with **RandomForestClassifier** in **scikit-learn**, aggregates multiple decision trees for classification but requires careful tuning to balance computation time and overfitting risk [1], [4].

## Results and Discussion
#### Metrics
To assess model performance, we will use accuracy, precision, recall, and F1-score. Accuracy measures the proportion of correct predictions among total predictions. Precision evaluates how many of the predicted **positive** classifications were actually correct. Recall measures the proportion of actual positives correctly classified. F1-score balances precision and recall, particularly useful when handling imbalanced datasets.

#### Project Goals
We hope to identify which classification models are the most accurate and have the highest F-1 score. The project also considers sustainability and ethical considerations, aiming to find a computationally efficient model that minimizes mis-categorization and overrepresentation of dominant research fields. 

#### Expected Results
We expect SVM to perform the best in accuracy and F1 Score due to its suitability for high dimensional text classification [3]. Naïve Bayes will offer strong results for smaller text but might struggle with complexity and will likely be the most resource-efficient [4]. Random Forest may overfit but offers interpretability and potential for high accuracy [1].

## References
[1] A. Gasparetto, M. Marcuzzo, A. Zangari, and A. Albarelli, “A survey on text classification algorithms: From text to predictions,” _Information_, vol. 13, no. 2, Feb. 2022. doi:10.3390/info13020083

[2] C. C. Aggarwal and C. Zhai, “A survey of text classification algorithms,” _Mining Text Data_, pp. 163–222, 2012. doi:10.1007/978-1-4614-3223-4_6

[3] I. Dawar, N. Kumar, S. Negi, S. Pathan, and S. Layek, “Text categorization using supervised machine learning techniques,” _2023 Sixth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU)_, Mar. 2023. doi:10.1109/wids-psu57071.2023.00046

[4] K. Shyrokykh, M. Girnyk, and L. Dellmuth, “Short text classification with machine learning in the Social Sciences: The case of climate change on Twitter,” _PLOS ONE_, vol. 18, no. 9, Sep. 2023. doi:10.1371/journal.pone.0290762

---   
## Other
### Gantt Chart
[**Full Gantt Chart**](https://gtvault-my.sharepoint.com/:x:/g/personal/akumar906_gatech_edu/EXJc6ihn5flFu38MAn05b_4BXmzhr109P-YNltiuoURhIg?e=5WeXAZ)

<img width="400" alt="Screenshot 2025-02-21 at 5 06 18 PM" src="https://github.com/user-attachments/assets/ed6d20b4-1640-4a3d-be48-236dd2529874" />


### Contribution Table
Based on the template on the class website, here is the contribution table:

| Name                         | Proposal Contributions                        |
| :--------------------------- | :-------------------------------------------- |
| Aryeman Singh                | GitHub Pages, Problem Definition, Motivation  |
| Sameer Arora                 | Introduction, Methods, Discussion, References |
| Naman Goyal                  | Video Recording, Video Creation               |
| Lokkit Sanjay Babu Narayanan | Methods, Presentation, Potential Dataset      |
| Aryika Kumar                 | Gantt Chart, Results, Presentation            |


### [Video Presentation](https://youtu.be/p5pqXeXo_5k)

### [GitHub Repository](https://github.gatech.edu/asingh899/ml_project_43/)

### We would like opt-in to be considered for the “Outstanding Project” award.
