---
layout: page
title: Proposal
permalink: /proposal
---
## Introduction/Background

Text classification is a supervised learning task to categorize textual data, with applications across various domains. This study focuses on classifying machine learning research abstracts based on research area and methodology. Prior work highlights effective models such as Support Vector Machines, Decision Trees, and Naïve Bayes, which, when trained on well-defined textual features provided from pre-processing methods such as stemming, TF-IDF, and bag-of-words can provide robust classification [1]–[4]. 

To evaluate classification accuracy, this study will test machine learning models on **Introduction** and **Conclusion** sections of research papers sourced from [**Papers with Code**](https://paperswithcode.com/). These sections often summarize the key contributions and methodology of a paper, making them a suitable dataset for classification. The objective is to determine the most effective approach for categorizing research papers based on content rather than simple keyword matching.

## Problem Definition
As the world of machine learning focused research expands, gathering relevant literature for review within a specific domain gets increasingly difficult. Modern keyword matching techniques do not accurately capture deeper semantic meaning and context. As a result, a structured approach to classify machine learning research papers and provide researchers with quick access to relevant papers and efficiently analyze their domain. 

## Methods
### Data-Pre-Processing Methods
To improve classification accuracy, the following preprocessing steps will be applied using **NLTK** and **scikit-learn**. 

Text normalization and tokenization refine the text by lowercasing words, expanding contractions, and splitting text into meaningful units. Lemmatization reduces words to their base forms, and stopword removal eliminates non-informative words [2]. This will be done using NLTK's **word_tokenize()**, **WordNetLemmatizer**, and **stopwords** module. 

Feature extraction techniques such as **bag-of-words** and **TF-IDF** with **sci-kit learn's TfidfVectorizer** help quantify the importance of words. While term frequency gives greater weight to common words, IDF penalizes frequently occurring terms, highlighting specific terminology relevant to research papers [1]. 

To improve semantic understanding, **BERT-based embeddings** via **Hugging Face's transformers library** generate dense vector representations, capturing contextual meaning beyond word frequency [1]. 

### Machine Learning Models
Once high-dimensional representations of text are generated, the following machine learning models will be tested for classification:

- **Naïve Bayes (NB)**: A probabilistic classifier using **MultinomialNB from scikit-learn**, known for its scalability and efficiency [3], [4].
    
- **Support Vector Machines (SVM)**: Uses **SVC from scikit-learn**, leveraging hyperplanes to separate text classes efficiently [1], [4].
    
- **Random Forest (RF)**: An ensemble method using **RandomForestClassifier from scikit-learn**, which reduces overfitting by averaging multiple decision trees [1], [4].

### 4. (Potential) Results and Discussion

- **Metrics**
We employ multi-class classification to evaluate model performance. The macro-F1 score averages the F1 score for each class, ensuring our results are precise in cases of imbalanced datasets. The weighted F1 score accounts for the imbalance of each class and weights them accordingly, providing a more realistic metric. A confusion matrix reveals misclassifications patterns and biases towards majority classes. Additionally, we will use precision which indicates how often a model is correct, accuracy which is overall correctness, and recall. For regression metrics, we will use R-squared which measures explained variance and mean square error which tells us the prediction error. 

- **Project Goals** 
Our goal is to identify which model performs the best for classifying the introductions and conclusions by assessing performance, sustainability, and ethical consequences. We will measure performance using our metrics and compare computational costs across different models. Ethical concerns include minimizing bias to prevent overrepresentation of dominant research fields.

- **Expected Results**
We expect logistic regression to perform the best in accuracy and F1 Score due to its suitability for high dimensional text classification. Naive Bayes will offer strong results for smaller text but might struggle with complexity and will likely be the most resource-efficient. Decision Trees may overfit but offers interpretability.  

---

### 5. References
[1] A. Gasparetto, M. Marcuzzo, A. Zangari, and A. Albarelli, “A survey on text classification algorithms: From text to predictions,” Information, vol. 13, no. 2, Feb. 2022. doi:10.3390/info13020083

[2] C. C. Aggarwal and C. Zhai, “A survey of text classification algorithms,” Mining Text Data, pp. 163–222, 2012. doi:10.1007/978-1-4614-3223-4_6

[3] I. Dawar, N. Kumar, S. Negi, S. Pathan, and S. Layek, “Text categorization using supervised machine learning techniques,” 2023 Sixth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU), Mar. 2023. doi:10.1109/wids-psu57071.2023.00046

[4] K. Shyrokykh, M. Girnyk, and L. Dellmuth, “Short text classification with machine learning in the Social Sciences: The case of climate change on Twitter,” PLOS ONE, vol. 18, no. 9, Sep. 2023. doi:10.1371/journal.pone.0290762

---   
- **Gantt Chart**: 
[**Full Gantt Chart**](https://gtvault-my.sharepoint.com/:x:/g/personal/akumar906_gatech_edu/EXJc6ihn5flFu38MAn05b_4BXmzhr109P-YNltiuoURhIg?e=5WeXAZ)

<img width="338" alt="Screenshot 2025-02-21 at 5 06 18 PM" src="https://github.com/user-attachments/assets/ed6d20b4-1640-4a3d-be48-236dd2529874" />


- **Contribution Table**: list all group members’ names and explicit contributions in preparing the proposal using the format below.

     | Name    | Proposal Contributions |
     |---------|------------------------|
     | Naman Goyal | Video Recording, Video Creation |
     | Aryeman Singh | Github Pages, Problem Definition, Motivation |
     | Sameer Arora  | References, Potential Dataset |
     | Aryika Kumar | Gantt Chart, Results, Presentation |
     | Sanjay Lokkit Babu Narayanan | Methods, Presentation |


2. **Video Presentation**  
   - 

3. **GitHub Repository**  
   - 

