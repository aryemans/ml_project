---
layout: page
title: Proposal
permalink: /proposal
---

## Proposal Sections & Checklist

### 1. Introduction/Background

Text classification is a supervised learning task to categorize textual data, with applications across various domains. This study focuses on classifying machine learning research abstracts based on research area and methodology. Prior work highlights effective models such as Support Vector Machines, Decision Trees, and Naive Bayes. These models, when trained on well-defined textual features, can provide robust classification [2]. 

Preprocessing techniques like stemming, term frequency-inverse document frequency, and bag-of-words play a crucial role in extracting relevant features [1]. This study will evaluate traditional machine learning models to determine which yields the highest accuracy in classifying abstracts. The dataset will consist of labeled machine learning research abstracts, sourced from [**Papers with Code**](https://paperswithcode.com/). A preprocessing pipeline will be implemented to extract key textual features, ensuring high-quality input for model training and evaluation. Ultimately, this research aims to identify the most effective approach for text classification in this context.
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

- **Metrics**
We employ multi-class classification to evaluate model performance. The macro-F1 score averages the F1 score for each class, ensuring our results are precise in cases of imbalanced datasets. The weighted F1 score accounts for the imbalance of each class and weights them accordingly, providing a more realistic metric. A confusion matrix reveals misclassifications patterns and biases towards majority classes. Additionally, we will use precision which indicates how often a model is correct, accuracy which is overall correctness, and recall. For regression metrics, we will use R-squared which measures explained variance and mean square error which tells us the prediction error. 

- **Project Goals** 
Our goal is to identify which model performs the best for classifying the introductions and conclusions by assessing performance, sustainability, and ethical consequences. We will measure performance using our metrics and compare computational costs across different models. Ethical concerns include minimizing bias to prevent overrepresentation of dominant research fields.

- **Expected Results**
We expect logistic regression to perform the best in accuracy and F1 Score due to its suitability for high dimensional text classification. Naive Bayes will offer strong results for smaller text but might struggle with complexity and will likely be the most resource-efficient. Decision Trees may overfit but offers interpretability.  

---

### 5. References
[1] C. C. Aggarwal and C. Zhai, “A survey of text classification algorithms,” Mining Text Data, pp. 163–222, 2012. doi:10.1007/978-1-4614-3223-4_6

[2] I. Dawar, N. Kumar, S. Negi, S. Pathan, and S. Layek, “Text categorization using supervised machine learning techniques,” 2023 Sixth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU), Mar. 2023. doi:10.1109/wids-psu57071.2023.00046 

[3] K. Shyrokykh, M. Girnyk, and L. Dellmuth, “Short text classification with machine learning in the Social Sciences: The case of climate change on Twitter,” PLOS ONE, vol. 18, no. 9, Sep. 2023. doi:10.1371/journal.pone.0290762 

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

