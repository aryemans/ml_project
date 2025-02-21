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
Identify several quantitative metrics you plan to use for the project (i.e. ML Metrics). Present goals in terms of these metrics, and state any expected results.

- **Metrics**
In this project, we are utilizing multi-class classification as we want to evaluate how well different models perform using quantitative metrics. First, we have the macro-F1 score which will calculate the F1-score for each class and take the average, ultimately measuring how well our classification model performs across all classes. This will be especially helpful since our dataset may be imbalanced and the macro-F1 score will treat every class equally, regardless of the number of datapoints, ensuring the precision of our results. Next, we will utilize a weighted F1 score which will take into account the imbalance of each class and weighting them accordingly. This is a more realistic metric and will tell us how our model performs on larger classes. We will also have a confusion matrix as this will help us see common misclassifications patterns and whether our model is biased towards majority classes. In addition, we will utilize precision, recall, and accuracy. Precision indicates when a model predicts, how often it is correct. Accuracy gives us the proportion of correct prediction among all predictions. Finally, we will use regression metrics such as R-squared which tells us the proportion of variance in the dependent variable explained by the model and mean square error which indicates gives us the average of the squared differences between predicted vs actual values. 

- **Project Goals** 
The goals of our project are to identify which model performs the best for classifying research papers abstracts. We want to assess performance, sustainability, and consider the ethical consequences to fully interpret our results. We will measure performance using our metrics and evaluate trade offs between models. We will consider sustainability and efficiency by evaluating the computational costs that come with using each model and compare them with lighter models to see if we can lower resource usage. Finally, we have to take into account the ethicality of performing an experiment like this. We need to make sure that there is little to no bias in our data and our models don’t overrepresent dominant fields in research. 

- **Expected Results**
We expect logistic regression to perform the best overall in terms of accuracy and F1 Score. This is because logistic regression is suited to high dimensional text classification problems. Naive Bayes will most likely offer strong results for smaller abstracts, but might struggle with complexity. However, we expect this to be the most efficient model and will require less training and resources. Decision Trees may overfit especially if our data is high dimensional, but they will also provide the most interpretable decision making process.  

---

### 5. References
[1] C. C. Aggarwal and C. Zhai, “A survey of text classification algorithms,” Mining Text Data, pp. 163–222, 2012. doi:10.1007/978-1-4614-3223-4_6

[2] I. Dawar, N. Kumar, S. Negi, S. Pathan, and S. Layek, “Text categorization using supervised machine learning techniques,” 2023 Sixth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU), Mar. 2023. doi:10.1109/wids-psu57071.2023.00046 

[3] K. Shyrokykh, M. Girnyk, and L. Dellmuth, “Short text classification with machine learning in the Social Sciences: The case of climate change on Twitter,” PLOS ONE, vol. 18, no. 9, Sep. 2023. doi:10.1371/journal.pone.0290762 

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

