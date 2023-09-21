# Table of Contents
- [Project Title](#project-title)
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)
- [Results and Findings](#results-findings)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Duplicate Values](#duplicate-values)
    - [Missing Values](#missing-values)  
    - [Outliers](#outlier)  
    - [Project Objectives Review](#project-objectives-review)
    - [Correlation Matrix](#correlation-matrix)
    - [Data Distribution](#data-distribution)
    - [Feature Engineering](#feature-engineering)   
    - [Correlation Matrix](#correlation-matrix)  
- [Model](#model)
    - [Logistic Regression](#logistic-regression)
    - [Random Forest](#random-forest) 
    - [Result](#result)    
- [Future Works](#future-works)

# [PREDICTING CUSTOMER CONVERSION (SIGN UP) FOR ECOMMERCE WEBSITES](#project-title)
The focal point of this study revolves around investigating the impact of data quality on the models employed to enhance conversions. According to [Statista](https://www.statista.com/outlook/dmo/ecommerce/worldwide), projections indicate that the worldwide e-commerce market is expected to reach a value of US$3.64tn by the year 2023, experiencing an annual growth rate of 11.17%. These trends will contribute to a market volume of around US$5.56tn by 2027. Many businesses widely recognise the significance of websites as the primary representation of a company. Consequently, companies heavily depend on the efficacy of their websites to attract and retain visitors. To achieve the desired conversion rate, evaluating user interactions with webpages and optimising the user experience is imperative. This evaluation process is pivotal in designing and developing an efficient website.

## [Project Objectives](#project-objectives)
1.	To identify the factors that influence user signup on websites.  
2.	To analyse the interaction patterns before signing up.  
3.	To explore strategies and techniques for enhancing user signup.  
4.	To provide recommendations for optimising website interaction to increase signup.

# [Features](#features)
The analysis begins by importing the necessary libraries and conducting exploratory data analysis to gain insightful information regarding the dataset. Following this, supervised machine learning models, namely Random Forest and Logistic Regression, are applied. The goal is to give readers insights into the factors contributing to website conversions (signup).

# [Technologies](#technologies)
The project utilizes the following technologies and Python libraries:

- Python: The core programming language used for the project.
- Pandas: A powerful data manipulation and analysis library.
- NumPy: A fundamental package for numerical operations in Python.
- Seaborn: A data visualization library based on Matplotlib, providing informative statistical graphics.
- Matplotlib: A comprehensive plotting library for creating static, animated, and interactive visualizations in Python.
- Scikit-learn (sklearn):
    - LabelEncoder / One Hot Encoding: Used for encoding categorical features to numeric values.
    - train_test_split: Utilized for splitting the dataset into training and testing sets.
    - GridSearchCV: Employed for hyperparameter tuning using cross-validated grid search.
    - RandomForestClassifier / Logistic Regression: Machine learning algorithm used for classification tasks.
    - roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix: Functions for evaluating machine learning model performance.

These technologies are fundamental in performing data analysis, visualization, preprocessing, modelling, and evaluation within the project.

# [Installation](#installation)
The installation is straightforward and will require you only to run the command below if you decide to run it within a virtual environment (if you do not wish to overwrite existing installations) or in the global environment of your computer's system.
 
 `pip install -r requirements.txt`

This command will read the **requirements.txt** file and install all the listed dependencies, including matplotlib, numpy, pandas, seaborn, and sklearn since Python's time module is a fundamental part. The installation will be done using the most recent version of the libraries at the time of installation since versions are not included in the **requirement.txt**.

There is an option to install each library in the requirements.txt file individually by typing **pip install** followed by the library name, as shown below. 

`pip install pandas`

# [Usage](#usage)
The fastest way to get this project running would be to install [Anacoda](https://www.anaconda.com/). After the installation, you open the Jupyter Notebook and open the folder containing the .ipynb file. Unfortunately, this project was done using proprietary datasets from **Esodora Electronics**; with that regard, we can only share 500 modified rows of the dataset for privacy concerns that can be found in the **/dataset** folder. This dataset is shared to create an understanding of the nature of the dataset use. Feel free to use tools like [Mockaroo](https://www.mockaroo.com/) to simulate additional rows for the dataset.

# [Folder Structure](#folder-structure)
- ecommerce-signup/
    - models/
        - random_forest.ipynb
        - log_regression.ipynb
     - eda/
        - dataset.csv
 	- m_dataset.csv (this will be generated after running the **eda.ipynb** script)  
    - eda/
        - eda.ipynb
    - visualizations/
    - README.md
    - md/  
        - CODE_OF_CONDUCT.md    
    - requirements.txt

# [Contributing](#contributing)
We welcome contributions to improve this project! To contribute, follow these steps:
1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make and commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Submit a pull request.

## Pull Request Guidelines
- Ensure your code follows the project's style and conventions.
- Explain the changes you've made and briefly overview the problem you're solving.
- Ensure your commits have clear messages.

## Code of Conduct
We have a [Code of Conduct](md/CODE_OF_CONDUCT.md). Please follow it in all your interactions with the project.

## Reporting Issues
If you encounter any issues with the project or would like to suggest enhancements, please open an issue on GitHub or contact me via my social media handles, Twitter (X): @chidiesobe or Instagram @ceesobe. The case will be reviewed and addressed accordingly.

## Contact
If you have any questions, need further assistance, or are just passing by, please contact me via my social media handles, Twitter (X): @chidiesobe or Instagram @ceesobe.

## [License](#license)
This project is licensed under the [MIT License](https://opensource.org/license/mit/).

# [Results and Findings](#results-findings)
This project involves analysing an e-commerce dataset comprising 11,112 rows and 23 columns. The column names and descriptions are outlined below:

- **Channel Grouping:** Inbound traffic source grouping (e.g., organic search, paid search, social).
- **Event Name:** Specific user interactions/actions on the website (e.g., video play, button click).
- **Operating System:** User's device operating system.
- **Screen Resolution:** Diagonal screen size of the user's device.
- **Search:** Number of searches carried out.
- **Added to Cart / Removed from Cart:** Number of products added to or removed from cart.
- **Begin Checkout:** Number of products purchased.
- **Browser:** User's browser.
- **Page Path:** Specific URL path of viewed pages.
- **Country / City:** User's location.
- **Day of Week:** Day when the session occurred.
- **Device Category:** Type of device used (desktop, mobile, tablet).
- **Session:** User's interactions within a time frame.
- **Views:** Number of pages viewed (excluding cart and checkout).
- **Event Count / Event Count per User:** Average events triggered per user.
- **Events per Session:** Average events triggered during a session.
- **User Engagement:** Measure of user interaction activity.
- **Engaged Session:** Session with meaningful user interaction.
- **Engagement Rate:** Percentage of sessions with meaningful interaction.
- **Conversion:** Indication if a user signed up or not.

## [Descriptive Statistics](#descriptive-statistics)
A further look into the dataset reveals the frequently occurring categorical value for each categorical column as the average, minimum and maximum values for the numerical columns.

**Categorical Variables Mode**

| session default channel grouping| event  name| operating systems| screen resolution| browser | page path |country  |city   |day of week |device category |
|---------------------------------|------------|------------------|------------------|---------|-----------|---------|-------|------------|----------------|
| organic search                  | page_view  | android          |1366x768          | Chrome  | /         | Nigeria | Lagos | Friday     | mobile         |


**Numerical Variables Description**


|      | search |added to cart | removed from cart |begin checkout| session | views | event count |event count per user | engaged session|conversion| user engagement | engaged session | engagement rate| 
|------|--------|--------------|-------------------|--------------|---------|-------|-------------|---------------------|----------------|----------|-----------------|-----------------|----------------|
| mean | 4.52   | 2.54         | 1.26              | 0.41         | 1.49    | 2.06  | 2.34        | 2.34                | 1.58           | 0.45     | 1.98            | 1.28            | 0.94           |
| min  | 0      | 0            | 0                 | 0            | 1       | 1     | 1           | 1                   | 1              | 0        | 1               | 1               | 0.13           |
| 25%  | 2      | 1            | 0                 | 0            | 1       | 1     | 1           | 1                   | 1              | 0        | 1               | 1               | 1              |
| 50%  | 5      | 3            | 1                 | 0            | 1       | 2     | 1           | 1                   | 1              | 0        | 1               | 1               | 1              |
| 75%  | 7      | 4            | 2                 | 1            | 1       | 2     | 2           | 2                   | 1.67           | 1        | 2               | 1               | 1              |
| max  | 9      | 5            | 5                 | 1            | 26      | 133   | 51          | 51                  | 51             | 1        | 35              | 23              | 1              |

## [Duplicate Values](#duplicate-values)
The event count and the event count per user column were duplicates of each other, so the event count column was dropped, leaving the dataset with 22 columns.

`dataset.drop('event count',axis=1, inplace=True)`

## [Missing Values](#missing-values)
A check for missing values returned 541 missing values, with channel grouping having the most missing rows with 189 missing values, as shown in the table below.

| Columns                           | Missing Values|
|-----------------------------------|---------------|
| session default channel grouping  | 189           |
| country                           | 12            |
| city                              | 158           |
| event count per user              | 91            |
| user engagement                   | 91            |

Dropping all rows with missing values left the dataset with 10678 rows and 22 columns.

## [Outliers](#outlier)
The identification of outliers in numerical variables is accomplished by utilising the z-score. This statistical metric quantifies the association between values and the mean of a given group of values. 

**Outliers**

| Column                | Number of Outliers | Variable Type |
|-----------------------|--------------------|---------------|
| session               | 240                | numeric       |
| views                 | 101                | numeric       |
| event count per user  | 230                | numeric       |
| events per session    | 219                | numeric       |
| user engagement       | 259                | numeric       |
| engaged session       | 167                | numeric       |
| engagement rate       | 281                | numeric       |



The z-score equation is represented below:

`z = (x-μ)/σ`  
Where:  
- X is the data point.  
- μ is the mean of the dataset.  
- σ is the standard deviation of the dataset.

The outlier was handled using Winsorization, a statistical technique used to manage outliers in numerical data. It involves capping extreme values in a dataset by replacing them with values at a specified percentile. This technique is a way to mitigate the impact of outliers on statistical analysis or modelling while retaining the overall distribution and relationships in the data. Then, we proceeded to get the summary statistics after Winsorization to validate the changes, as shown below.

|          | search |added to cart | removed from cart |begin checkout| session | views |event count per user | engaged session|conversion| user engagement | engaged session | engagement rate| 
|----------|--------|--------------|-------------------|--------------|---------|-------|---------------------|----------------|----------|-----------------|-----------------|----------------|
| Mean     | 4.51   | 2.54         | 1.23              | 0.41         | 1.39    | 1.86  | 2.08                | 1.45           | 0.46     | 1.82            | 1.21            | 0.94           |
| Min      | 0      | 0            | 0                 | 0            | 1       | 1     | 1                   | 1              | 0        | 1               | 1               | 0.50           |
| 25%      | 2      | 1            | 0                 | 0            | 1       | 1     | 1                   | 1              | 0        | 1               | 1               | 1              |
| 50%      | 4      | 3            | 1                 | 0            | 1       | 2     | 1                   | 1              | 0        | 1               | 1               | 1              |
| 75%      | 7      | 4            | 2                 | 1            | 1       | 2     | 2                   | 1.67           | 1        | 2               | 1               | 1              |
| Max      | 9      | 5            | 4                 | 1            | 4       | 4     | 8                   | 4              | 1        | 5               | 3               | 1              |


## [Project Objectives Review](#project-objectives-review)

1.	To identify the factors that influence user signup on websites.  
 ![Items added to cart](/visualisations/addcart.png)

The histogram above illustrates a uniform distribution concerning how many things are added to the cart. It further indicates that visiting users exhibited the highest conversion rate, specifically regarding signups, when the maximum number of items were added to the cart. The KDE figure below illustrates a negative correlation between the number of products removed from the cart and the conversion rate (signup).

![Itms removed from cart](/visualisations/removecart.png)

2.	To analyse the interaction patterns before signing up.  
As previously mentioned, most activities took place in instances where the engagement rate, a metric that gauges user interaction with the website's page, was at its peak.
![Engagement Rate](/visualisations/engagementrate.png)


## [Data Distribution](#data-distribution)
Finally, we checked the balance distribution of the dataset, revealing a relatively balanced dataset, as depicted in the figure below.

[Data distribution](/visualisations/distribution.png)

## [Feature Engineering](#feature-engineering)
Feature engineering was carried out to get better descriptive information about each row of the dataset and also aid in applying machine learning models to predict the likelihood of a flight being delayed. We converted the **day of week** column to their numerical equivalent using Python's` map()` function.
This was followed by splitting the **screen resolution** column into two separate columns called **screen_width** and **screen_height** while dropping the parent column.

Additional feature engineering includes label encoding and one hot encoding, which converts categorical variables into a numerical representation. One-hot encoding and label encoding are important ways to prepare category data for machine learning models. They are used for different things and have different qualities.

One-hot encoding changes categorical variables into binary ones and zeros. This makes the feature space bigger by giving each group its own binary column. In this way of showing the data, only one column (the "hot" column) has a 1 in it to show the group, while the other columns have 0. This method is very important for algorithms like neural networks, in which each group is represented in a unique way without any order. However, it can make the information more sparse and have more dimensions.

Label encoding, on the other hand, gives each group a unique number. It doesn't add more features, but it gives each category a number. This method works well when the order of the categories makes sense, but it should be used carefully so that the model thinks that the order is not important. Label-encoded data works well with algorithms that can handle numerical data, like decision trees.

Finally, the newly created columns were renamed appropriately, and all column names were converted to lowercase, with the target variable conversion renamed to signup. We dropped the country column as well. Leaving the dataset with 10678 rows and 28 columns, as well as the following list of columns shown below:
- event_name
- search
- added_to_cart
- removed_from_cart
- begin_checkout
- browser
- page_path
- city
- day_of_week
- device_category
- session
- views
- event_count_per_user
- events_per_session
- **signup**
- user_engagement
- engaged_session
- engagement_rate
- screen_width
- screen_height
- direct
- organic_search
- organic_social
- referral
- android
- ios
- macintosh
- windows


# [Model](#model)

The modelling phase starts by importing the newly formatted dataset and passing it through the three selected models for the project, as shown below:

## [Logistic Regression](#logistic-regression)
Logistic regression is a statistical method used for modelling the relationship between a dependent variable and one or more independent variables. We conducted a binary classification analysis using logistic regression in this project section. The dataset is split into training and testing sets. Hyperparameter tuning is performed using Grid Search to optimize the logistic regression model's regularization strength ('C'). The model is trained with the best hyperparameters, and its performance is evaluated using metrics such as accuracy, precision, recall, and F1-score on the test set. The best model and hyperparameters are identified, with 'C' value of 100 deemed optimal. The execution time of the entire process is measured, providing insights into the efficiency of the code and the model's training and evaluation. The results comprehensively assess the model's ability to predict signups and non-signups in the given dataset.

The feature importance indicates the influence of specific features in your dataset on the model's predictions, as shown below:

1. **referral (0.1150):** Referral significantly influences the model's predictions. In this context, "referral" refers to users who arrived at the platform through a referral link or source. The fact that this is highly important suggests that how users land on the platform (via referrals) is crucial in determining if they signup. 
2. **organic_search (0.0909):** Similarly, organic_search also has notable feature importance. This typically refers to users who reach the platform through organic search results (e.g., using search engines like Google). The relatively high importance indicates that the way users discover the platform plays a significant predictor of the target variable. Improving search engine visibility or optimizing content for organic search can be key strategies to enhance user acquisition and achieve desired outcomes.
3. **event_name (0.0844):** event_name is the least influential feature of the three considered, and this suggests that the type or category of events taking place (e.g., clicks, views, interactions) plays a significant role in predicting signup. 

From the confusion matrix of the logistic regression, we can see that. 
- There are 352 instances of True Positives (TP). That is, the model predicted the sample to be in class 1, and it belongs to class 1.
- There are 800 instances of True Negatives (TN). That is, the model predicted the sample to be in class 0, and it belongs to class 0.
- There are 354 instances of False Positives (FP). The model predicted the sample to be in class 1, but it belongs to class 0.
- There are 630 instances of False Negatives (FN). The model predicted the sample to be in class 0, but it belongs to class 1.

![Confusion Matrix Logistic Regression](/visualisations/log_confusion.png)

While the ROC curve (AUC) of 0.55 indicates that the model's performance in distinguishing between the two classes (0 and 1) is somewhat better than random chance, it is not a strong classifier. AUC between 0.5 and 1 indicates the model has some discriminatory power, with a higher AUC indicating a better-performing model.
In this project, the AUC of 0.55 indicates that the model is slightly better than random chance but not significant enough.
![ROC Curve Logistic Regression](/visualisations/log_roc.png)

## [Random Forest](#random-forest)
The Random Forest algorithm is a popular ensemble learning technique in machine learning. It involves the combination of many decision trees in order to improve the accuracy of predictions and mitigate the issue of overfitting. In this project phase, the model underwent evaluation and hyperparameter tuning using a Random Forest classifier. Hyperparameter tuning involved testing 108 combinations using techniques like grid search. The best hyperparameters for the model were identified as 'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, and 'max_depth': None. An accuracy of approximately 81% was achieved on the test dataset. Precision, recall, F1-score, and AUC-ROC metrics were computed, indicating a balanced performance in predicting signups. The evaluation also considered the time to run the model, which amounted to around 112.34 seconds.
This project involved splitting the dataset, defining a parameter grid for hyperparameter tuning, initializing the random forest classifier, and performing a randomized search for the best hyperparameters. The best model was then evaluated on the test set, and relevant performance metrics were computed. The execution time was monitored and reported for efficient analysis and comparison of the model's performance and the feature importance was also determined for the best model, with the top three features shown below:

1. **page_path** has an importance of approximately 18.09%.
2. **screen_height** has an importance of approximately 9.58%.
3. **day_of_week** has an importance of approximately 8.93%.

These values indicate each feature's relative contribution to the model's predictive power, as explained in the logistic regression section of the project. Higher values suggest that the feature has a greater influence on the model's predictions.

The confusion matrix below shows that the model correctly predicted 1401 not-signup (0) as 0 while wrongly predicting 288 not-signup as 1. The same occurred with the signup, with 312 signups (1) wrongly predicted as not-signup (0), while 1203 signups (1) were rightly predicted. 

![Correlation Matrix Random Forest](/visualisations/rand_confusion.png)

The ROC curve visually demonstrates the trade-off between true and false positive rates, helping you choose an appropriate classification threshold for your model based on the problem's context and priorities. The ROC curve value of 0.91 indicates that the model can correctly predict positive and negative cases, making it a strong performer in the binary classification task.

![ROC Curve Random Forest](/visualisations/rand_roc.png)

# [Result](#result)

In comparing model accuracy, Random Forest outperformed Logistic Regression, achieving an 81% accuracy compared to the latter's 55%. Random Forest demonstrated superior predictive ability, showcasing its strength as a machine learning model over traditional Logistic Regression in this context.

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 55%      |
| Random Forest         | 81%      |


# [Future Works](#future-works)

Here are potential practical implications for this project:

1. **Machine Learning-driven Personalisation:** Utilising predictive algorithms to offer personalised product recommendations based on user behaviour can significantly enhance the user experience by providing relevant and tailored content.
2. **Predictive Pre-loading:** Implementing predictive pre-loading of pages or content based on user behaviour can ensure a faster and more seamless browsing experience, further improving user satisfaction.
3. **Improved Search Functionality:** The study emphasises the significance of a robust and user-friendly search function. Implementing predictive search with autosuggestions can improve user experience by assisting people in more efficiently finding what they are looking for.


**NOTE: The results shown in the .ipynb files are obtained using the full dataset. Running these scripts would alter the result.**