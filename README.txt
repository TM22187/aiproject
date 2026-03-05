Spam & Malicious URL Detector
Am AI that detects spam messages and malicious URLs using two independent models combined into a single prediction pipeline.

Requirements

Python 3.9+. I actually use 3.13.7 as environment in notebooks.
conda (recommended)


1. Environment Setup
Create and activate a conda environment:
bashconda create -n spam-detector python=3.9
conda activate spam-detector
Install required packages:
bashpip install pandas numpy scikit-learn matplotlib seaborn wordcloud jupyter

2. Data Preparation
I placed the raw datasets in the data/raw/ folder

3. Running the Notebooks
Launch Jupyter:
bashjupyter notebook
Run the notebooks in order for each datase. Must run in order if not it will be error.
Spam
notebooks/spam/1_spam_data_processing.ipynb   for clean & process data
notebooks/spam/2_spam_visualization.ipynb     for charts & word clouds
notebooks/spam/3_spam_model_training.ipynb    for train & save model
URL
notebooks/url/1_url_data_processing.ipynb     for extract URL features
notebooks/url/2_url_visualization.ipynb       for charts & analysis
notebooks/url/3_url_model_training.ipynb      for train & save model

Note: Always run the data processing notebook before the training notebook. The training notebook reads from data/processed/.


4. Trained Models
After running all training notebooks, the following files will be saved in models/:
1, spam_model.pkl, spam_tfidf.pkl for Spam.
2, url_feature_cols.pkl, url_model.pkl for malicious url.


5. Running Predictions
Make sure the models/ folder contains all 4 .pkl files, then run:
python predict.py
You will see an interactive prompt:
Spam & Malicious URL Detector
   Type 'quit' to exit

Enter message:

7. Models Overview
Spam Detection

Input: Raw message text
Method: TF-IDF vectorization (5000 features, unigram + bigram)
Algorithms compared: Multinomial Naive Bayes vs Logistic Regression
Task: Binary classification — ham (0) / spam (1)

URL Classification

Input: Raw URL string
Method: Structural feature extraction (14 numeric features)
Algorithms compared: Random Forest vs Gradient Boosting
Task: Multi-class classification benign/defacement/phishing/malware


8. Troubleshooting
FileNotFoundError: models/spam_model.pkl
--> Run all training notebooks first before using predict.py
UnicodeDecodeError when loading spam dataset
--> The spam CSV uses latin-1 encoding — this is handled automatically in the notebook
