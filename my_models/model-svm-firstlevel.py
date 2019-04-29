import sys
import numpy as np
import pandas as pd
import os

os.system("(pip install -U keras==2.2.4)")
os.system("(pip install -U scikit-learn==0.20.3)")
os.system("(pip install nltk)")

import nltk
nltk.download('punkt')

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

train_file = ""
test_file = ""

def main(argv):

    if len(argv) < 4:
        sys.exit("Not enough arguments provided.")

    global train_file, test_file
    i = 1
    while i <= 4:
        arg = str(argv[i])
        if arg == "--train":
            train_file = str(argv[i+1])
        elif arg == "--test":
            test_file = str(argv[i+1])
        i += 2

def prepare_data(filename):
    data = pd.read_csv(filename, sep="\t")
    data = data[['text', 'subj']]
    data['subj'] = data['subj'].apply(lambda subj: subj.split('\\'))
    mlb = MultiLabelBinarizer()
    encoded_subjects = pd.DataFrame(mlb.fit_transform(data.pop('subj')), columns=mlb.classes_, index=data.index)
    data = data.join(encoded_subjects)
    return data, mlb.classes_

if __name__ == "__main__":
    main(sys.argv)

train, categories = prepare_data(train_file)
test, _ = prepare_data(test_file)

X_train = train.text
X_test = test.text
Y_train = train[categories]
Y_test = test[categories]

pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.2, ngram_range=(1, 2), max_features=None)),
    ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
], memory='cache')

pipeline.fit(X_train, Y_train)

predictions = pipeline.predict(X_test)

print(classification_report(np.array(Y_test), predictions, target_names=categories))