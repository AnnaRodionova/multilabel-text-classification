import sys
import numpy as np
import pandas as pd
import os

os.system("(pip install -U keras==2.2.4)")
os.system("(pip install -U scikit-learn==0.20.3)")
os.system("(pip install nltk)")

import nltk
nltk.download('punkt')

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.layers import GRU
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import Constant
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from nltk.tokenize import word_tokenize


train_file = ""
test_file = ""
glove_file = ""
model_weights = os.environ["RESULT_DIR"]+"/weights"

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

xLengths = [len(word_tokenize(x)) for x in X_train]
h = sorted(xLengths)
maxLength = h[int(len(h) * 0.70)]

max_vocab_size = 200000
tokenizer = Tokenizer(max_vocab_size)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
totalX = np.array(pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxLength))

embedding_dim = maxLength
num_categories = len(categories)
model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=maxLength))
model.add(GRU(256, dropout=0.6517968154887782, return_sequences=True))
model.add(GRU(256, dropout=0.4371162594318422))
model.add(Dense(num_categories, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(totalX, Y_train, batch_size=128, epochs=10, validation_split=0.1)

totalX_test = np.array(pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxLength))
predicted = model.predict(totalX_test)

for i, row in enumerate(predicted):
    for j, prob in enumerate(row):
        if prob >= 0.5:
            predicted[i][j]=1
        else:
            predicted[i][j]=0

print(classification_report(np.array(Y_test), predicted, target_names=categories))