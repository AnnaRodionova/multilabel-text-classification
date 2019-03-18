import sys
import numpy as np
import pandas as pd
import os

os.system("(pip install -U keras==2.2.4)")
os.system("(pip install -U scikit-learn==0.20.3)")
os.system("(pip install nltk)")

import nltk
nltk.download('punkt')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Embedding
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

    global train_file, test_file, glove_file

    i = 1
    while i <= 6:
        arg = str(argv[i])
        if arg == "--train":
            train_file = str(argv[i+1])
        elif arg == "--test":
            test_file = str(argv[i+1])
        elif arg == "--glove":
            glove_file = str(argv[i+1])
        i += 2

def prepare_data(filename):
    data = pd.read_csv(filename, sep="\t")
    data = data[['text', 'ipv']]
    data['ipv'] = data['ipv'].apply(lambda ipv: str(ipv).split('\\'))
    return data

def one_hot_transform(mlb, df):
    encoded = pd.DataFrame(mlb.transform(df.pop('ipv')), columns=mlb.classes_, index=df.index)
    df = df.join(encoded)
    return df, mlb.classes_

if __name__ == "__main__":
    main(sys.argv)

train = prepare_data(train_file)
test = prepare_data(test_file)

mlb = MultiLabelBinarizer()
mlb.fit(train['ipv'])

train, categories = one_hot_transform(mlb, train)
test, _ = one_hot_transform(mlb, test)

X_train = train.text
X_test = test.text
Y_train = train[categories]
Y_test = test[categories]

xLengths = [len(word_tokenize(x)) for x in X_train]
h = sorted(xLengths) 
maxLength = h[len(h)-1]
maxLength = h[int(len(h) * 0.70)]

vocab_size = 20000
tokenizer = Tokenizer(vocab_size)

embeddings_index = dict()
f = open(glove_file, 'r+', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

totalX = np.array(pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxLength))

embedding_dim = 100
num_categories = len(categories)
model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=maxLength, 
                    weights=[embedding_matrix], 
                    trainable=False))
model.add(GRU(256, dropout=0.6517968154887782, return_sequences=True))
model.add(GRU(256, dropout=0.4371162594318422))
model.add(Dense(num_categories, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(totalX, Y_train, batch_size=128, epochs=10, validation_split=0.1)

model.save_weights(model_weights)

totalX_test = np.array(pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxLength))
predicted = model.predict(totalX_test)

for i, row in enumerate(predicted):
    for j, prob in enumerate(row):
        if prob >= 0.5:
            predicted[i][j]=1
        else:
            predicted[i][j]=0

print(classification_report(np.array(Y_test), predicted, target_names=categories))