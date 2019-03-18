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

maxLength = 100

vocab_size = 25000
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
totalX = np.array(pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxLength))

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(glove_file, 'r+', encoding="utf-8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, maxLength))
for word, i in word_index.items():
    if i >= vocab_size: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


embedding_dim = maxLength
num_categories = len(categories)
model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    weights=[embedding_matrix], 
                    input_length=maxLength, 
                    trainable=False))
model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_categories, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(totalX, Y_train, batch_size=32, epochs=10, validation_split=0.1)

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