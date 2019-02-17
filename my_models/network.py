import sys
import numpy as np
import os

os.system("(pip install keras)")

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Embedding
from keras.initializers import Constant

train_features_file = ""
train_labels_file = ""
embedding_matrix_part1_file = ""
embedding_matrix_part2_file = ""
model_path = os.environ["RESULT_DIR"]+"/model"

def main(argv):

    if len(argv) < 8:
        sys.exit("Not enough arguments provided.")

    global train_features_file, train_labels_file, embedding_matrix_part1_file, embedding_matrix_part2_file

    i = 1
    while i <= 8:
        arg = str(argv[i])
        if arg == "--trainFeaturesFile":
            train_features_file = str(argv[i+1])
        elif arg == "--trainLabelsFile":
            train_labels_file = str(argv[i+1])
        elif arg == "--embeddingMatrixPart1File":
            embedding_matrix_part1_file = str(argv[i+1])
        elif arg == "--embeddingMatrixPart2File":
            embedding_matrix_part2_file = str(argv[i+1])
        i += 2

if __name__ == "__main__":
    main(sys.argv)

train_features = np.load(train_features_file)['arr_0']
train_labels = np.load(train_labels_file)['arr_0']
embedding_matrix_part1 = np.load(embedding_matrix_part1_file)['arr_0']
embedding_matrix_part2 = np.load(embedding_matrix_part2_file)['arr_0']
embedding_matrix = np.vstack((embedding_matrix_part1, embedding_matrix_part2))

model = Sequential()

num_words = 286513
max_length = 765
num_of_categories = 205
model.add(Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix), input_length=max_length, trainable=False))
model.add(GRU(256, dropout=0.9, return_sequences=True))
model.add(GRU(256, dropout=0.9))
model.add(Dense(num_of_categories, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(train_features, train_labels, batch_size=64, epochs=10, validation_split=0.1, verbose=2)

model.save(model_path)
