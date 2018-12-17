# multilabel-text-classification

#### model_svm

The model processes source texts using tfidf, then classifies by department codes using the linear svm classifier. Hyperparameters of the model were selected by a grid search.

#### reccurent_network_model

The model includes:
* Pre-process text with a tokenizer,
* Embedding layer to represent words,
* GRU layers (based on the same principles as LSTM, but using fewer filters and operations for calculations),
* Dense layer with sigmoid function.
