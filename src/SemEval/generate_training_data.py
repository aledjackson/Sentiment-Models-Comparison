import json
import csv
import time
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import pandas as pd
import csv

SEMEVAL_PATH = "datasets\SemEval2013B\\"

obj_to_neutral = np.vectorize(lambda x: "neutral" if x in ("objective-OR-neutral","objective") else x)

def remove_quotes(x):
    out = x
    if x[0] == "\"":
        out = x[1:-1]
    return out

rmv_quotes = np.vectorize(remove_quotes)

def get_datasets():
    train = pd.read_csv(SEMEVAL_PATH + "train.tsv", sep='\t', header=None, error_bad_lines=False, usecols=[2,3],quoting=csv.QUOTE_NONE)
    dev = pd.read_csv(SEMEVAL_PATH + "dev.tsv", sep='\t', header=None, error_bad_lines=False, usecols=[2,3],quoting=csv.QUOTE_NONE)
    test = pd.read_csv(SEMEVAL_PATH + "test.tsv", sep='\t', header=None, error_bad_lines=False, usecols=[2,3],quoting=csv.QUOTE_NONE)
    train.values[:,0] = obj_to_neutral(train.values[:,0])
    dev.values[:, 0] = obj_to_neutral(dev.values[:, 0])
    test.values[:, 0] = obj_to_neutral(test.values[:, 0])
    return train, dev, test


def predict_emoji(training_data, maxlen):
    '''
    predicts the emojis commonly associated with the sentences then adds it to the
    :param sentences: list of sentences to predict
    :param maxlen: max length of the setences given
    :return:
    '''

    sentences = training_data

    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(sentences)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Running predictions.')
    prob = model.predict(tokenized, batch_size=100)


    return prob


train, dev, test = get_datasets()

train.values[:,0] = obj_to_neutral(rmv_quotes(train.values[:,0]))
dev.values[:,0] = obj_to_neutral(rmv_quotes(dev.values[:,0]))
test.values[:,0] = obj_to_neutral(test.values[:,0])

# predict emojis and export predictions to a csv file
train_predictions = predict_emoji(train.values[:,1], train[3].str.len().max())
dev_predictions = predict_emoji(dev.values[:,1], dev[3].str.len().max())
test_predictions = predict_emoji(test.values[:,1], test[3].str.len().max())

pd_train_predictions = pd.DataFrame(train_predictions)
pd_dev_predictions = pd.DataFrame(dev_predictions)
pd_test_predictions = pd.DataFrame(test_predictions)


column_names = ["sentiment", "sentence"]
column_names += [i for i in range(64)]

emoji_to_binary_train = pd.concat([train.reset_index(drop=True), pd_train_predictions.reset_index(drop=True)],
                                      axis=1, ignore_index=True)
emoji_to_binary_train.columns = column_names


emoji_to_binary_dev = pd.concat([dev.reset_index(drop=True), pd_dev_predictions.reset_index(drop=True)], axis=1,
                                    ignore_index=True)
emoji_to_binary_dev.columns = column_names

emoji_to_binary_test = pd.concat([test.reset_index(drop=True), pd_test_predictions.reset_index(drop=True)], axis=1,
                                     ignore_index=True)
emoji_to_binary_test.columns = column_names

emoji_to_binary_train.to_csv(path_or_buf=SEMEVAL_PATH + "emoji_train.csv", sep='|', index=False)
emoji_to_binary_dev.to_csv(path_or_buf=SEMEVAL_PATH + "emoji_dev.csv", sep='|',index=False)
emoji_to_binary_test.to_csv(path_or_buf=SEMEVAL_PATH + "emoji_test.csv", sep='|', index=False)
