import json
import csv
import time
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import pandas

SST_PATH = "datasets\stanfordSentimentTreebank\\"


def get_datasets():
    labels = pandas.read_csv(SST_PATH + "\\sentiment_labels.txt", sep='|')
    dictionary = pandas.read_csv(SST_PATH + "\\dictionary.txt", sep='|')
    sentences = pandas.read_csv(SST_PATH + "\\datasetSentences.txt", sep='\t')
    splits = pandas.read_csv(SST_PATH + "\\datasetSplit.txt", sep=',')

    labels_w_phrases = labels.merge(dictionary, on="phrase ids", how='inner')

    sentences_w_labels = pandas.merge(sentences, labels_w_phrases, left_on='sentence', right_on='phrases', how='inner')
    del sentences_w_labels['phrase ids']
    del sentences_w_labels['phrases']

    splitset_dict = {
        "train": 1,
        "test": 2,
        "dev": 3
    }

    split_labeled_data = sentences_w_labels.merge(splits, on='sentence_index', how='inner')
    del split_labeled_data['sentence_index']

    training_data = split_labeled_data.where(split_labeled_data['splitset_label'] == splitset_dict['train']).dropna()
    dev_data = split_labeled_data.where(split_labeled_data['splitset_label'] == splitset_dict['test']).dropna()
    test_data = split_labeled_data.where(split_labeled_data['splitset_label'] == splitset_dict['dev']).dropna()
    del training_data['splitset_label']
    del dev_data['splitset_label']
    del test_data['splitset_label']

    return training_data, dev_data, test_data


def predict_emoji(training_data, maxlen):
    '''
    predicts the emojis commonly associated with the sentences then adds it to the
    :param sentences: list of sentences to predict
    :param maxlen: max length of the setences given
    :return:
    '''

    def top_elements(array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    sentences = training_data['sentence']

    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(sentences)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Running predictions.')
    prob = model.predict(tokenized, batch_size=500)

    # Find top emojis for each sentence. Emoji ids (0-63)
    # correspond to the mapping in emoji_overview.png
    # at the root of the DeepMoji repo.
    # print('Writing results to {}'.format(OUTPUT_PATH))
    # scores = []
    # for i, t in enumerate(sentences):
    #     t_tokens = tokenized[i]
    #     t_score = [t]
    #     t_prob = prob[i]
    #     ind_top = top_elements(t_prob, 5)
    #     t_score.append(sum(t_prob[ind_top]))
    #     t_score.extend(ind_top)
    #     t_score.extend([t_prob[ind] for ind in ind_top])
    #     scores.append(t_score)
    #     print(t_score)

    return prob

    # with open(output_filename, 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    #     writer.writerow(['Text', 'Top5%',
    #                      'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
    #                      'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
    #     for i, row in enumerate(scores):
    #         try:
    #             writer.writerow(row)
    #         except:
    #             print("Exception at row {}!".format(i))


train, dev, test = get_datasets()

# predict emojis and export predictions to a csv file
train_predictions = predict_emoji(train, train['sentence'].str.len().max())
dev_predictions = predict_emoji(dev, dev['sentence'].str.len().max())
test_predictions = predict_emoji(test, test['sentence'].str.len().max())

pd_train_predictions = pandas.DataFrame(train_predictions)
pd_dev_predictions = pandas.DataFrame(dev_predictions)
pd_test_predictions = pandas.DataFrame(test_predictions)


column_names = ["sentence", "sentiment_score"]
column_names += [i for i in range(64)]

emoji_to_binary_train = pandas.concat([train.reset_index(drop=True), pd_train_predictions.reset_index(drop=True)],
                                      axis=1, ignore_index=True)
emoji_to_binary_train.columns = column_names


emoji_to_binary_dev = pandas.concat([dev.reset_index(drop=True), pd_dev_predictions.reset_index(drop=True)], axis=1,
                                    ignore_index=True)
emoji_to_binary_dev.columns = column_names

emoji_to_binary_test = pandas.concat([test.reset_index(drop=True), pd_test_predictions.reset_index(drop=True)], axis=1,
                                     ignore_index=True)
emoji_to_binary_test.columns = column_names

emoji_to_binary_train.to_csv(path_or_buf=SST_PATH + "emoji_train.csv", sep='|')
emoji_to_binary_dev.to_csv(path_or_buf=SST_PATH + "emoji_dev.csv", sep='|')
emoji_to_binary_test.to_csv(path_or_buf=SST_PATH + "emoji_test.csv", sep='|')
