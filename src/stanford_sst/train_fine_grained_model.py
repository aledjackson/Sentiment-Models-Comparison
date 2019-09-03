import pandas as pd
import numpy as np

from helper_functions import discrete, constant_distribution, categorize
from model.sst_fine_grained import sst_fine_predictor

SST_PATH = "datasets\stanfordSentimentTreebank\\"

training_data = pd.read_csv(SST_PATH + "emoji_train.csv", sep="|")
test_data = pd.read_csv(SST_PATH + "emoji_test.csv", sep="|")

train_ins = np.delete(training_data.values, [0,1,2], axis=1)
train_outs = categorize(training_data.values[:, 2], discrete, 5)

# test_outs are changed from 0 to 1 so we can see how many sentences are correctly classified
test_ins = np.delete(test_data.values, [0,1,2], axis=1)
test_outs = categorize(test_data.values[:, 2], discrete, 5)

print("building model")
print("...")
model = sst_fine_predictor()
print("model built")

print("press enter to begin training")
input()

print("training model")
history = model.train(train_ins, train_outs, test_ins, test_outs, max_epochs=4000)

