import pandas as pd
import numpy as np

from src.stanford_sst.models.sst_mojii_binary import sst_binary_predictor

SST_PATH = "datasets\stanfordSentimentTreebank\\"

f = lambda x: 0 if x < 0.5 else 1
f = np.vectorize(f)


training_data = pd.read_csv(SST_PATH + "emoji_train.csv", sep="|")
test_data = pd.read_csv(SST_PATH + "emoji_test.csv", sep="|")

train_ins = np.delete(training_data.values, [0,1,2], axis=1)
train_outs = f(training_data.values[:,2])

# test_outs are changed from 0 to 1 so we can see how many sentences are correctly classified
test_ins = np.delete(test_data.values, [0,1,2], axis=1)
test_outs = f(test_data.values[:,2])

print("building model")
print("...")
model = sst_binary_predictor()
print("model built")

print("press enter to begin training")
input()

print("training model")
history = model.train(train_ins, train_outs, test_ins, test_outs, max_epochs=4000)
