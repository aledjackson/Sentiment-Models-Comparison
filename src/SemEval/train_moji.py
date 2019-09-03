import pandas as pd
import numpy as np

from src.SemEval.models.moji_to_trinary import MojiToTrinary

from src.helper_functions import categorize

SEMEVAL_PATH = "datasets\SemEval2013B\\"

def convert_to_vector(x):
    if x == "positive":
        return [0,0,1]
    elif x == 'neutral':
        return [0, 1, 0]
    else:
        return [1, 0, 0]

training_data = pd.read_csv(SEMEVAL_PATH + "emoji_train.csv", sep="|")
test_data = pd.read_csv(SEMEVAL_PATH + "emoji_test.csv", sep="|")

train_ins = np.delete(training_data.values, [0,1], axis=1)
train_outs = categorize(training_data.values[:,0], convert_to_vector, 3)

# test_outs are changed from 0 to 1 so we can see how many sentences are correctly classified
test_ins = np.delete(test_data.values, [0,1], axis=1)
test_outs = categorize(test_data.values[:,0], convert_to_vector,3)

print("building model")
print("...")
model = MojiToTrinary()
print("model built")

print("press enter to begin training")
input()

print("training model")
history = model.train(train_ins, train_outs, test_ins, test_outs, max_epochs=20000)
