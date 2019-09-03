import numpy as np


# TODO: give these better names
from matplotlib import pyplot


def discrete(x):
    if x < 0.2:
        return np.array([1, 0, 0, 0, 0])
    elif x < 0.4:
        return np.array([0, 1, 0, 0, 0])
    elif x < 0.6:
        return np.array([0, 0, 1, 0, 0])
    elif x < 0.8:
        return np.array([0, 0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 0, 1])

# if  I have more time I should use a gaussian distribution instead of a constant distributio
# TODO: give these better names
def constant_distribution(x):
    pos_0, pos_1, pos_2, pos_3, pos_4 = 0,0,0,0,0
    if x < 0.1:
        pos_0 = 1
    elif x < 0.3:
        pos_1 = (x - 0.1) / 0.2
        pos_0 = 1 - pos_1
    elif x < 0.5:
        pos_2 = (x - 0.3) / 0.2
        pos_1 = 1 - pos_2
    elif x < 0.7:
        pos_3 = (x - 0.5) / 0.2
        pos_2 = 1 - pos_3
    elif x < 0.9:
        pos_4 = (x - 0.7) / 0.2
        pos_3 = 1 - pos_4
    else:
        pos_4 = 1
    return [pos_0, pos_1, pos_2, pos_3, pos_4]


def categorize(np_array, cat_function, n):
    out_array = np.zeros((np_array.shape[0], n))
    for i, n in enumerate(np_array):
        out_array[i,:] = cat_function(n)
    return out_array

def show_loss_graph(history):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def show_accuracy_graph(history):
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()