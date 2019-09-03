import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

layers = tf.keras.layers

SAVE_PATH = "keras_model_files\StanfordSST\emoji_to_fine.h5"


class sst_fine_predictor:
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(128,input_shape=(64, ), activation='tanh'))
        self.model.add(
            layers.Dense(128, activation='tanh'))
        self.model.add(
            layers.Dense(128, activation='tanh'))
        self.model.add(
            layers.Dense(128, activation='tanh'))
        self.model.add(layers.Dense(5, activation='softmax'))
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.01),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return

    def __init__(self):
        self.build_model()
        self.stopper_1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        # self.stopper_2 = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=200)
        self.checkpointer = ModelCheckpoint(SAVE_PATH, monitor='val_acc', mode='max', save_best_only=True)

    def __call__(self):
        return self.model

    def train(self, train_ins, train_outs, test_ins, test_outs, max_epochs=4000):
        history = self().fit(train_ins, train_outs, validation_data=(test_ins, test_outs), epochs=max_epochs, verbose=1,
                             callbacks=[self.stopper_1, self.checkpointer], batch_size=8117)
        return history

