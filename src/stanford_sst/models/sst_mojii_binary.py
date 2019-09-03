import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

layers = tf.keras.layers

SAVE_PATH = "keras_model_fils\StanfordSST\\emoji_to_binary.h5"


class sst_binary_predictor:
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(128,input_shape=(64, ), activation='tanh'))
        self.model.add(layers.Dense(128, input_shape=(64,), activation='tanh'))
        self.model.add(layers.Dense(128, input_shape=(64,), activation='tanh'))
        self.model.add(layers.Dense(1,activation='sigmoid'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return

    def __init__(self):
        self.build_model()
        self.stopper = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        self.checkpointer = ModelCheckpoint(SAVE_PATH, monitor='val_acc', mode='max', save_best_only=True)

    def __call__(self):
        return self.model

    def train(self, train_ins, train_outs, test_ins, test_outs, max_epochs=4000):
        history = self().fit(train_ins, train_outs, validation_data=(test_ins, test_outs), epochs=max_epochs, verbose=1,
                             callbacks=[self.stopper, self.checkpointer], batch_size=8117)
        return history

