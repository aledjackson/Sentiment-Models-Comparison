import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import regularizers

layers = tf.keras.layers

SAVE_PATH = "keras_model_files\SemEval\\emoji_to_trinary.h5"

L2_RATIO = 0.000001
LEARNING_RATE = 0.0001
PATIENCE_RATIO = 0.05

# best acheived 66.2% accuracy on SemEval with 0.0001 L2_RATIO, LEARNING_RATE=0.00001

class MojiToTrinary:
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Dense(128,input_shape=(64, ), activation='tanh',kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.add(
            layers.Dense(128, activation='tanh',kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.add(
            layers.Dense(128, activation='tanh',kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.add(
            layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.add(
            layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.add(
            layers.Dense(3, activation='softmax',kernel_regularizer=regularizers.l2(L2_RATIO),
                activity_regularizer=regularizers.l1(L2_RATIO)))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return

    def __init__(self):
        self.build_model()
        self.stopper_1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_RATIO/LEARNING_RATE)
        # self.stopper_2 = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=200)
        self.checkpointer = ModelCheckpoint(SAVE_PATH, monitor='val_acc', mode='max', save_best_only=True)

    def __call__(self):
        return self.model

    def train(self, train_ins, train_outs, test_ins, test_outs, max_epochs=4000):
        history = self().fit(train_ins, train_outs, validation_data=(test_ins, test_outs), epochs=max_epochs, verbose=1,
                             callbacks=[self.stopper_1, self.checkpointer], batch_size=6021)
        return history

