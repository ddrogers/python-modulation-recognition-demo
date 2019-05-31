'''
Modulation Recognition with Deep Artificial Neural Networks
Analog Modulation
'''
import os,sys,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
import matplotlib.pyplot as plt
import pickle as cPickle
import keras
class AnalogBenchmarker():

    def __init__(self):
        # Load the dataset
        #  You will need to seperately download or generate this file
        Xd = cPickle.load(open("data/RML2016.10a_dict.dat", 'rb'), encoding="latin1")
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
        X = []
        lbl = []
        analog_mods = ['WBFM', 'AM-SSB', 'AM-DSB']
        for mod in analog_mods:
            for snr in snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
        X = np.vstack(X)

        # Partition the data
        np.random.seed(2018)
        n_examples = X.shape[0]
        n_train = int(n_examples * 0.5)
        train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        self.X_train = X[train_idx]
        self.X_test = X[test_idx]

        def to_onehot(yy):
            yy = list(yy)
            yy1 = np.zeros([len(yy), max(yy) + 1])
            yy1[np.arange(len(yy)), yy] = 1
            return yy1

        self.Y_train = to_onehot(map(lambda x: analog_mods.index(lbl[x][0]), train_idx))
        self.Y_test = to_onehot(map(lambda x: analog_mods.index(lbl[x][0]), test_idx))

        self.in_shp = list(self.X_train.shape[1:])
        print(self.X_train.shape, self.in_shp)
        self.classes = analog_mods

    def set_parameters(self):
        self.nb_epoch = 50 # number of epochs to train
        self.batch_size = 1024
        self.nodes1 = 500

    def build_ANN_model(self):
        # Initialize the ANN
        self.analog_classifier = Sequential()
        self.analog_classifier.add(Reshape(self.in_shp + [1], input_shape=self.in_shp))
        self.analog_classifier.add(Flatten())
        # Stack Dense Layers with Dropout
        self.analog_classifier.add(Dense(self.nodes1,kernel_initializer='glorot_uniform',
                                    activation="relu"))
        # Output Layer
        self.analog_classifier.add(Dense(len(self.classes),kernel_initializer='he_normal'))
        self.analog_classifier.add(Activation('softmax'))
        self.analog_classifier.add(Reshape([len(self.classes)]))

        # Compiling the ANN
        self.analog_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.analog_classifier.summary()

    def train_ANN_model(self):
        # perform training
        filepath = 'data/models/ann_analog_benchmark.h5'
        '''Early Stopping'''
        self.history = self.analog_classifier.fit(self.X_train, self.Y_train,
                                             batch_size = self.batch_size,
                                            epochs = self.nb_epoch,
                                            verbose = 2,
                                            validation_data = (self.X_test, self.Y_test),
                                            callbacks = [
                                                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',verbose=0,
                                                                                save_best_only=True,mode='auto'),
                                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                                                              mode='auto')
                                            ])
        # Re-load the best weights after training has finished
        self.analog_classifier.load_weights(filepath)


    def evaluate_ANN_Model(self):
        score = self.analog_classifier.evaluate(self.X_test, self.Y_test, verbose=0, batch_size=self.batch_size)
        print(score)

    def display_loss_curves(self):
        plt.figure()
        plt.title('Benchmark: Training Performance - Analog')
        plt.plot(self.history.epoch, self.history.history['loss'],label='Train Loss+Error')
        plt.plot(self.history.epoch, self.history.history['val_loss'], label='val_error')
        plt.legend()
        plt.show()
        # Save image to data/loss_curves
        plt.savefig("data/loss_curves/ann_analog_benchmark_loss_curves.png")


if __name__ == '__main__':
    print("The analog_benchmark.py script has been loaded.")
    ab = AnalogBenchmarker()
    ab.set_parameters()
    ab.build_ANN_model()
    ab.train_ANN_model()
    ab.evaluate_ANN_Model()
    ab.display_loss_curves()
else:
    print("The analog_benchmark.py script has been imported.")