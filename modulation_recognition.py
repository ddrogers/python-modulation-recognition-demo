import os
import numpy as np
from keras.layers.core import Reshape, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import *
import matplotlib.pyplot as plt
import pickle as cpickle
import keras
from keras.layers import Dropout
# Parameter Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

os.environ["KERAS_BACKEND"] = "tensorflow"

#Design Modulation Recognition Neural Network Models for Cross Validation


def ann_with_dropout(nodes, act_func, optimizer, weights, o_weights, in_shp, classes):
    # Initialize ANN
    digital_classifier = Sequential()
    digital_classifier.add(BatchNormalization(input_shape=in_shp))
    digital_classifier.add(Flatten())

    layer_filters = nodes

    for layer in layer_filters:
        digital_classifier.add(Dense(layer, kernel_initializer=weights,
                                     activation=act_func))
        digital_classifier.add(Dropout(0.5))

    # Output Layer
    digital_classifier.add(Dense(len(classes), kernel_initializer=o_weights, name="dense4"))
    digital_classifier.add(Activation('softmax'))
    # Compiling the ANN
    digital_classifier.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return digital_classifier


def cnn_with_dropout(nodes, act_func, optimizer,
                     weights, o_weights, in_shp, classes):
    # Initialize ANN
    digital_classifier = Sequential()
    digital_classifier.add(Reshape(in_shp + [1], input_shape=in_shp))
    digital_classifier.add(ZeroPadding2D((0, 2)))
    # Step 1 - Convolution
    digital_classifier.add(Conv2D(16, (3, 3), kernel_initializer="lecun_normal",
                                  padding='same', name="conv1"))
    digital_classifier.add(Activation("relu"))
    digital_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    digital_classifier.add(Dropout(0.5))
    digital_classifier.add(Flatten())
    layer_filters = nodes

    for layer in layer_filters:
        digital_classifier.add(Dense(layer, kernel_initializer=weights,
                                     activation=act_func))
        digital_classifier.add(Dropout(0.5))

    # Output Layer
    digital_classifier.add(Dense(len(classes), kernel_initializer=o_weights, name="dense4"))
    digital_classifier.add(Activation('softmax'))
    # Compliing the ANN
    digital_classifier.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return digital_classifier

# Build a Modulation Recognition Class for Signal Classification and Result Visualization


class SignalClassifier():
    '''
    This class object is designed to read in the number of cross validations to be completed.
    At the end of each cross validation trial. The best Loss value and accuracy are stored.
    There will be two models for each trial, with and without dropout.
    '''
    def __init__(self,*argv):
        self.row = argv[0]
        self.col = argv[1]
        self.neural_model = argv[2]
        self.parameters = argv[3]
        self.cross_val = argv[4]
        self.drop = argv[5]
        self.mod_name = argv[6]
        self.mod_type = argv[7]
        self.modulations = argv[8]
        self.choice = argv[9]
        self.seed = argv[10]
        # Load the dataset
        #  You will need to seperately download or generate this file
        self.Xd = cpickle.load(open("data/RML2016.10a_dict.dat", 'rb'), encoding="latin1")
        self.snrs, self.mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.Xd.keys())))), [1, 0])
        self.X = []
        self.lbl = []
        self.mods = self.modulations
        self.snrs_choice = self.choice
        for mod in self.mods:
            for snr in self.snrs_choice:
                self.X.append(self.Xd[(mod, snr)])
                for i in range(self.Xd[(mod, snr)].shape[0]): self.lbl.append((mod, snr))
        self.X = np.vstack(self.X)
        # Partition the data
        np.random.seed(self.seed)
        self.n_examples = self.X.shape[0]
        self.n_train = int(self.n_examples * 0.8)
        self.train_idx = np.random.choice(range(0, self.n_examples), size=self.n_train, replace=False)
        self.test_idx = list(set(range(0, self.n_examples)) - set(self.train_idx))
        self.X_train = self.X[self.train_idx]
        self.X_test = self.X[self.test_idx]
        self.X_train = np.asarray(list(map(lambda x: np.reshape(x, (self.row, self.col)), self.X_train)))
        self.X_test = np.asarray(list(map(lambda x: np.reshape(x, (self.row, self.col)), self.X_test)))
        self.Y_train = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.train_idx))
        self.Y_test = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.test_idx))
        self.in_shp = list(self.X_train.shape[1:])
        self.classes = self.mods
        print(self.X_train.shape, self.in_shp, self.seed)

    def build_cv_model(self):
        '''Evaluate Model'''
        self.classifier = KerasClassifier(build_fn=self.neural_model)

        self.grid_search = GridSearchCV(estimator=self.classifier,
                                        param_grid=self.parameters,
                                        scoring='accuracy',
                                        cv=self.cross_val)

    def train_cv_model(self):
        filepath = 'data/models/{}_{}_{}_{}_cv_{}_epoch_{}_{}.h5'.format(self.mod_name, self.mod_type, self.row,
                                                                         self.col, self.cross_val,
                                                                         self.parameters['epochs'], self.drop)
        self.grid_search = self.grid_search.fit(self.X_train, self.Y_train,
                                                validation_data=(self.X_test, self.Y_test),
                                                callbacks=[
                                                    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                                                    verbose=0, save_best_only=True,
                                                                                    mode='auto'),
                                                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=100,
                                                                                  verbose=0, mode='auto')
                                                ])

        self.epochs = self.grid_search.best_estimator_.model.model.history.epoch
        self.best_loss = self.grid_search.best_estimator_.model.model.history.history["loss"]
        self.best_val_loss = self.grid_search.best_estimator_.model.model.history.history["val_loss"]
        self.best_acc = self.grid_search.best_estimator_.model.model.history.history["acc"]
        self.best_val_acc = self.grid_search.best_estimator_.model.model.history.history["val_acc"]

    def plot_loss_curves(self):
        '''
        Loss Curve
        '''
        # Show loss curves
        plt.figure()
        plt.title('Training performance - {} ({},{})CV: {}_{}'.format(self.mod_name, self.row, self.col, self.cross_val,
                                                                      self.drop))
        plt.plot(self.epochs, self.best_loss, label='train loss+error')
        plt.plot(self.epochs, self.best_val_loss, label='val_error')
        plt.legend()
        plt.savefig('data/loss_curves/lc_{}_{}_{}_{}_{}'.format(self.mod_name, self.row, self.col, self.cross_val,
                                                                self.drop))
        plt.close()

    def plot_overall_confusion_matrix(self):
        '''
        Overall Confusion Matrix (Includes Modulation Recognition based on all SNR)
        '''

        def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        self.batch_size = 1024

        def to_onehot(yy):
            yy1 = np.zeros([len(yy), max(yy) + 1])
            yy1[np.arange(len(yy)), yy] = 1
            return yy1

        self.Y_train = to_onehot(self.Y_train)
        self.Y_test = to_onehot(self.Y_test)
        test_Y_hat = self.grid_search.best_estimator_.model.model.predict(self.X_test, batch_size=self.batch_size)
        conf = np.zeros([len(self.classes), len(self.classes)])
        confnorm = np.zeros([len(self.classes), len(self.classes)])
        for i in range(0, self.X_test.shape[0]):
            j = list(self.Y_test[i, :]).index(1)
            k = int(np.argmax(test_Y_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(self.classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        title = 'Confusion Matrix {} ({},{}) CV: {}'.format(self.mod_name, self.row, self.col, self.cross_val)
        plot_confusion_matrix(confnorm, title=title, labels=self.classes)
        plt.savefig(
            'data/confusion_matrix/cm_{}_{}_{}_cv _{}_{}'.format(self.mod_name, self.row, self.col, self.cross_val,
                                                                     self.drop))
        plt.close()

    def plot_individual_confusion_matrices(self):
        '''
        Individual Confusion Matrices (Modulation Recognition by SNR values)
        '''

        def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        self.acc = {}
        self.acc_string = []
        for snr in self.snrs_choice:

            # extract classes @ SNR
            test_SNRs = list(map(lambda x: self.lbl[x][1], self.test_idx))
            test_X_i = self.X_test[np.where(np.array(test_SNRs) == snr)]
            test_Y_i = self.Y_test[np.where(np.array(test_SNRs) == snr)]
            # estimate classes
            test_Y_i_hat = self.grid_search.best_estimator_.model.model.predict(test_X_i)
            conf = np.zeros([len(self.classes), len(self.classes)])
            confnorm = np.zeros([len(self.classes), len(self.classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(self.classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            title = " {} Confusion Matrix (Dim = ({},{}), CV = {}, SNR={})".format(self.mod_name, self.row, self.col,
                                                                                   self.cross_val, snr)
            plot_confusion_matrix(confnorm, labels=self.classes, title=title)
            plt.savefig(
                'data/confusion_matrix/cm_{}_{}_{}_cv _{}_snr_{}_{}'.format(self.mod_name, self.row, self.col,
                                                                                self.cross_val, snr, self.drop))
            plt.close()
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            self.acc_string.append("Overall Accuracy (SNR = {}): {}".format(snr, cor / (cor + ncor)))
            self.acc[snr] = 1.0 * cor / (cor + ncor)

    def plot_accuracy_curves(self):
        '''
        Overall Accuracy Curve
        '''
        plt.figure()
        plt.title(
            'Overall Accuracy Curve - {} ({},{})CV: {}_{}'.format(self.mod_name, self.row, self.col, self.cross_val,
                                                                  self.drop))
        plt.plot(self.epochs, self.best_acc, label='training accuracy')
        plt.plot(self.epochs, self.best_val_acc, label='val_accuracy')
        plt.legend()
        plt.savefig(
            'data/accuracy_curves/ac_{}_{}_{}_{}_{}'.format(self.mod_name, self.row, self.col, self.cross_val,
                                                            self.drop))
        plt.close()

    def plot_roc_curves(self):
        '''
        ROC Curve
        '''
        for snr in self.snrs_choice:
            self.y_score = self.grid_search.best_estimator_.model.model.predict(self.X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(len(self.classes)):
                fpr[i], tpr[i], _ = roc_curve(self.Y_test[:, i], self.y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(self.Y_test.ravel(), self.y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.classes))]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(self.classes)):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= len(self.classes)

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            lw = 2
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(len(self.classes)), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            title = " {} Receiver Operating Characteristic Multi-Class (Dim=({},{}), CV={})".format(self.mod_name,
                                                                                                        self.row, self.col,
                                                                                                        self.cross_val)
            plt.title(title)
            plt.legend(loc="lower right")
            plt.savefig(
                'data/roc/roc_{}_{}_{}_cv _{}_snr_{}_{}'.format(self.mod_name, self.row, self.col, self.cross_val, snr,
                                                                    self.drop))
            plt.close()

    def print_final_results(self):
        loss_string = " {} ({}: {} x {}) CV: {}\n SNRs = {}\n Epochs: {}\n Loss: {}\n Accuracy: {}\n Validation Loss: {}\n Validation Accuracy: {}".format(
            self.mod_type, self.mod_name, self.row, self.col, self.cross_val, self.snrs_choice, self.epochs[-1],
            self.best_loss[-1], self.best_acc[-1], self.best_val_loss[-1], self.best_val_acc[-1])
        print(loss_string)
        for i in range(len(self.acc_string)):
            print(self.acc_string[i])


if __name__ == '__main__':
    print("The digital_benchmark.py script has been loaded.")
    # Set the seed for reproducibility
    #seed = np.random.randint(1000, size=1)
    digital_training_loss_results = []
    digital_training_accuracy_results = []
    analog_training_loss_results = []
    analog_training_loss_results = []
    seed = 711
    """
    Training Demo: Digital Modulation (2 x 128)
    """
    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK', 'PAM4', 'GFSK']  # ['AM-DSB', 'WBFM']
    snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    parameters = {
        'nodes': [[500, 500]],
        'act_func': ['relu'],
        'weights': ['lecun_uniform'],
        'o_weights': ['lecun_normal'],
        'epochs': [100],
        'batch_size': [1024],
        'optimizer': ['adagrad'],
        'in_shp': [[2, 128]],  # current digital modulation structure
        'classes': [['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK', 'PAM4', 'GFSK']]  # digital modulation classes
    }
    rows = 2
    cols = 128
    digital_modulation_recognition = SignalClassifier(rows, cols, cnn_with_dropout, parameters, 2, 'with_dropout',
                                                      'CNN', 'digital', mods, snrs, seed)
    digital_modulation_recognition.build_cv_model()
    digital_modulation_recognition.train_cv_model()
    digital_modulation_recognition.plot_loss_curves()
    digital_modulation_recognition.plot_overall_confusion_matrix()
    digital_modulation_recognition.plot_individual_confusion_matrices()
    digital_modulation_recognition.plot_accuracy_curves()
    digital_modulation_recognition.plot_roc_curves()
    digital_modulation_recognition.print_final_results()

    '''
    Training Demo: Analog Modulation (2 x 128)
    '''
    mods = ['WBFM', 'AM-SSB', 'AM-DSB']
    snrs = [2, 4, 6, 8, 10]
    parameters = {
        'nodes': [[500, 500]],
        'act_func': ['relu'],
        'weights': ['lecun_uniform'],
        'o_weights': ['lecun_normal'],
        'epochs': [100],
        'batch_size': [1024],
        'optimizer': ['adagrad'],
        'in_shp': [[2, 128]],  # current analog modulation structure
        'classes': [['WBFM', 'AM-SSB', 'AM-DSB']]  # analog modulation classes
    }
    rows = 2
    cols = 128
    analog_mod_rec = SignalClassifier(rows, cols, cnn_with_dropout, parameters, 10, 'with_dropout', 'CNN',
                                      'analog', mods, snrs,seed)
    analog_mod_rec.build_cv_model()
    analog_mod_rec.train_cv_model()
    analog_mod_rec.plot_loss_curves()
    analog_mod_rec.plot_overall_confusion_matrix()
    analog_mod_rec.plot_individual_confusion_matrices()
    analog_mod_rec.plot_accuracy_curves()
    analog_mod_rec.plot_roc_curves()
    analog_mod_rec.print_final_results()
else:
    print("The digital_benchmark.py script has been imported.")