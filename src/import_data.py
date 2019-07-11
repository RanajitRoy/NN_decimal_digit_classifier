_author_ = "Ranajit_Roy"

import idx2numpy as id2np
import os


def train_data():
    train_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'trainDB'

    train_imgs = id2np.convert_from_file(train_dir + os.sep + 'train-images.idx3-ubyte') / 255

    train_labels = id2np.convert_from_file(train_dir + os.sep + 'train-labels.idx1-ubyte')

    return train_imgs, train_labels


def test_data():
    test_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'testDB'

    test_imgs = id2np.convert_from_file(test_dir + os.sep + 't10k-images.idx3-ubyte') / 255

    test_labels = id2np.convert_from_file(test_dir + os.sep + 't10k-labels.idx1-ubyte')

    return test_imgs, test_labels
