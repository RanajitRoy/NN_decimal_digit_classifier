import numpy as np
import os


def save_model(model):
    folder = str(model.hu_1) + '_' + str(model.hu_2)
    if model.layers == 3:
        folder = folder + '_' + str(model.hu_3)
    model_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'model' + os.sep + folder + os.sep
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    np.save(model_path + 'Theta1', model.Theta1)
    np.save(model_path + 'Theta2', model.Theta2)
    np.save(model_path + 'Theta3', model.Theta3)
    np.save(model_path + 'accuracy', np.array(model.accuracy))
    if model.layers == 3:
        np.save(model_path + 'Theta4', model.Theta4)
    print('Model Saved!')


def save_untrained_model(model):
    folder = str(model.hu_1) + '_' + str(model.hu_2)
    if model.layers == 3:
        folder = folder + '_' + str(model.hu_3)
    model_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'model' + os.sep + folder + os.sep
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    np.save(model_path + 'ini_Theta1', model.Theta1)
    np.save(model_path + 'ini_Theta2', model.Theta2)
    np.save(model_path + 'ini_Theta3', model.Theta3)
    if model.layers == 3:
        np.save(model_path + 'ini_Theta4', model.Theta4)
    print('Untrained Model Saved!')


def load_model(model, folder):
    model_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'model' + os.sep + folder + os.sep
    if not os.path.isdir(model_path):
        print("Invalid address ->", model_path)
    model.Theta1 = np.load(model_path + 'Theta1.npy')
    model.Theta2 = np.load(model_path + 'Theta2.npy')
    model.Theta3 = np.load(model_path + 'Theta3.npy')
    model.accuracy = np.load(model_path + 'accuracy.npy')
    model.hu_1 = model.Theta1.shape[0]
    model.hu_2 = model.Theta2.shape[0]
    model.inp_sz = model.Theta1.shape[1] - 1
    model.out_sz = model.Theta3.shape[0]
    if model.layers == 3:
        model.Theta4 = np.load(model_path + 'Theta4.npy')
        model.hu_3 = model.Theta3.shape[0]
        model.out_sz = model.Theta4.shape[0]
    print('Model Loaded!')


def load_untrained_model(model, folder):
    model_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'model' + os.sep + folder + os.sep
    if not os.path.isdir(model_path):
        print("Invalid address ->", model_path)
    model.Theta1 = np.load(model_path + 'ini_Theta1.npy')
    model.Theta2 = np.load(model_path + 'ini_Theta2.npy')
    model.Theta3 = np.load(model_path + 'ini_Theta3.npy')
    model.hu_1 = model.Theta1.shape[0]
    model.hu_2 = model.Theta2.shape[0]
    model.inp_sz = model.Theta1.shape[1] - 1
    model.out_sz = model.Theta3.shape[0]
    if model.layers == 3:
        model.Theta4 = np.load(model_path + 'ini_Theta4.npy')
        model.hu_3 = model.Theta3.shape[0]
        model.out_sz = model.Theta4.shape[0]
    print('Untrained Model Loaded!')
