_author_ = "Ranajit_Roy"

import import_data as imp_data
import numpy as np
import create_model as cm
import  load_save_model as lsm

train_imgs, train_labels = imp_data.train_data()
test_imgs, test_labels = imp_data.test_data()

NNmodel = cm.TwoHLNN(800, 100)
folder = str(NNmodel.hu_1) + '_' + str(NNmodel.hu_2)

# lsm.load_untrained_model(NNmodel)
# lsm.save_untrained_model(NNmodel)

lsm.load_model(NNmodel, folder)

prev_acc = NNmodel.accuracy

NNmodel.train(train_imgs, train_labels, lr_rate=0.5, iter=200)

acc = NNmodel.test(test_imgs, test_labels)

if acc > prev_acc:
    lsm.save_model(NNmodel)
