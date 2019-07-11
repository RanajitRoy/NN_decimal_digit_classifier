import import_data as imp_data
import create_model as cm
import load_save_model as lsm

train_imgs, train_labels = imp_data.train_data()
test_imgs, test_labels = imp_data.test_data()

NNmodel = cm.TwoHLNN(500, 200)
# NNmodel = cm.ThreeHLNN(500, 200, 80, initialize=0.01)

folder = str(NNmodel.hu_1) + '_' + str(NNmodel.hu_2)
if NNmodel.layers == 3:
    folder = folder + '_' + str(NNmodel.hu_3)

# lsm.load_untrained_model(NNmodel, folder)
# lsm.save_untrained_model(NNmodel)

lsm.load_model(NNmodel, folder)

prev_acc = NNmodel.accuracy

# NNmodel.train(train_imgs, train_labels, lr_rate=0.5, maxiter=100, lmda=0.7)

acc = NNmodel.test(test_imgs, test_labels)

# if acc > prev_acc:
#     lsm.save_model(NNmodel)

prediction = NNmodel.predict()

print("Prediction ->", prediction)
