import numpy as np
from keras.models import load_model
from sklearn import metrics
import time
import tensorflow as tf
from keras import backend as K

CPU = False
GPU = True

num_cores = 10

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


def compute_metrics(y_true, y_pred):
    TP,TN,FP,FN = 0,0,0,0
    p = np.sum(y_true[:,0])/y_true.shape[0]
    acc_r = 0
    for actual, prediction in zip(y_true,y_pred):
        #print(actual, prediction)
        if actual[0] == 1:
            if np.argmax(prediction) == 0:
                TN += 1
            else:
                FP += 1
            if np.random.rand() < p:
                acc_r += 1
        else:
            if np.argmax(prediction) == 1:
                TP += 1
            else:
                FN += 1
            if np.random.rand() >= p:
                acc_r += 1
        #print(TP, FP, TN, FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    acc_r /= y_true.shape[0]
    kappa = (acc-acc_r)/(1-acc_r)
    return acc, sens, spec, kappa

start_time = time.time()

x = np.memmap('Data-2/x_test.npy', dtype=np.uint8, shape=(400000,27,27,3))
y = np.memmap('Data-2/y_test.npy', dtype=np.uint8, shape=(400000,2))

model = load_model('Data-2/nopool.keras')

y_pred = model.predict(x)
auc = metrics.roc_auc_score(y, y_pred)
print("AUC:\t", auc)
acc, sens, spec, kappa = compute_metrics(y,y_pred)
print("acc:\t", acc, "\nkappa:\t", kappa, "\nsens:\t", sens, "\nspec:\t", spec)

end_time = time.time()
print('Time taken: ', end_time-start_time)
