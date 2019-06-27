import numpy as np
from keras.models import load_model
from sklearn import metrics
import time
import tensorflow as tf
from keras import backend as K
'''
config = tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10, \
                        allow_soft_placement=True, device_count = {'CPU': 1, 'GPU': 0})
sess = tf.Session(config=config)
K.set_session(sess)
'''
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


def print_metrics(model_name, x, y):
    model = load_model('Data-5/'+model_name+'.keras')
    y_pred = model.predict(x)
    auc = metrics.roc_auc_score(y, y_pred)
    acc, sens, spec, kappa = compute_metrics(y,y_pred)
    with open('Data-6/metrics.log', 'a') as f:
        f.write(model_name+"\nAUC:\t"+str(auc)+"\nAcc:\t"+str(acc)+"\nKappa:\t"\
                    +str(kappa)+"\nSens:\t"+str(sens)+"\nSpec:\t"+str(spec)+"\n")

num = '6'

x = np.memmap('Data-'+num+'/x_test.npy', dtype=np.uint8, mode='r', shape=(400000, 27, 27, 3))
y = np.memmap('Data-'+num+'/y_test.npy', dtype=np.uint8, mode='r', shape=(400000, 2))

print_metrics('plain', x, y)
print_metrics('balanced', x, y)
print_metrics('nopool', x, y)

del x
x = np.memmap('Data-'+num+'/x_gcn_test.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
print_metrics('gcn', x, y)

del x
x = np.memmap('Data-'+num+'/x_zca_test.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
print_metrics('zca', x, y)
