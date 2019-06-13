import numpy as np
from keras.models import load_model
from sklearn import metrics
import time

def compute_metrics(y_true, y_pred, threshold):
    TP,TN,FP,FN = 0,0,0,0
    p = np.sum(y_true[:,0])/np.shape(y_true)[0]
    acc_r = 0
    for actual, prediction in zip(y_true,y_pred):
        #print(actual, prediction)
        if actual[0] == 1:
            if prediction[0] > threshold and prediction[1] < threshold:
                TN += 1
            elif prediction[0] < threshold and prediction[1] > threshold:
                FP += 1
            if np.random.rand() < p:
                acc_r += 1
        else:
            if prediction[0] > threshold and prediction[1] < threshold:
                FN += 1
            elif prediction[0] < threshold and prediction[1] > threshold:
                TP += 1
            if np.random.rand() >= p:
                acc_r += 1
        #print(TP, FP, TN, FN)
    print(TP+FP+TN+FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    acc_r /= y_true.shape[0]
    kappa = (acc-acc_r)/(1-acc_r)
    return acc, sens, spec, kappa

start_time = time.time()

x = np.memmap('data/x_gcn_train.npy',dtype=np.float32,shape=(400000,49,49,3))
y = np.memmap('data/y_train.npy',dtype=np.uint8,shape=(400000,2))

x = x[:,11:38,11:38,:]

model = load_model('gcn.keras')

y_pred = model.predict(x)
auc = metrics.roc_auc_score(y,y_pred)
print("AUC:\t", auc)
acc, sens, spec, kappa = compute_metrics(y,y_pred,0.5)
print("acc:\t", acc, "\nkappa:\t", kappa, "\nsens:\t", sens, "\nspec:\t", spec)

end_time = time.time()
print('Time taken: ', end_time-start_time)
