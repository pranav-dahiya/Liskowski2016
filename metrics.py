import numpy as np
from keras.models import load_model
from sklearn import metrics


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
    acc = (TP+TN)/(TP+TN+FP+FN)
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    acc_r /= y_true.shape[0]
    print(acc_r)
    kappa = (acc-acc_r)/(1-acc_r)
    return acc, sens, spec, kappa


x = np.memmap('data/x_test.npy',dtype=np.uint8,shape=(400000,27,27,3))
y = np.memmap('data/y_test.npy',dtype=np.uint8,shape=(400000,2))

model = load_model('plain.keras')

y_pred = model.predict(x)
acc, sens, spec, kappa = compute_metrics(y,y_pred,0.5)
print("acc:\t", acc, "\nsens:\t", sens, "\nspec:\t", spec, "\nkappa:\t", kappa)
auc = metrics.roc_auc_score(y,y_pred)
print("AUC:\t", auc)
