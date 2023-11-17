import math
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, metrics
from sklearn.model_selection import StratifiedKFold

def MetricsCalculate(y_true_label, y_predict_label, y_predict_pro):
    metrics_value = []
    confusion = []
    tn, fp, fn, tp = metrics.confusion_matrix(y_true_label, y_predict_label).ravel()
    sn = round(tp / (tp + fn) * 100, 3) if (tp + fn) != 0 else 0
    sp = round(tn / (tn + fp) * 100, 3) if (tn + fp) != 0 else 0
    pre = round(tp / (tp + fp) * 100, 3) if (tp + fp) != 0 else 0
    acc = round((tp + tn) / (tp + fn + tn + fp) * 100, 3)
    mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 3) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    f1 = round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) != 0 else 0

    if y_predict_pro is not None:
        fpr, tpr, thresholds = roc_curve(y_true_label, y_predict_pro)
        precision, recall, thresholds = precision_recall_curve(y_true_label, y_predict_pro)

        auroc = auc(fpr, tpr)
        auprc = auc(recall, precision)

    if y_predict_pro is None:
        auroc = 0
        auprc = 0

    metrics_value.append(sn)
    metrics_value.append(sp)
    metrics_value.append(pre)
    metrics_value.append(acc)
    metrics_value.append(mcc)
    metrics_value.append(f1)
    metrics_value.append(auroc)
    metrics_value.append(auprc)

    confusion.append(tp)
    confusion.append(fn)
    confusion.append(tn)
    confusion.append(fp)

    return metrics_value, confusion

def TrainBaselineMLModel(train_x, train_y, test_x, test_y, model_obj, cv_fold, **kwargs):

    train_feature_class = []
    train_feature_pro = []
    train_y_new = []
    test_feature_class = {}
    test_feature_pro = {}
    species_feature_class = []
    species_feature_pro = []
    models = []
    arr_valid = []
    folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x, train_y)

    for i, (train, valid) in enumerate(folds):

        ml_dict = {'NB': GaussianNB(),
                   'RF': RandomForestClassifier(random_state=100),
                   'GBDT': GradientBoostingClassifier(random_state=100),
                   'SVM': svm.SVC(random_state=100, probability=True)}

        train_X, train_Y = train_x[train], train_y[train]
        valid_X, valid_Y = train_x[valid], train_y[valid]
        model = ml_dict[model_obj]
        model.fit(train_X, train_Y)
        models.append(model)

        predict_valid_y_class = model.predict(valid_X)
        train_feature_class.extend(predict_valid_y_class)
        predict_valid_y_pro = np.array(model.predict_proba(valid_X))[:, 1]
        train_feature_pro.extend(predict_valid_y_pro)

        train_y_new.extend(valid_Y)
        metrics_value, confusion = MetricsCalculate(valid_Y, predict_valid_y_class, predict_valid_y_pro)
        arr_valid.append(metrics_value)

    valid_scores = np.around(np.array(arr_valid).sum(axis=0) / cv_fold, 3)
    valid_scores_std = np.std(np.array(arr_valid), axis=0)

    if kwargs['test'] == True:
        print("validation_dataset_scores: ", valid_scores)

    for test_dataset in test_x.keys():
        for ml in models:
            species_feature_class.append(ml.predict(test_x[test_dataset]))
            species_feature_pro.append(np.array(ml.predict_proba(test_x[test_dataset]))[:, 1])
        test_feature_class[test_dataset] = np.around(np.array(species_feature_class).sum(axis=0) / cv_fold, 3)
        test_feature_pro[test_dataset] = np.around(np.array(species_feature_pro).sum(axis=0) / cv_fold, 3)
        species_feature_class.clear()
        species_feature_pro.clear()
        if kwargs['test'] == True:
            predict_y_class = np.where(test_feature_pro[test_dataset] >= 0.5, 1, 0)
            predict_y_pro = test_feature_pro[test_dataset]
            print(test_dataset + ':', MetricsCalculate(test_y[test_dataset], predict_y_class, predict_y_pro))
    return [np.array(train_feature_class), np.array(train_feature_pro), test_feature_class, test_feature_pro, np.array(train_y_new), np.array(valid_scores), np.array(arr_valid), valid_scores_std, models]
