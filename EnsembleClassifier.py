import time
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from FeatureDescriptor import ReadFileFromFasta, FeatureGenerator
from FeatureSelection import EFISS_ES_FeatureSelection
from MLModel import MetricsCalculate, TrainBaselineMLModel

def Ensemble_Classifier_CV(train_x, train_y, test_x, test_y, cv_fold, index_arr:dict, model_obj_arr, vote, count):

    ml_dict = {'NB': GaussianNB(),
               'RF': RandomForestClassifier(random_state=100),
               'GBDT': GradientBoostingClassifier(random_state=100),
               'SVM': svm.SVC(random_state=100, probability=True)}

    model_weight = {}
    test_x_new = {}
    for key in model_obj_arr:
        for test_dataset in test_x.keys():
            test_x_new[test_dataset] = test_x[test_dataset][:, index_arr[key]]
        clf_arr = TrainBaselineMLModel(train_x[:, index_arr[key]], train_y, test_x_new, test_y, key, 10, test=False)[5]
        model_weight[key] = clf_arr[3]

    arr_valid = []
    folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x, train_y)

    for i, (train, valid) in enumerate(folds):

        train_X, train_Y = train_x[train], train_y[train]
        valid_X, valid_Y = train_x[valid], train_y[valid]

        if vote == 'hard':
            predict_valid_y_class = []
            predict_valid_y_pro_1 = []

            valid_y_class_arr = {}
            valid_y_pro_1_arr = {}
            valid_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_X[:, index_arr[key]], train_Y)
                valid_y_class_arr[key] = np.array(model.predict(valid_X[:, index_arr[key]]))
                valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]
                valid_y_pro_0_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 0]

            for index in range(len(valid_X)):

                valid_y_class_1 = 0
                valid_y_pro_1 = 0
                valid_y_pro_0 = 0

                for key in model_obj_arr:
                    valid_y_class_1 = valid_y_class_1 + valid_y_class_arr[key][index]
                    valid_y_pro_1 = valid_y_pro_1 + valid_y_pro_1_arr[key][index]
                    valid_y_pro_0 = valid_y_pro_0 + valid_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if valid_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_valid_y_class.append(1)

                    if valid_y_class_1 == (len(model_obj_arr)) / 2:
                        if valid_y_pro_1 >= valid_y_pro_0:
                            predict_valid_y_class.append(1)
                        if valid_y_pro_1 < valid_y_pro_0:
                            predict_valid_y_class.append(0)

                    if valid_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_valid_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if valid_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_valid_y_class.append(1)
                    else:
                        predict_valid_y_class.append(0)

                predict_valid_y_pro_1.append(valid_y_pro_1 / len(model_obj_arr))

            metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
            arr_valid.append(metrics_value)

        if vote == 'soft':
            predict_valid_y_class = []
            predict_valid_y_pro_1 = []

            valid_y_pro_1_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_X[:, index_arr[key]], train_Y)
                valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]

            for index in range(len(valid_X)):

                valid_y_pro_1 = 0

                for key in model_obj_arr:
                    valid_y_pro_1 = valid_y_pro_1 + valid_y_pro_1_arr[key][index]

                valid_y_pro_1 = valid_y_pro_1 / len(model_obj_arr)

                if valid_y_pro_1 >= 0.5:
                    predict_valid_y_class.append(1)

                if valid_y_pro_1 < 0.5:
                    predict_valid_y_class.append(0)

                predict_valid_y_pro_1.append(valid_y_pro_1)

            metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
            arr_valid.append(metrics_value)

        if vote == 'devs':
            predict_valid_y_class = []
            predict_valid_y_pro_1 = []

            valid_y_class_arr = {}
            valid_y_pro_1_arr = {}
            valid_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_X[:, index_arr[key]], train_Y)
                valid_y_class_arr[key] = np.array(model.predict(valid_X[:, index_arr[key]]))
                valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]
                valid_y_pro_0_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 0]

            for index in range(len(valid_X)):

                valid_y_class_1 = 0
                valid_y_pro_1 = 0
                valid_y_pro_0 = 0

                for key in model_obj_arr:
                    valid_y_class_1 = valid_y_class_1 + valid_y_class_arr[key][index]
                    valid_y_pro_1 = valid_y_pro_1 + valid_y_pro_1_arr[key][index]
                    valid_y_pro_0 = valid_y_pro_0 + valid_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if valid_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_valid_y_class.append(1)

                    if valid_y_class_1 == (len(model_obj_arr)) / 2:
                        if valid_y_pro_1 >= valid_y_pro_0:
                            predict_valid_y_class.append(1)
                        if valid_y_pro_1 < valid_y_pro_0:
                            predict_valid_y_class.append(0)

                    if valid_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_valid_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if valid_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_valid_y_class.append(1)
                    else:
                        predict_valid_y_class.append(0)

                predict_valid_y_pro_1.append(valid_y_pro_1 / len(model_obj_arr))

            for index in range(len(valid_X)):

                valid_y_pro_0 = 0
                valid_y_pro_1 = 0

                flag = 0

                model_u = {}

                DC_dict = {}

                for key in model_obj_arr:
                    model_u[key] = abs(valid_y_pro_1_arr[key][index] - valid_y_pro_0_arr[key][index])

                for key in model_obj_arr:
                    if model_u[key] < 0.1:
                        flag += 1
                    DC_dict[key] = model_weight[key] + model_u[key]

                DC_sum = sum(DC_dict.values())

                for key in model_obj_arr:
                    DC_dict[key] = DC_dict[key] / DC_sum

                model_obj_arr_ranked = sorted(DC_dict.items(), key=lambda d: d[1], reverse=True)

                if flag >= (len(model_obj_arr) / 2):
                    for model in model_obj_arr_ranked[:count]:
                        valid_y_pro_0 = valid_y_pro_0 + model[1] * valid_y_pro_0_arr[model[0]][index]
                        valid_y_pro_1 = valid_y_pro_1 + model[1] * valid_y_pro_1_arr[model[0]][index]

                    if valid_y_pro_1 >= valid_y_pro_0:
                        predict_valid_y_class[index] = 1

                    if valid_y_pro_1 < valid_y_pro_0:
                        predict_valid_y_class[index] = 0

                    predict_valid_y_pro_1[index] = valid_y_pro_1

            metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
            arr_valid.append(metrics_value)

    valid_scores = np.around(np.array(arr_valid).sum(axis=0) / cv_fold, 3)
    print("validation_dataset_scores: ", valid_scores)

def Ensemble_Classifier_Pred(train_x, train_y, test_x, test_y, index_arr: dict, model_obj_arr, vote, count):

    ml_dict = {'NB': GaussianNB(),
               'RF': RandomForestClassifier(random_state=100),
               'GBDT': GradientBoostingClassifier(random_state=100),
               'SVM': svm.SVC(random_state=100, probability=True)}
    if vote == 'hard':

        for test_dataset in test_x.keys():
            print(test_dataset)
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + test_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_test_y_class.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) / 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_test_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_test_y_class.append(1)
                    else:
                        predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1 / len(model_obj_arr))

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'soft':
        for test_dataset in test_x.keys():
            print(test_dataset)
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_pro_1_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]

            for index in range(len(test_x[test_dataset])):

                test_y_pro_1 = 0

                for key in model_obj_arr:
                    test_y_pro_1 = test_y_pro_1 + test_y_pro_1_arr[key][index]

                test_y_pro_1 = test_y_pro_1 / len(model_obj_arr)

                if test_y_pro_1 >= 0.5:
                    predict_test_y_class.append(1)

                if test_y_pro_1 < 0.5:
                    predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'devs':

        model_weight = {}

        test_x_new = {}

        for key in model_obj_arr:
            for test_dataset in test_x.keys():
                test_x_new[test_dataset] = test_x[test_dataset][:, index_arr[key]]
            clf_arr = TrainBaselineMLModel(train_x[:, index_arr[key]], train_y, test_x_new, test_y, key, 10, test=False)[5]
            model_weight[key] = clf_arr[3]

        for test_dataset in test_x.keys():

            print(test_dataset)

            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + test_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_test_y_class.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) / 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_test_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_test_y_class.append(1)
                    else:
                        predict_test_y_class.append(0)
                predict_test_y_pro_1.append(test_y_pro_1 / len(model_obj_arr))

            for index in range(len(test_x[test_dataset])):

                test_y_pro_0 = 0
                test_y_pro_1 = 0

                flag = 0

                model_u = {}

                DC_dict = {}

                for key in model_obj_arr:
                    model_u[key] = abs(test_y_pro_1_arr[key][index] - test_y_pro_0_arr[key][index])

                for key in model_obj_arr:
                    if model_u[key] < 0.1:
                        flag += 1
                    DC_dict[key] = model_weight[key] + model_u[key]

                DC_sum = sum(DC_dict.values())

                for key in model_obj_arr:
                    DC_dict[key] = DC_dict[key] / DC_sum

                model_obj_arr_ranked = sorted(DC_dict.items(), key=lambda d: d[1], reverse=True)

                if flag >= (len(model_obj_arr) / 2):
                    for model in model_obj_arr_ranked[:count]:
                        test_y_pro_0 = test_y_pro_0 + model[1] * test_y_pro_0_arr[model[0]][index]
                        test_y_pro_1 = test_y_pro_1 + model[1] * test_y_pro_1_arr[model[0]][index]

                    if test_y_pro_1 >= test_y_pro_0:
                        predict_test_y_class[index] = 1

                    if test_y_pro_1 < test_y_pro_0:
                        predict_test_y_class[index] = 0

                    predict_test_y_pro_1[index] = test_y_pro_1

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

if __name__ == '__main__':

    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()

    ath_train_seq = ReadFileFromFasta("data/ath_train_320条.txt")
    ath_imbalanced_test_seq = ReadFileFromFasta("data/independent_testing_data_set_1.txt")
    hybrid_species = ReadFileFromFasta("data/independent_testing_data_set_2.txt")

    # 特征生成
    train_features, train_labels, train_features_name = FeatureGenerator(ath_train_seq)
    independent_test1_features, independent_test1_labels, independent_test1_features_name = FeatureGenerator(ath_imbalanced_test_seq)
    independent_test2_features, independent_test2_labels, independent_test2_features_name = FeatureGenerator(hybrid_species)

    index_arr = {}

    for key in train_features.keys():

        if key == 'ESMER5':
            print('key: ', key)
            filtered_feature = EFISS_ES_FeatureSelection(train_features[key], train_labels, 1)
            print('filtered_feature: ', filtered_feature)

            index_arr['NB'] = filtered_feature[: 32]
            index_arr['SVM'] = filtered_feature[: 11]
            index_arr['RF'] = filtered_feature[: 31]
            index_arr['GBDT'] = filtered_feature[: 46]

            train_features_fs = train_features[key]

            test_x = {'independent_test1': independent_test1_features[key],
                      'independent_test2': independent_test2_features[key]}
            test_y = {'independent_test1': independent_test1_labels,
                      'independent_test2': independent_test2_labels}

            model_obj_arr = ['NB', 'SVM', 'RF', 'GBDT']

            Ensemble_Classifier_CV(train_features_fs, train_labels, test_x, test_y, 10, index_arr, model_obj_arr, 'devs', 3)
            print('--------------------------------------------')
            Ensemble_Classifier_Pred(train_features_fs, train_labels, test_x, test_y, index_arr, model_obj_arr, 'devs', 3)

    print(time.time() - start)
    print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))