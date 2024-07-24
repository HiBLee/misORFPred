import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

def ExtraTreesFeatureRanking(X, Y):

    model = ExtraTreesClassifier(random_state=100)
    model.fit(X, Y)
    importance = np.array(np.abs(model.feature_importances_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def GradientBoostingFeatureRanking(X, Y):

    model = GradientBoostingClassifier(random_state=100)
    model.fit(X, Y)
    importance = np.array(np.abs(model.feature_importances_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def EFISS_ES_FeatureSelection(X, Y, p=1):

    embedding_importance_arr = []
    embedding_rank_arr = []

    et_rank = []
    gbdt_rank = []

    et_indices, et_importance, et_importance_mean = ExtraTreesFeatureRanking(X, Y)
    gbdt_indices, gbdt_importance, gbdt_importance_mean = GradientBoostingFeatureRanking(X, Y)

    m = len(et_indices)

    et_indices = list(map(int, et_indices))
    gbdt_indices = list(map(int, gbdt_indices))

    for i in range(m):

        et_rank.append(et_indices.index(i) + 1)
        gbdt_rank.append(gbdt_indices.index(i) + 1)

    embedding_importance_arr.append(et_importance)
    embedding_importance_arr.append(gbdt_importance)
    embedding_rank_arr.append(et_rank)
    embedding_rank_arr.append(gbdt_rank)

    embedding_importance_average_arr = np.mean(embedding_importance_arr, axis=0)
    embedding_rank_average_arr = np.mean(embedding_rank_arr, axis=0)

    lnls_total_average = []
    for i in range(m):
        s = (m - embedding_rank_average_arr[i]) / m * embedding_importance_average_arr[i]
        lnls_total_average.append(s)

    feature_index = np.argsort(lnls_total_average)[::-1]
    num = int(np.round(len(feature_index) * p))
    if num == 0:
        num = 1
    feature_index = feature_index[:num]

    return feature_index
