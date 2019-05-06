import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
import lightgbm as lgb
import catboost as cat
from mlxtend.feature_selection import SequentialFeatureSelector

from matplotlib import pyplot as plt
import seaborn as sns


class Stacking(BaseEstimator):
    def __init__(self, models, ens_model, p=0.3, predict_type='predict_proba', seed=127):
        self.models = models
        self.ens_model = ens_model
        self.n = len(models)
        self.valid = None
        self.predict_type = predict_type
        self.p = p
        self.random_state = seed

    def fit(self, X, y):
        train, valid, y_train, y_valid = train_test_split(X, y, test_size=self.p, random_state=self.random_state)

        self.valid = np.zeros((valid.shape[0], self.n))
        for t, clf in enumerate(self.models):
            # clf.fit(train, y_train)
            if self.predict_type == 'predict_proba':
                self.valid[:, t] = clf.predict_proba(valid)[:, 1]
            elif self.predict_type == 'predict':
                self.valid[:, t] = clf.predict(valid)
        self.ens_model.fit(self.valid, y_valid)
        return self

    def predict(self, X):
        X_meta = np.zeros((X.shape[0], self.n))
        for t, clf in enumerate(self.models):
            if self.predict_type == 'predict_proba':
                X_meta[:, t] = clf.predict_proba(X)[:, 1]
            elif self.predict_type == 'predict':
                X_meta[:, t] = clf.predict(X)
        a = self.ens_model.predict(X_meta)
        return a


    def predict_proba(self, X):
        X_meta = np.zeros((X.shape[0], self.n))
        for t, clf in enumerate(self.models):
            if self.predict_type == 'predict_proba':
                X_meta[:, t] = clf.predict_proba(X)[:, 1]
            elif self.predict_type == 'predict':
                X_meta[:, t] = clf.predict(X)
        a = self.ens_model.predict_proba(X_meta)
        return a


def hist_features_distrib(pos_features, neg_feature):
    x = pos_features.index
    plt.figure(figsize=(14, 10))
    plt.plot(x, neg_feature.values, 'r')
    plt.plot(x, pos_features.values, 'g')
    plt.show()
    # plt.savefig('plot_features_distrib.png')


def feature_selection(X, y,  method=1, k_features=5, save_params=False, seed=127):
    logit = LogisticRegression(C=1, random_state=seed, solver='liblinear')

    if method == 1:
        rfe = RFE(logit, n_features_to_select=k_features, verbose=2)
        rfe.fit(X, y)
        if save_params:
            with open('rfe.pkl', 'wb') as file:
                pickle.dump(rfe, file, pickle.HIGHEST_PROTOCOL)
        return rfe
    elif method == 2:
        sfs = SequentialFeatureSelector(logit, cv=0, k_features=k_features,
                                        forward=False, scoring='roc_auc',
                                        verbose=2, n_jobs=-1)
        sfs.fit(X, y)
        if save_params:
            with open('sfs.pkl', 'wb') as file:
                pickle.dump(sfs, file, pickle.HIGHEST_PROTOCOL)
        return sfs


def data_preprocess():
    train_df = pd.read_csv('../../DataSets/flight-delays-fall-2018/flight_delays_train.csv')
    test_df = pd.read_csv('../../DataSets/flight-delays-fall-2018/flight_delays_test.csv')

    y_train = train_df['dep_delayed_15min']
    y_train = y_train.map({'Y': 1, 'N': 0})
    train_df = train_df.drop(['dep_delayed_15min'], axis=1)

    # preprocessing Merged Train and Test data
    df = pd.concat([train_df, test_df], sort=False)
    trn_len = train_df.shape[0]

    date_columns = ['Month', 'DayofMonth', 'DayOfWeek']
    cat_columns = ['UniqueCarrier', 'Origin', 'Dest']

    for column in date_columns:
        df.loc[:, column] = df.loc[:, column].str.split('-').str[1].astype(int)

    df['DepTime_hour'] = df['DepTime'] // 100
    df['DepTime_hour'] = df['DepTime_hour'].apply(lambda t: t if t < 24 else t - 24)
    df['DepTime_minute'] = df['DepTime'] % 100
    df = df.drop(['DepTime'], axis=1)

    df = pd.get_dummies(data=df, columns=cat_columns)

    df['morning'] = ((df['DepTime_hour'] >= 6) & (df['DepTime_hour'] < 12)).astype('int')
    df['day'] = ((df['DepTime_hour'] >= 12) & (df['DepTime_hour'] < 18)).astype('int')
    df['evening'] = ((df['DepTime_hour'] >= 18) & (df['DepTime_hour'] < 24)).astype('int')
    df['night'] = ((df['DepTime_hour'] >= 0) & (df['DepTime_hour'] < 6)).astype('int')
    df['is_weekend'] = (df['DayOfWeek'] > 5).astype(int)

    df['winter'] = (df['Month'].isin([12, 1, 2])).astype('int')
    df['spring'] = (df['Month'].isin([3, 4, 5])).astype('int')
    df['summer'] = (df['Month'].isin([6, 7, 8])).astype('int')
    df['autumn'] = (df['Month'].isin([9, 10, 11])).astype('int')

    df['month_x'] = (df['Month'].isin([12, 6, 7])).astype('int')
    df['month_y'] = (df['Month'].isin([4, 5, 9, 2])).astype('int')

    df['week_high'] = (df['DayOfWeek'].isin([4, 5, 1, 7])).astype('int')
    df['week_low'] = (df['DayOfWeek'].isin([6, 2, 3])).astype('int')

    df = pd.get_dummies(data=df, columns=date_columns)

    # Back split data
    X_train = df[:trn_len]
    X_test = df[trn_len:]

    return X_train, y_train, X_test


def grid_search(X, y, seed):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # CatBoost
    cat_grid = {'iterations': np.arange(50, 500, 30),
                'depth': np.arange(4, 10, 2),
                'learning_rate': [0.01, 0.03, 0.1, 0.3]
                }
    catboost = cat.CatBoostClassifier(random_seed=seed)
    grid_cat = GridSearchCV(catboost, param_grid=cat_grid, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_cat.fit(X, y)
    print('CatBoost = {0}, {1}'.format(grid_cat.best_score_, grid_cat.best_params_))

    # LGBM
    lgbm_params = {'n_estimators': [50, 75, 100, 150],
                   'bagging_fraction': [0.8, 0.9],
                   'min_data_in_leaf': [0, 5, 15, 30],
                   'max_depth': [2, 3, 5, 7, 10],
                   'learning_rate': [0.05, 0.1]
                   }
    lgbm = lgb.LGBMClassifier()
    grid_lgbm = GridSearchCV(lgbm, param_grid=lgbm_params, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_lgbm.fit(X, y)
    print('LGBM = {0}, {1}'.format(grid_lgbm.best_score_, grid_lgbm.best_params_))

    # ax = lgb.plot_importance(lgbm, max_num_features=20)
    # plt.show()

    # Logistic Regression
    logit_params = {'C': np.logspace(-3, 3, 7)}
    logit = LogisticRegression(C=1, random_state=seed, solver='liblinear')
    grid_logit = GridSearchCV(logit, param_grid=logit_params, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_logit.fit(X, y)
    print('Logistic regression = {0}, {1}'.format(grid_logit.best_score_, grid_logit.best_params_))

    # KNN
    knn_params = {'n_neighbors': [3, 5, 6, 7, 8, 9, 10]}
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, param_grid=knn_params, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_knn.fit(X, y)
    print('KNN = {0}, {1}'.format(grid_knn.best_score_, grid_knn.best_params_))

    # Random Forest
    forest_params = {'n_estimators': [5, 10, 30, 50, 100],
                     'max_depth': [2, 3, 5, 7, 10],
                     'max_features': [10, 30, 50, 100],
                     'min_samples_leaf': [3, 5, 7, 9, 10]
                     }
    rand_forest = RandomForestClassifier()
    grid_forest = GridSearchCV(rand_forest, param_grid=forest_params, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_forest.fit(X, y)
    print('Random Forest = {0}, {1}'.format(grid_forest.best_score_, grid_forest.best_params_))


def train(X, y, seed=127):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=seed, shuffle=True)

    # CatBoost
    catboost = cat.CatBoostClassifier(iterations=150,
                                      learning_rate=0.05,
                                      max_depth=10,
                                      random_seed=seed,
                                      silent=True)
    score_cat = cross_val_score(catboost, X, y, scoring='roc_auc', cv=skf, verbose=0)
    catboost.fit(X_train, y_train,
                 eval_set=(X_val, y_val),
                 use_best_model=True,
                 early_stopping_rounds=20)
    print('CatBoost = {0}'.format(np.mean(score_cat)))

    # LGBM
    lgbm = lgb.LGBMClassifier(objective='binary',
                              learning_rate=0.1,
                              n_estimators=150,
                              num_leaves=180,
                              max_depth=10,
                              random_state=seed,
                              silent=True)
    score_lgbm = cross_val_score(lgbm, X, y, scoring='roc_auc', cv=skf, verbose=0)
    lgbm.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, eval_metric='auc')
    print('LGBM = {0}'.format(np.mean(score_lgbm)))

    # Random Forest
    rand_forest = RandomForestClassifier(n_estimators=30,
                                         min_samples_leaf=3)
    score_forest = cross_val_score(rand_forest, X, y, scoring='roc_auc', cv=skf)
    rand_forest.fit(X, y)
    print('Random Forest = {0}'.format(np.mean(score_forest)))

    # Logistic Regression
    logit = LogisticRegression(C=0.01, random_state=seed, solver='liblinear')
    score_logit = cross_val_score(logit, X, y, scoring='roc_auc', cv=skf)
    logit.fit(X, y)
    print('Logistic regression = {0}'.format(np.mean(score_logit)))

    # knn = KNeighborsClassifier(n_neighbors=5)
    # score_knn = cross_val_score(knn, X, y, scoring='roc_auc', cv=skf)
    # knn.fit(X, y)
    # print('KNN = {0}'.format(np.mean(score_knn)))

    return catboost, lgbm, rand_forest, logit


def main():
    seed = 17
    run_grid_search = False
    is_feature_selection = False
    load_feature_selector = False

    # Preprocessing
    X_train, y_train, X_test = data_preprocess()

    # Training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_fold, X_val, y_train_fold, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                                random_state=seed, shuffle=True)

    # Feature Selection
    if is_feature_selection:
        if load_feature_selector:
            # load params
            with open('rfe.pkl', 'rb') as file:
                feature_selector = pickle.load(file)
        else:
            feature_selector = feature_selection(X_val, y_val, save_params=True, k_features=550, seed=seed)

        X_train_fold = feature_selector.transform(X_train_fold)
        X_val = feature_selector.transform(X_val)
        X_test = feature_selector.transform(X_test)

    # grid search
    if run_grid_search:
        grid_search(X_train_fold, y_train_fold, seed)
    else:
        # train models
        catboost, lgbm, rand_forest, logit = train(X_train_fold, y_train_fold, seed)

        ens_model = Ridge()
        ens = Stacking([catboost, lgbm, rand_forest, logit], ens_model, p=0.3,
                       predict_type='predict_proba', seed=seed)
        ens.fit(X_val, y_val)
        preds_val_ens = np.abs(ens.predict(X_val))
        print('Ensamble Val = {}'.format(roc_auc_score(y_val, preds_val_ens)))

        # submit
        test_preds_lgb = lgbm.predict_proba(X_test)[:, 1]
        test_preds_ens = np.abs(ens.predict(X_test))
        pd.Series(test_preds_lgb, name='dep_delayed_15min').to_csv('simple_lgbm.csv', index_label='id', header=True)
        pd.Series(test_preds_ens, name='dep_delayed_15min').to_csv('ensamble.csv', index_label='id', header=True)


if __name__ == '__main__':
    main()
