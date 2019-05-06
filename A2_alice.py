import os
import pickle
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression


PATH_TO_DATA = '../../DataSets/catch-me/'
AUTHOR = 'Denis_Finogenov'

SEED = 17
N_JOBS = 1
NUM_TIME_SPLITS = 10  # for time-based cross-validation
SITE_NGRAMS = (1, 5)  # site ngrams for "bag of sites"
MAX_FEATURES = 50000  # max features for "bag of sites"
BEST_LOGIT_C = 3.5998 # precomputed tuned C for logistic regression
UNDERSAMPLING_COUNT = 150000


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_data(path_to_train, path_to_test):
    times = ['time%s' % i for i in range(1, 11)]
    train_df = pd.read_csv(path_to_train, index_col='session_id', parse_dates=times)

    train_df_1 = train_df[train_df['target'] == 1]
    train_df_0 = train_df[train_df['target'] == 0].sample(UNDERSAMPLING_COUNT)
    train_df = pd.concat([train_df_1, train_df_0])
    test_df = pd.read_csv(path_to_test, index_col='session_id', parse_dates=times)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')
    return train_df, test_df


def prepare_sparse_features(train_df, test_df, path_to_site_dict, vectorizer_params):
    times = ['time%s' % i for i in range(1, 11)]
    # read site -> id mapping provided by competition organizers
    with open(path_to_site_dict, 'rb') as f:
        site2id = pickle.load(f)
    # create an inverse id _> site mapping
    id2site = {v: k for (k, v) in site2id.items()}
    # we treat site with id 0 as "unknown"
    id2site[0] = 'unknown'

    # Transform data into format which can be fed into TfidfVectorizer
    # This time we prefer to represent sessions with site names, not site ids.
    # It's less efficient but thus it'll be more convenient to interpret model weights.
    sites = ['site%s' % i for i in range(1, 11)]
    train_sessions = train_df[sites].fillna(0).astype('int').apply(lambda row:
                                                                   ' '.join([id2site[i] for i in row]), axis=1).tolist()
    test_sessions = test_df[sites].fillna(0).astype('int').apply(lambda row:
                                                                 ' '.join([id2site[i] for i in row]), axis=1).tolist()
    # we'll tell TfidfVectorizer that we'd like to split data by whitespaces only
    # so that it doesn't split by dots (we wouldn't like to have 'mail.google.com'
    # to be split into 'mail', 'google' and 'com')
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    y_train = train_df['target'].astype('int').values

    # we'll need site visit times for further feature engineering
    train_times, test_times = train_df[times], test_df[times]

    return X_train, X_test, y_train, vectorizer, train_times, test_times


def get_top10(sites):
    site_dict = pd.Series(sites.values.flatten()).value_counts().sort_values(ascending=False)
    top_10 = site_dict.drop(0).iloc[0:10]
    return top_10, site_dict


def add_top10_feature(sites, top10):
    have_top_10 = np.zeros((sites.shape[0], 1), dtype=int)
    ind = 0
    for row in sites.values:
        unique = np.unique(row)
        for site_id in unique:
            if site_id in top10.index:
                have_top_10[ind] = 1
        ind += 1

    return have_top_10


def get_time_diff(row):
    time_length = row.shape[0] - 1
    time_diff = [0] * time_length
    i = 0
    while (i < time_length) and pd.notnull(row[i + 1]):
        time_diff[i] = (row[i + 1] - row[i]) / np.timedelta64(1, 's')
        i += 1
    return time_diff


def add_time_diff(times):
    time_diff = []
    for row in times.values:
        time_diff.append(get_time_diff(row))
    time_diff = np.log1p(np.array(time_diff).astype(float))
    return time_diff


def add_features(times, sites, X_sparse, scaler=None, top10=None, site_dict=None):
    new_features_df = pd.DataFrame()
    hour = times['time1'].apply(lambda ts: ts.hour)
    new_features_df['morning'] = ((hour >= 7) & (hour <= 11)).astype('int')
    new_features_df['day'] = ((hour >= 12) & (hour <= 18)).astype('int')
    new_features_df['evening'] = ((hour >= 19) & (hour <= 23)).astype('int')
    new_features_df['night'] = ((hour >= 0) & (hour <= 6)).astype('int')
    new_features_df['sess_duration'] = (times.max(axis=1) - times.min(axis=1)).astype('timedelta64[s]').astype('int')
    new_features_df['day_of_week'] = times['time1'].apply(lambda t: t.weekday())
    new_features_df['month'] = times['time1'].apply(lambda t: t.month)
    new_features_df['year_month'] = times['time1'].apply(lambda t: 100 * t.year + t.month) / 1e5
    new_features_df['holiday'] = (times['time1'].dt.dayofweek >= 5).astype(int)

    if top10 is None:
        top10, site_dict = get_top10(sites)
    new_features_df['have_top10'] = add_top10_feature(sites, top10)

    time_names = ['time_diff' + str(j) for j in range(1, 10)]
    time_diff = add_time_diff(times)
    for ind, column_name in enumerate(time_names):
        new_features_df[column_name] = time_diff[:, ind]

    if scaler is None:
        scaler = StandardScaler()
        new_features_scaled = scaler.fit_transform(new_features_df)
    else:
        new_features_scaled = scaler.transform(new_features_df)

    X = hstack([X_sparse, new_features_scaled])
    return X, scaler, top10, site_dict


with timer('Building sparse site features'):
    train_df, test_df = load_data(path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                                  path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'))
    X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = prepare_sparse_features(
        train_df, test_df, path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),
        vectorizer_params={'ngram_range': SITE_NGRAMS,
                           'max_features': MAX_FEATURES,
                           'tokenizer': lambda s: s.split()})

with timer('Building additional features'):
    sites = ['site%s' % i for i in range(1, 11)]
    train_sites = train_df[sites].fillna(0).astype('int')
    test_sites = test_df[sites].fillna(0).astype('int')
    X_train_final, scaler, top10, site_dict = add_features(train_times, train_sites, X_train_sites)
    X_test_final, _, __, ___ = add_features(test_times, test_sites, X_test_sites, scaler, top10, site_dict)

with timer('Cross-validation'):
    time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
    logit = LogisticRegression(random_state=SEED, solver='liblinear')

    # I've done cross-validation locally, and do not reproduce these heavy computations here,
    # but this is the vest C that I've found
    c_values = [BEST_LOGIT_C]
    # c_values = np.logspace(-2, 2, 20)

    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                       scoring='roc_auc', n_jobs=N_JOBS, cv=time_split, verbose=1)

    logit_grid_searcher.fit(X_train_final, y_train)
    print(logit_grid_searcher.best_score_, logit_grid_searcher.best_params_)

with timer('Test prediction and submission'):
    test_pred = logit_grid_searcher.predict_proba(X_test_final)[:, 1]
    pred_df = pd.DataFrame(test_pred, index=np.arange(1, test_pred.shape[0] + 1),
                           columns=['target'])
    pred_df.to_csv(f'submission_{AUTHOR}.csv', index_label='session_id')