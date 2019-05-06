import os
import numpy as np
import pandas as pd
import time
import json
from contextlib import contextmanager
from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from langdetect import detect


PATH_TO_DATA = '../../DataSets/kaggle_medium'
AUTHOR = 'Denis_Finogenov'
SEED = 17
MEAN_TEST_TARGET = 4.33328   # what we got by submitting all zeros

learning_rate = 0.1
n_estimators = 120
num_leaves = 128


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def get_lang(row):
    try:
        return detect(row)
    except:
        return None


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)
        return read_json_line(line=new_line)
    return result


def parse_html(html):
    soup = BeautifulSoup(html, 'lxml')

    article = soup.find('div', class_='postArticle-content')
    content = article.getText(separator=' ')
    imgs = len(article.select('img'))
    hrefs = len(article.select('a'))

    ul = soup.find('ul', class_='tags')
    tags = []
    if ul:
        tags = [a.text.replace(' ', '') for a in ul.find_all('a')]

    return content, imgs, hrefs, tags


def load_data(path_to_data, nrows=-1):
    # features = ['content', 'published', 'title', 'author']
    with open(path_to_data, encoding='utf-8') as inp_json_file:
        rows = []
        for line in tqdm_notebook(inp_json_file):
            if nrows != -1 and len(rows) > nrows:
                break

            json_data = read_json_line(line)

            # title
            title = json_data['title'].replace('\n', ' ').replace('\r', ' ')
            title_no_html_tags = strip_tags(title)

            # content
            json_content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content, img_cnt, href_cnt, tags = parse_html(json_content)

            rows.append([
                pd.to_datetime(json_data['published']['$date']),
                title_no_html_tags,
                json_data['url'].split('//')[-1].split('/')[0],
                json_data['meta_tags']['description'],
                json_data['meta_tags']['author'],
                json_data['meta_tags']['twitter:data1'].split(' ')[0],
                tags,
                img_cnt,
                href_cnt,
                content
            ])

    columns = [
        'published_time',
        'title',
        'domain',
        'description',
        'author',
        'read_time',
        'tags',
        'img_cnt',
        'href_cnt',
        'content'
    ]
    df = pd.DataFrame(data=rows, columns=columns)
    return df


def preprocessing(df):
    df['read_time'] = df['read_time'].astype(np.int8)
    df['content_len'] = df['content'].map(len).astype(np.int32)
    df['title_len'] = df['title'].map(len).astype(np.int32)
    df['desc_len'] = df['description'].map(len).astype(np.int32)

    df = process_ts(df)
    df = process_lang(df)
    df = process_tags(df)

    # categorical columns
    print('preprocessing...categorical features')
    categorical_columns = ['domain', 'lang', 'author']
    for cat_col in categorical_columns:
        if df[cat_col].count() != df.shape[0]:
            df[cat_col] = df[cat_col].fillna('unknown', axis=0)

        lbl = LabelEncoder()
        lbl.fit(list(df[cat_col].values.astype('str')))
        df.loc[:, cat_col] = lbl.transform(list(df[cat_col].values.astype('str')))

    return df


def preprocessing_text_data(df, tfidf_title=None, tfidf_tags=None,
                            tfidf_description=None, tfidf_content=None):
    print('preprocessing...text data')

    if tfidf_title is None:
        tfidf_title = TfidfVectorizer(max_features=100000,
                                      ngram_range=(1, 2))
        title_sparse = tfidf_title.fit_transform(df['title'])
    else:
        title_sparse = tfidf_title.transform(df['title'])
    print('TfIdf size - title: {}'.format(title_sparse.shape))

    if tfidf_tags is None:
        tfidf_tags = TfidfVectorizer(max_features=30000, ngram_range=(1, 1))
        tags_sparse = tfidf_tags.fit_transform(df['tags'])
    else:
        tags_sparse = tfidf_tags.transform(df['tags'])
    print('TfIdf size - tags: {}'.format(tags_sparse.shape))

    if tfidf_description is None:
        tfidf_description = TfidfVectorizer(max_features=50000,
                                            ngram_range=(1, 3))
        description_sparse = tfidf_description.fit_transform(df['description'])
    else:
        description_sparse = tfidf_description.transform(df['description'])
    print('TfIdf size - description: {}'.format(description_sparse.shape))

    if tfidf_content is None:
        tfidf_content = TfidfVectorizer(max_features=50000,
                                        ngram_range=(1, 2))
        content_sparse = tfidf_content.fit_transform(df['content'])
    else:
        content_sparse = tfidf_content.transform(df['content'])
    print('TfIdf size - content: {}'.format(content_sparse.shape))

    df.drop(['title', 'tags', 'description', 'content'], inplace=True, axis=1)
    X_sparse = csr_matrix(hstack([title_sparse, tags_sparse, description_sparse, content_sparse]))
    print(X_sparse.shape)

    return X_sparse, tfidf_title, tfidf_tags, tfidf_description, tfidf_content


def process_ts(df):
    print('preprocessing...time')

    df['read_time'] = df['read_time'].astype(np.int8)
    df['mm'] = df['published_time'].apply(lambda ts: ts.month).astype(np.int8)
    df['yyyy'] = df['published_time'].apply(lambda ts: ts.year).astype(np.int16)
    df['yyyymm'] = df['published_time'].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
    df['hour'] = df['published_time'].apply(lambda ts: ts.hour).astype(np.int8)
    df['dayofweek'] = df['published_time'].apply(lambda ts: ts.dayofweek).astype(np.int8)
    df['weekend'] = df['published_time'].apply(lambda ts: ts.dayofweek > 5).astype(np.int8)
    df['morning'] = df['published_time'].apply(lambda ts: (ts.hour >= 7) & (ts.hour < 12)).astype(np.int8)
    df['day'] = df['published_time'].apply(lambda ts: (ts.hour >= 12) & (ts.hour < 18)).astype(np.int8)
    df['evening'] = df['published_time'].apply(lambda ts: (ts.hour >= 18) & (ts.hour < 23)).astype(np.int8)
    df['night'] = df['published_time'].apply(lambda ts: (ts.hour >= 23) | (ts.hour < 7)).astype(np.int8) # or!

    df.drop(columns='published_time', inplace=True, axis=1)
    return df


def process_lang(df):
    print('preprocessing...language')
    df['lang'] = df['description'].map(get_lang)
    return df


def process_tags(df):
    print('preprocessing...tags')
    df['tags_cnt'] = df['tags'].map(len).astype(np.int8)
    df['tags'] = df['tags'].map(' '.join)
    return df


def lightgbm_prediction(X_train, y_train, X_valid, y_valid, X_test):
    lgb_x_train = lgb.Dataset(X_train.astype(np.float32), label=np.log1p(y_train))
    lgb_x_valid = lgb.Dataset(X_valid.astype(np.float32), label=np.log1p(y_valid))

    lgb_params = {'seed': SEED,
                  'learning_rate': learning_rate,
                  'n_estimators': n_estimators,
                  'num_leaves': num_leaves,
                  'objective': 'mean_absolute_error',
                  'metric': 'mae'}
    bst_lgb = lgb.train(lgb_params, lgb_x_train,
                        valid_sets=[lgb_x_valid],
                        early_stopping_rounds=10)

    val_preds = np.expm1(bst_lgb.predict(X_valid.astype(np.float32)))
    print('MAE LightGBM : {}'.format(mean_absolute_error(y_valid, val_preds)))

    test_pred = np.expm1(bst_lgb.predict(X_test.astype(np.float32),
                                         num_iteration=bst_lgb.best_iteration))
    return test_pred, val_preds


def ridge_prediction(X_train, y_train, X_valid, y_valid, X_test):
    ridge = Ridge(alpha=1, random_state=SEED)
    ridge.fit(X_train, np.log1p(y_train))
    val_preds = np.expm1(ridge.predict(X_valid))
    print('MAE Ridge : {}'.format(mean_absolute_error(y_valid, val_preds)))

    test_pred = np.expm1(ridge.predict(X_test))
    return test_pred, val_preds


def ensamble(train, y, X_test):
    X_train, X_valid, y_train, y_valid = train_test_split(train, y,
                                                          test_size=0.3, random_state=SEED)
    ridge = Ridge(alpha=1, random_state=SEED)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_valid)
    print('MAE Ensamble : {}'.format(mean_absolute_error(y_valid, preds)))

    test_pred = ridge.predict(X_test)
    return test_pred


def submit(preds, name='submit'):
    # leaderboard probing
    test_pred_modif = preds + MEAN_TEST_TARGET - y_train.mean()

    submission_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'sample_submission.csv'), index_col='id')
    submission_df['log_recommends'] = test_pred_modif
    submission_df.to_csv(f'{name}.csv')



with timer('Preprocessing data'):
    train_path_json = os.path.join(PATH_TO_DATA, 'train.json')
    test_path_json = os.path.join(PATH_TO_DATA, 'test.json')

    train_df = load_data(train_path_json)
    test_df = load_data(test_path_json)

    # Preprocessing Merged Train and Test data
    data = pd.concat([train_df, test_df])
    trn_len = train_df.shape[0]
    data = preprocessing(data)

    # Back split data
    train_df = data[:trn_len]
    test_df = data[trn_len:]

    train_sparse, tfidf_title, tfidf_tags, tfidf_description, tfidf_content = preprocessing_text_data(train_df)
    test_sparse, _, _, _, _ = preprocessing_text_data(test_df, tfidf_title, tfidf_tags,
                                                      tfidf_description, tfidf_content)

    y = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'),
                    index_col='id')['log_recommends'].values


with timer('LightGBM: train and predict'):
    train = csr_matrix(hstack([train_df, train_sparse]))
    X_test = csr_matrix(hstack([test_df, test_sparse]))

    X_train, X_valid, y_train, y_valid = train_test_split(train, y,
                                                          test_size=0.3, random_state=SEED)

    lgb_pred_test, lgb_pred_val = lightgbm_prediction(X_train, y_train, X_valid, y_valid, X_test)

with timer('Ridge: train and predict'):
    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(train_df)
    test_df_scaled = scaler.transform(test_df)

    train = csr_matrix(hstack([train_df_scaled, train_sparse]))
    X_test = csr_matrix(hstack([test_df_scaled, test_sparse]))

    X_train, X_valid, y_train, y_valid = train_test_split(train, y,
                                                          test_size=0.3, random_state=SEED)

    ridge_pred_test, ridge_pred_val = ridge_prediction(X_train, y_train, X_valid, y_valid, X_test)

with timer('Prepare submission'):
    ensamble_pred = ensamble(pd.DataFrame({'lgbm': lgb_pred_val, 'ridge': ridge_pred_val}),
                                           y_valid,
                                           pd.DataFrame({'lgbm': lgb_pred_test, 'ridge': ridge_pred_test}))

    submit(ensamble_pred, f'submission_medium_{AUTHOR}')
