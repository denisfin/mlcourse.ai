import os
import numpy as np
import pandas as pd
import json
import ujson as json
import pickle
import timeit
import sys
import re
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import catboost as cat
import lightgbm as lgb


def progress_bar(value, end_value, bar_length=20):
    percent = float(value) / end_value
    arrow = '-' * int(round(percent*bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}% ({2} from {3})".format(arrow + spaces,
                                                                   int(round(percent * 100)),
                                                                   value, end_value))


def read_matches(matches_file):
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)

    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)


def plot_feature_importance(lgb_model, X, y, seed=28):
    print('plot feature importance...')
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        lgb_model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',
                  verbose=0, early_stopping_rounds=20)

        y_pred_valid = lgb_model.predict_proba(X_valid)[:, 1]

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X.columns
        fold_importance["importance"] = lgb_model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    feature_importance["importance"] /= n_fold
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:75].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(14, 16))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)')
    plt.savefig('feature_importance.png')


def add_new_features(df_features, matches_file):
    file_name = os.path.split(matches_file)[1].split('.')[0]

    # load params
    with open(f'{file_name}_preproc.pkl', 'rb') as file:
        return pickle.load(file)

    start = timeit.default_timer()
    # Process raw data and add new features
    ind_progress = 1
    for match in read_matches(matches_file):
        progress_bar(ind_progress, df_features.shape[0])
        ind_progress += 1

        match_id_hash = match['match_id_hash']
        slots = [x['player_slot'] for x in match['players']]

        # Section 'objective'
        radiant_tower_kills = 0
        dire_tower_kills = 0

        radiant_roshan_kills = 0
        dire_roshan_kills = 0

        radiant_aegis_kills = 0
        dire_aegis_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1
            if objective['type'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                if objective['team'] == 2:
                    radiant_roshan_kills += 1
                if objective['team'] == 3:
                    dire_roshan_kills += 1
            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                continue
            if objective['type'] == 'CHAT_MESSAGE_TOWER_DENY':
                continue
            if objective['type'] == 'CHAT_MESSAGE_AEGIS_STOLEN':
                continue
            if objective['type'] == 'CHAT_MESSAGE_DENIED_AEGIS':
                continue
            if objective['type'] == 'CHAT_MESSAGE_AEGIS':
                ind = slots.index(objective['player_slot'])
                if ind < 5:
                    radiant_aegis_kills += 1
                else:
                    dire_aegis_kills += 1

        df_features.loc[match_id_hash, 'radiant_aegis'] = radiant_aegis_kills
        df_features.loc[match_id_hash, 'dire_aegis'] = dire_aegis_kills
        df_features.loc[match_id_hash, 'diff_aegis'] = radiant_aegis_kills - dire_aegis_kills

        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills

        df_features.loc[match_id_hash, 'radiant_roshan_kills'] = radiant_roshan_kills
        df_features.loc[match_id_hash, 'dire_roshan_kills'] = dire_roshan_kills
        df_features.loc[match_id_hash, 'diff_roshan_kills'] = radiant_roshan_kills - dire_roshan_kills

        # Section 'chat'
        for ind in range(10):
            if ind < 5:
                player_name = 'r%d' % (ind + 1)
            else:
                player_name = 'd%d' % (ind - 4)
            df_features.loc[match_id_hash, f'{player_name}_chat_count_messages'] = 0

        for chat in match['chat']:
            player_slot = chat['player_slot']
            if player_slot is None:
                continue
            ind = slots.index(player_slot)

            if ind < 5:
                player_name = 'r%d' % (ind + 1)
            else:
                player_name = 'd%d' % (ind - 4)

            df_features.loc[match_id_hash, f'{player_name}_chat_count_messages'] += 1


        # Section 'players'
        for slot, player in enumerate(match['players']):
            if slot < 5:
                player_name = 'r%d' % (slot + 1)
            else:
                player_name = 'd%d' % (slot - 4)

            df_features.loc[match_id_hash, f'{player_name}_max_hero_hit'] = player['max_hero_hit']['value']
            df_features.loc[match_id_hash, f'{player_name}_purchase_count'] = len(player['purchase_log'])

            # damage
            damage_by_hero = sum([player['damage'][x] for x in player['damage'] if x.startswith('npc_dota_hero')])
            damage_taken_by_hero = sum([player['damage_taken'][x] for x in player['damage_taken']
                                        if x.startswith('npc_dota_hero')])

            df_features.loc[match_id_hash, f'{player_name}_damage_dealt'] = damage_by_hero
            df_features.loc[match_id_hash, f'{player_name}_damage_received'] = damage_taken_by_hero
            df_features.loc[match_id_hash, f'{player_name}_diff_damage'] = damage_by_hero - damage_taken_by_hero

            # items
            # items = player['item_uses'].keys()
            # for item in items:
            #     new_column = f'{player_name}_item_use_{item}'
            #     df_features.loc[match_id_hash, new_column] = player['item_uses'][item]

            # abilities
            # abilities = player['ability_uses'].keys()
            # for ability in abilities:
            #     new_column = f'{player_name}_ability_use_{ability}'
            #     df_features.loc[match_id_hash, new_column] = player['ability_uses'][ability]
            #
            # df_features.loc[match_id_hash, f'{player_name}_ability_upgrades'] = len(player['ability_upgrades'])

            # hero inventary
            # inventories = player['hero_inventory']
            # for inventory in inventories:
            #     id = inventory['id']
            #     new_column = f'{player_name}_inventary_{id}'
            #     df_features.loc[match_id_hash, new_column] = 1


    # save data
    with open(f'{file_name}_preproc.pkl', 'wb') as file:
        pickle.dump(df_features, file, pickle.HIGHEST_PROTOCOL)

    stop = timeit.default_timer()
    print('\n{0} preprocessing: {1} sec.'.format(file, int(stop - start)))

    return df_features


def get_top_radiant_heros(X, y, top_count=30):
    df = pd.concat([X, y], axis=1)
    df = df[df['radiant_win'] == 1].drop(['radiant_win'], axis=1)
    top_radiant_heros = pd.Series(df.values.ravel()).value_counts()[:top_count].index.values
    return top_radiant_heros


def preprocess_hero_ids(full_df_heros, y):
    r_heros = [f'r{i}_hero_id' for i in range(1, 6)]
    d_heros = [f'd{i}_hero_id' for i in range(1, 6)]
    top_radiant_heros = get_top_radiant_heros(full_df_heros[r_heros], y, top_count=10)
    top_heros_couples = list(combinations(top_radiant_heros, 2))

    for row in full_df_heros.iterrows():
        cur_r_heros = row[1][r_heros].values.astype(int)
        cur_d_heros = row[1][d_heros].values.astype(int)
        for hero in cur_r_heros:
            full_df_heros.loc[row[0], f'r_hero_{hero}'] = 1
        for hero in cur_d_heros:
            full_df_heros.loc[row[0], f'd_hero_{hero}'] = 1

        cur_r_heros_cuples = list(combinations(cur_r_heros, 2))
        cur_r_heros_intersect = list(set(cur_r_heros_cuples) & set(top_heros_couples))
        for hero in cur_r_heros_intersect:
            full_df_heros.loc[row[0], f'r_hero_{hero[0]}_{hero[1]}'] = 1

        cur_d_heros_cuples = list(combinations(cur_d_heros, 2))
        cur_d_heros_intersect = list(set(cur_d_heros_cuples) & set(top_heros_couples))
        for hero in cur_d_heros_intersect:
            full_df_heros.loc[row[0], f'd_hero_{hero[0]}_{hero[1]}'] = 1

    full_df_heros.drop(r_heros + d_heros, inplace=True, axis=1)
    full_df_heros = full_df_heros.fillna(0)

    with open('full_df_heros.pkl', 'wb') as file:
        pickle.dump(full_df_heros, file, pickle.HIGHEST_PROTOCOL)

    return full_df_heros


def make_coordinate_features(df):
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        for player in players:
            df[f'{player}_distance'] = np.sqrt(df[f'{player}_x']**2 + df[f'{player}_y']**2)
            df.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)
    return df


def hero_id_subset_analyzer(text):
    ids = set()
    for i in range(1, 2):
        hero_ids = text.split(' ')
        hero_ids.sort()
        combs = set(combinations(hero_ids, i))
        ids = ids.union(combs)
    ids = {"_".join(item) for item in ids}
    return ids


def replace_hero_ids(df, train=True):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000,
                                 tokenizer=lambda s: s.split(),
                                 analyzer=hero_id_subset_analyzer)
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        hero_columns = [f'{player}_hero_id' for player in players]

        # combine all hero id columns into one
        df_hero_id_as_text = df[hero_columns].apply(lambda row: ' '.join([str(i) for i in row]), axis=1).tolist()

        if train:
            new_cols = pd.DataFrame(vectorizer.fit_transform(df_hero_id_as_text).todense(),
                                    columns=vectorizer.get_feature_names())
        else:
            new_cols = pd.DataFrame(vectorizer.transform(df_hero_id_as_text).todense(),
                                    columns=vectorizer.get_feature_names())

        # add index to vectorized dataset - needed for merge?
        new_cols['match_id_hash'] = df.index.values
        new_cols = new_cols.set_index('match_id_hash').add_prefix(f'{team}_hero_')  # e.g.r_hero_10_21

        df = pd.merge(df, new_cols, on='match_id_hash')
        df.drop(columns=hero_columns, inplace=True)
    return df


def data_preprocess(path):
    train_df = pd.read_csv(os.path.join(path, 'train_features.csv'), index_col='match_id_hash')
    y_train = pd.read_csv(os.path.join(path, 'train_targets.csv'), index_col='match_id_hash')
    y_train = y_train['radiant_win'].astype(int)
    test_df = pd.read_csv(os.path.join(path, 'test_features.csv'), index_col='match_id_hash')

    # add new features
    print('Add train features...')
    train_df = add_new_features(train_df, os.path.join(path, 'train_matches.jsonl'))
    print('Add test features...')
    test_df = add_new_features(test_df, os.path.join(path, 'test_matches.jsonl'))

    # preprocessing Merged Train and Test data
    full_df = pd.concat([train_df, test_df], sort=False)
    full_df = full_df.fillna(0)
    train_size = train_df.shape[0]

    delete_columns = ['game_mode', 'lobby_type']
    full_df.drop(delete_columns, inplace=True, axis=1)

    hero_id_columns = [x for x in full_df.columns if '_hero_id' in x]
    full_df_without_heros = full_df[list(set(full_df.columns) - set(hero_id_columns))]

    # coordinate features
    full_df_without_heros = make_coordinate_features(full_df_without_heros)

    # total by team
    total_columns = ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana', 'level',
                    'distance', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups', 'firstblood_claimed',
                    'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed', 'sen_placed',

                    'max_hero_hit', 'purchase_count', 'ability_upgrades',
                    'damage', 'damage_received', 'damage_taken', 'damage_dealt', 'diff_damage', 'chat_count_messages']

    drop_columns = []
    for i, c in enumerate(total_columns):
        progress_bar(i+1, len(total_columns))

        r_columns = [col for col in full_df_without_heros.columns if re.match(f'r\d_{c}$', col) is not None]
        d_columns = [col for col in full_df_without_heros.columns if re.match(f'd\d_{c}$', col) is not None]

        if len(r_columns) == 0 or len(d_columns) == 0:
            continue

        full_df_without_heros['r_total_' + c] = full_df_without_heros[r_columns].sum(1)
        full_df_without_heros['d_total_' + c] = full_df_without_heros[d_columns].sum(1)

        full_df_without_heros['r_max_' + c] = full_df_without_heros[r_columns].max(1)
        full_df_without_heros['d_max_' + c] = full_df_without_heros[d_columns].max(1)

        if 'item_use_' not in c and 'ability_use_' not in c:

            full_df_without_heros['r_std_' + c] = full_df_without_heros[r_columns].std(1)
            full_df_without_heros['d_std_' + c] = full_df_without_heros[d_columns].std(1)

        drop_columns += r_columns
        drop_columns += d_columns

    full_df_without_heros.drop(drop_columns, inplace=True, axis=1)

    # add hero id and couples ids
    df_hero_ids = preprocess_hero_ids(full_df[hero_id_columns], y_train)

    # drop couple of hero ids, because it decrease result
    col_couple_heros = [col for col in df_hero_ids.columns if re.search(r'_hero_\d+_\d+', col) is not None]
    df_hero_ids = df_hero_ids.drop(col_couple_heros, axis=1)

    full_df = pd.concat([full_df_without_heros, df_hero_ids], axis=1)

    # time features
    full_df['game_time_hour'] = full_df['game_time'] // 3600
    full_df['game_time_minute'] = full_df['game_time'] // 60
    full_df['game_time_sec'] = full_df['game_time'] % 60
    full_df = full_df.drop(['game_time'], axis=1)

    # Back split data
    X_train = full_df[:train_size]
    X_test = full_df[train_size:]

    # save data
    with open('X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file, pickle.HIGHEST_PROTOCOL)
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file, pickle.HIGHEST_PROTOCOL)

    return X_train, y_train, X_test


def model_zoo_cv(X, y, plot_feat_importance=False, seed=28):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)

    model_rf = RandomForestClassifier(n_estimators=100, n_jobs=4,
                                      max_depth=None, random_state=17)
    cv_scores_rf = cross_val_score(model_rf, X, y, cv=cv, scoring='roc_auc')

    model_lgb = lgb.LGBMClassifier(random_state=seed)
    cv_scores_lgb = cross_val_score(model_lgb, X, y, cv=cv, scoring='roc_auc')
    if plot_feat_importance:
        plot_feature_importance(model_lgb, X, y, seed)

    model_cat = cat.CatBoostClassifier(random_state=seed, silent=True)
    cv_scores_cat = cross_val_score(model_cat, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    # params = {'eval_metric': 'AUC',
    #           'iterations': 100,
    #           'early_stopping_rounds': 20,
    #           'random_state': seed}
    # data_cat = cat.Pool(X, y, cat_features=cat_columns)
    # cv_scores_cat2 = cat.cv(data_cat, params, seed=seed)

    pipe_logit = Pipeline([('scaler', StandardScaler()),
                           ('logit', LogisticRegression(random_state=seed))])
    cv_scores_logit = cross_val_score(pipe_logit, X, y, cv=cv, scoring='roc_auc')

    cv_results = pd.DataFrame(data={'RF': cv_scores_rf,
                                    'LGB': cv_scores_lgb,
                                    'CAT': cv_scores_cat,
                                    'LOGIT': cv_scores_logit})
    print(cv_results)
    print(cv_results.describe())


def catboost_grid_search(X, y, seed=28):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
    search_spaces = {'iterations': range(10, 1000, 10),
                     'depth': range(1, 8),
                     'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                     'border_count': range(1, 255, 10),
                     'l2_leaf_reg': range(2, 30, 2)}

    catboost = cat.CatBoostClassifier(random_seed=seed, eval_metric='AUC')
    grid_cat = RandomizedSearchCV(catboost, param_distributions=search_spaces, cv=cv,
                                  scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_cat.fit(X, y)
    print('CatBoost = {0}, {1}'.format(grid_cat.best_score_, grid_cat.best_params_))


def train_catboost(X, y, seed=28):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=seed, shuffle=True)

    best_params = {'iterations': 1000,
                     'depth': 7,
                     'learning_rate': 0.1,
                     'border_count': 111,
                     'l2_leaf_reg': 6}

    catboost = cat.CatBoostClassifier(**best_params, random_seed=seed, eval_metric='AUC')
    catboost.fit(X=X_train, y=y_train, use_best_model=True,
                 early_stopping_rounds=30, eval_set=(X_val, y_val))


def train_linear(X, y, seed=28):
    pipe_logit = Pipeline([('scaler', StandardScaler()),
                           ('logit', LogisticRegression(random_state=seed))])
    cv_scores_logit = cross_val_score(pipe_logit, X, y, cv=3, scoring='roc_auc')
    print(cv_scores_logit.mean())


def train(X, y, seed=28):
    # evaluate models for EDA
    # model_zoo_cv(X, y, plot_feat_importance=True, seed=seed)

    # train best model and finetuning params
    # train_catboost(X, y, seed=seed)
    # train_linear(X, y, seed=seed)

    pipe_logit = Pipeline([('scaler', StandardScaler()),
                           ('logit', LogisticRegression(random_state=seed))])
    pipe_logit.fit(X, y)

    return pipe_logit


def submit(model, X_test, save_path='submission_dota2.csv'):
    submission_filename = save_path
    y_test_pred = model.predict_proba(X_test)[:, 1]
    df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, index=X_test.index)
    df_submission.to_csv(submission_filename)
    print('Submission saved to {}'.format(submission_filename))


def main():
    PATH_TO_DATA = '../../DataSets/mlcourse-dota2-win-prediction'
    SEED = 28
    PREPROC = True

    if PREPROC == False:
        # load params
        with open('X_train.pkl', 'rb') as file:
            X = pickle.load(file)
        with open('X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        y = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'),
                        index_col='match_id_hash')['radiant_win'].astype(int)
    else:
        X, y, X_test = data_preprocess(path=PATH_TO_DATA)

    model = train(X, y, seed=SEED)

    # submit
    submit(model, X_test=X_test, save_path='submission_dota2.csv')


if __name__ == '__main__':
    main()
