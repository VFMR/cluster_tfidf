import sys
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering


def test_rng():
    X = np.random.randn(100, 3)
    y = np.random.randint(0,2, len(X))
    X, _, y, __ = train_test_split(X, y, test_size=0.1)

    clf = RandomForestClassifier(n_estimators=200,n_jobs=1)

    clf.fit(X, y)
    preds = clf.predict(X)
    print(preds.mean(), np.std(preds))

    seed = random.randrange(999999999)
    print(f'Seed: {seed}')

    clustering = AgglomerativeClustering(distance_threshold=0.3, n_clusters=None)
    preds2 = clustering.fit_predict(X)
    print('N clusters: ', np.max(preds2))

