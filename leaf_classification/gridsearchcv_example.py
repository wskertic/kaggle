# Copied almost entirely from Xu  Xu Yinan's script - uses slightly
# different calculation method to achieve higher score.


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


np.random.seed(42)

train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

params = {'C': [1, 10, 50, 100, 500, 1000, 2000],
          'tol': [0.001, 0.0001, 0.005]}
log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
clf = GridSearchCV(log_reg, params, scoring='neg_log_loss',
                   refit='True', n_jobs=1, cv=5)
clf.fit(x_train, y_train)

print("best params: " + str(clf.best_params_))
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
    print(scores)

df = pd.DataFrame(clf.cv_results_)
output_scores = lambda x: "%0.3f (+/-%0.03f) for %r\n%s, %s" % (x['mean_train_score'], x['std_train_score'], x['params'], x['split0_train_score'],x['split1_train_score'])

print df.applymap(output_scores)

test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
