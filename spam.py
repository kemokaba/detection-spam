import os
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


DATA_DIR = 'enron'
target_names = ['ham', 'spam']

def get_data(DATA_DIR):
    subfolders = ['enron%d' % i for i in range(1,7)]

    data = []
    target = []
    for subfolder in subfolders:
        # spam
        spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
        for spam_file in spam_files:
            with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding='ascii', errors='ignore') as f:
                data.append(f.read())
                target.append(1)
                
        # ham
        ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
        for ham_file in ham_files:
            with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding='ascii', errors='ignore') as f:
                data.append(f.read())
                target.append(0)
                
    target = np.array(target)
    return(data, target)
    
X, y = get_data(DATA_DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

"""count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf = SGDClassifier(tol=None, n_jobs=-1).fit(X_train_tfidf, y_train)

X_test_counts = count_vec.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)"""

text_clf = Pipeline([('count_vec', CountVectorizer()), 
                     ('tfidf_transformer', TfidfTransformer()),
                     ('clf', SGDClassifier(tol=None, n_jobs=-1))]).fit(X_train, y_train)

# y_pred = clf.predict(X_test_tfidf)
y_pred = text_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred, target_names=target_names))
print(metrics.classification.accuracy_score(y_test, y_pred))


