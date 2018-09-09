
import sklearn_crfsuite
import sklearn_crfsuite as c
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from processed.linguistic_features import sent2features, train_sents, sent2labels, test_sents

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = c.CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
y_pred = crf.predict(X_test)


sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.classification_report(
    MultiLabelBinarizer().fit_transform(y_test), MultiLabelBinarizer().fit_transform(y_pred)
))