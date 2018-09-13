"""http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle


def pr_curve(y_true, y_score, pos_label=None, y_group=None):
    res = []
    res.append(("All",) + metrics.precision_recall_curve(y_true, y_score, pos_label=pos_label))
    dat = pd.DataFrame({"y": y_true, "p": y_score, "g": y_group})
    for g, tmp in dat.groupby(["g"]):
        res.append((str(g),) + metrics.precision_recall_curve(tmp["y"], tmp["p"], pos_label=pos_label))
    return res


def roc_curve(y_true, y_score, pos_label=None, y_group=None):
    res = []
    res.append(("All",) + metrics.roc_curve(y_true, y_score, pos_label=pos_label))
    dat = pd.DataFrame({"y": y_true, "p": y_score, "g": y_group})
    for g, tmp in dat.groupby(["g"]):
        res.append((str(g),) + metrics.roc_curve(tmp["y"], tmp["p"], pos_label=pos_label))
    return res


def confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    """
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    y_true = y_test
    y_pred = y_pred
    classes = ["class {}".format(i) for i in range(y_test.max()+1)]
    confusion_matrix(y_true, y_pred, classes)
    """
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        conf_matrix = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(conf_matrix, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def F1(precision, recall):
    return 2*precision*recall / (precision+recall)


def proposal_threshold(precision, recall, thresholds, format=True, func=F1):
    p, r, f, t1, t2 = 0, 0, 0, 0, 0
    for pi, ri, ti in zip(precision, recall, thresholds):
        fi = func(pi, ri)
        if fi > f:
            p, r, f, t1, t2 = pi, ri, fi, ti, ti
        elif fi == f:
            t2 = ti
    if format:
        return "f1 {:.2f} threshold in [{:.2f}, {:.2f}]".format(f, t1, t2)
    return p, r, f, t1, t2


# Plot the Precision-Recall curve
def plt_pr_curve_bin_class(y_true, y_score):
    """
    # In binary classification
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    import numpy as np
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2], test_size=.5, random_state=random_state)
    # Create a classifier
    classifier = svm.LinearSVC(random_state=random_state)
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)

    y_true = y_test
    y_score = y_score
    plt_pr_curve_bin_class(y_true, y_score)
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(12, 8))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve [{}]".format(proposal_threshold(precision, recall, thresholds)))
    plt.show()


def plt_pr_curve_multi_class(y_true, y_score, classes):
    """
    # In multi-label classification
    from sklearn import svm, datasets
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import train_test_split
    import numpy as np
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # Use label_binarize to be multi-label like settings
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]
    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=random_state)
    # Create a classifier
    from sklearn.multiclass import OneVsRestClassifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
    classifier.fit(X_train, Y_train)
    Y_score = classifier.decision_function(X_test)

    y_true = Y_test
    y_score = Y_score
    classes = ["class {}".format(i) for i in range(Y_test.shape[1])]
    plt_pr_curve_multi_class(y_true, y_score, classes)
    """
    precision = dict()
    recall = dict()
    thresholds = dict()
    for i, name in enumerate(classes):
        precision[name], recall[name], thresholds[name] = metrics.precision_recall_curve(y_true[:, i], y_score[:, i])
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    plt.figure(figsize=(12, 8))
    lines = []
    labels = []
    for f_score in np.linspace(0.2, 0.8, num=4):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    for name, color in zip(classes, colors):
        l, = plt.plot(recall[name], precision[name], color=color, lw=2)
        lines.append(l)
        labels.append("[{}], {}".format(name, proposal_threshold(precision[name], recall[name], thresholds[name])))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve to multi-class")
    plt.legend(lines, labels, loc="lower left")
    plt.show()