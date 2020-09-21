import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import data_process


titanic_data_sets = data_process.get_titanic()
titanic_train_X = pd.concat([titanic_data_sets[0], titanic_data_sets[1]])
titanic_train_y = pd.concat([titanic_data_sets[3], titanic_data_sets[4]])
titanic_test_X = titanic_data_sets[2]
titanic_test_y = titanic_data_sets[5]

credit_data_sets = data_process.get_credit()
credit_train_X = pd.concat([credit_data_sets[0], credit_data_sets[1]])
credit_train_y = pd.concat([credit_data_sets[3], credit_data_sets[4]])
credit_test_X = credit_data_sets[2]
credit_test_y = credit_data_sets[5]

# region KNN


def knn_grid_search(data_sets, prints=False, weights='uniform'):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    metrics = sorted(sklearn.neighbors.VALID_METRICS['brute'])
    metrics.remove('haversine')
    metrics.remove('mahalanobis')
    metrics.remove('precomputed')
    metrics.remove('seuclidean')
    metrics.remove('wminkowski')
    best = -1
    best_m = "N/A"
    best_n = -1
    best_m_avg = "N/A"
    best_avg = -1
    best_model = None
    for metric in metrics:
        metric_best = -1
        metric_n = -1
        scores = []

        for n_neighbors in np.arange(100)+1:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
            knn.fit(X_train, y_train)
            score = knn.score(X_val, y_val)
            scores.append(score)
            if score > metric_best:
                metric_n = n_neighbors
                metric_best = score

            if score > best:
                best = score
                best_m = metric
                best_n = n_neighbors
                best_model = knn

        avg = np.mean(scores)
        if avg > best_avg:
            best_m_avg = metric
            best_avg = avg

    if prints:
        print("best metric individual: " + best_m)
        print("best k for metric: " + str(best_n))
        print("score: " + str(best))

    return best_model, best_m, best_m_avg

# cityblock, 5
print("KNN grid search:")
print("Titanic:")
knn_titanic_grid, titanic_m, titanic_m_avg = knn_grid_search(titanic_data_sets, True, 'uniform')
# hamming, 6
print("Credit:")
knn_credit_grid, credit_m, credit_m_avg = knn_grid_search(credit_data_sets, True, 'distance')


def knn_viz_vals(data_sets, weights, metric):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    k_vals = np.arange(12) + 1
    scores = []
    trains = []
    for n_neighbors in k_vals:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
        knn.fit(X_train, y_train)
        score = knn.score(X_val, y_val)
        train = knn.score(X_train, y_train)
        scores.append(score)
        trains.append(train)

    return k_vals, scores, trains


def knn_subset_vals(data_sets, weights, metric):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    sizes = ((np.arange(9) + 1)/10 * len(X_train)).astype(int)
    scores = []
    for size in sizes:
        subset_X = X_train.sample(size)
        subset_y = y_train[subset_X.index]
        knn = KNeighborsClassifier(n_neighbors=6, metric=metric, weights=weights)
        knn.fit(subset_X, subset_y)
        score = knn.score(X_val, y_val)
        scores.append(score)

    return sizes, scores


def knn_viz(titanic_data_sets, credit_data_sets):

    # K scores
    titanic_ks, t_scores, t_trains = knn_viz_vals(titanic_data_sets, 'uniform', 'cityblock')
    credit_ks, c_scores, c_trains = knn_viz_vals(credit_data_sets, 'uniform', 'cityblock')

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(titanic_ks, t_trains, label='training')
    ax1.plot(titanic_ks, t_scores, label='validation')
    ax1.set_title('Titanic K vs. Validation Score')
    ax1.set_xlabel('K neighbors considered')
    ax1.set_ylabel('Validation Accuracy')
    ax1.legend()

    ax2.plot(credit_ks, c_trains, label='training')
    ax2.plot(credit_ks, c_scores, label='validation')
    ax2.set_title('Credit K vs. Validation Score')
    ax2.set_xlabel('K neighbors considered')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()

    f.savefig('knn_vs_score.png')


    # Dataset Size Scores
    titanic_sizes, t_scores = knn_subset_vals(titanic_data_sets, 'uniform', 'cityblock')
    credit_sizes, c_scores = knn_subset_vals(credit_data_sets, 'uniform', 'cityblock')

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(titanic_sizes, t_scores)
    ax1.set_title('Titanic Subset Size vs. Validation Score')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')

    ax2.plot(credit_sizes, c_scores)
    ax2.set_title('Credit Subset Size vs. Validation Score')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Validation Accuracy')

    f.savefig('knn_subset_score.png')





# make knn viz
knn_viz(titanic_data_sets, credit_data_sets)

# tuned models
knn_titanic = KNeighborsClassifier(n_neighbors=7, metric='cityblock', weights='uniform')
knn_titanic.fit(titanic_train_X, titanic_train_y)
titanic_score = knn_titanic.score(titanic_test_X, titanic_test_y)
titanic_grid = knn_titanic_grid.score(titanic_test_X, titanic_test_y)

knn_credit = KNeighborsClassifier(n_neighbors=8, metric='cityblock', weights='uniform')
knn_credit.fit(credit_train_X, credit_train_y)
credit_score = knn_credit.score(credit_test_X, credit_test_y)
credit_grid = knn_credit_grid.score(credit_test_X, credit_test_y)

print("KNN test scores")

print("Titanic")
print("Grid: ")
print(str(titanic_grid))
print("Tuned: ")
print(str(titanic_score))

print("Credit")
print("Grid: ")
print(str(credit_grid))
print("Tuned: ")
print(str(credit_score))


print("Train time")
print(timeit.timeit(lambda: knn_titanic.fit(titanic_train_X, titanic_train_y), number=100)/100)
print("Score time")
print(timeit.timeit(lambda: knn_titanic.score(titanic_test_X, titanic_test_y), number=100)/100)

# endregion


# region Decision Tree

def tree_depth_search(data_sets, prints=False):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    depths = np.arange(15) + 1

    best_depth = -1
    best_acc = -1
    best_model = None
    acc_train = []
    acc_val = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X_train, y_train)

        train_score = tree.score(X_train, y_train)
        acc_train.append(train_score)
        val_score = tree.score(X_val, y_val)
        acc_val.append(val_score)

        if val_score > best_acc:
            best_model = tree
            best_depth = depth
            best_acc = val_score

    if prints:
        print("best depth: " + str(best_depth))
        print("score: " + str(best_acc))

    return best_model, depths, acc_train, acc_val


titanic_tree_grid, t_depths, t_acc_train, t_acc_val = tree_depth_search(titanic_data_sets, True)
credit_tree_grid, c_depths, c_acc_train, c_acc_val = tree_depth_search(credit_data_sets, True)

# Charts
def tree_subset_vals(data_sets, max_depth):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    sizes = ((np.arange(19) + 1)/19 * len(X_train)).astype(int)
    scores = []
    for size in sizes:
        subset_X = X_train.sample(size)
        subset_y = y_train[subset_X.index]
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree.fit(subset_X, subset_y)
        score = tree.score(X_val, y_val)
        scores.append(score)

    return sizes, scores


def tree_viz(titanic_data_sets, credit_data_sets):

    # K scores
    titanic_tree_grid, t_depths, t_acc_train, t_acc_val = tree_depth_search(titanic_data_sets, False)
    credit_tree_grid, c_depths, c_acc_train, c_acc_val = tree_depth_search(credit_data_sets, False)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_depths, t_acc_train, label='training')
    ax1.plot(t_depths, t_acc_val, label='validation')
    ax1.set_title('Titanic Tree Depth vs. Accuracy')
    ax1.set_xlabel('Max Tree Depth')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(c_depths, c_acc_train, label='training')
    ax2.plot(c_depths, c_acc_val, label='validation')
    ax2.set_title('Credit Tree Depth vs. Accuracy')
    ax2.set_xlabel('Max Tree Depth')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    f.savefig('tree_depth_score.png')


    # Dataset Size Scores
    titanic_sizes, t_scores = tree_subset_vals(titanic_data_sets, 6)
    credit_sizes, c_scores = tree_subset_vals(credit_data_sets, 4)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(titanic_sizes, t_scores)
    ax1.set_title('Titanic Subset Size vs. Validation Score')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')

    ax2.plot(credit_sizes, c_scores)
    ax2.set_title('Credit Subset Size vs. Validation Score')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Validation Accuracy')

    f.savefig('tree_subset_score.png')

tree_viz(titanic_data_sets, credit_data_sets)


# tuned models
tree_titanic = DecisionTreeClassifier(max_depth=6, random_state=0)
tree_titanic.fit(titanic_train_X, titanic_train_y)
titanic_score = tree_titanic.score(titanic_test_X, titanic_test_y)
titanic_grid = titanic_tree_grid.score(titanic_test_X, titanic_test_y)

tree_credit = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_credit.fit(credit_train_X, credit_train_y)
credit_score = tree_credit.score(credit_test_X, credit_test_y)
credit_grid = credit_tree_grid.score(credit_test_X, credit_test_y)

print("Tree test scores")

print("Titanic")
print("Grid: ")
print(str(titanic_grid))
print("Tuned: ")
print(str(titanic_score))

print("Credit")
print("Grid: ")
print(str(credit_grid))
print("Tuned: ")
print(str(credit_score))


print("Train time")
print(timeit.timeit(lambda: tree_titanic.fit(titanic_train_X, titanic_train_y), number=100)/100)
print("Score time")
print(timeit.timeit(lambda: tree_titanic.score(titanic_test_X, titanic_test_y), number=100)/100)

# endregion

# region Boosted Trees

def boosted_tree_estimator_search(data_sets, prints=False):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    estimators = (np.arange(19) + 2)*10

    best_est = -1
    best_acc = -1
    best_model = None
    acc_train = []
    acc_val = []
    for estimator in estimators:
        tree = DecisionTreeClassifier(max_depth=1, random_state=0)
        boosted_tree = AdaBoostClassifier(base_estimator=tree, n_estimators=estimator, random_state=0)
        boosted_tree.fit(X_train, y_train)

        train_score = boosted_tree.score(X_train, y_train)
        acc_train.append(train_score)
        val_score = boosted_tree.score(X_val, y_val)
        acc_val.append(val_score)

        if val_score > best_acc:
            best_model = boosted_tree
            best_est = estimator
            best_acc = val_score

    if prints:
        print("best depth: " + str(best_est))
        print("score: " + str(best_acc))

    return best_model, estimators, acc_train, acc_val


titanic_boosted_tree_grid, t_depths, t_acc_train, t_acc_val = boosted_tree_estimator_search(titanic_data_sets, True)
credit_boosted_tree_grid, c_depths, c_acc_train, c_acc_val = boosted_tree_estimator_search(credit_data_sets, True)

# Charts
def boosted_tree_subset_vals(data_sets, estimator):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    sizes = ((np.arange(19) + 1)/19 * len(X_train)).astype(int)
    scores = []
    for size in sizes:
        subset_X = X_train.sample(size)
        subset_y = y_train[subset_X.index]
        tree = DecisionTreeClassifier(max_depth=1, random_state=0)
        boosted_tree = AdaBoostClassifier(base_estimator=tree, n_estimators=estimator, random_state=0)
        boosted_tree.fit(subset_X, subset_y)
        score = boosted_tree.score(X_val, y_val)
        scores.append(score)

    return sizes, scores


def boosted_tree_viz(titanic_data_sets, credit_data_sets):

    # K scores
    titanic_boosted_tree_grid, t_depths, t_acc_train, t_acc_val = boosted_tree_estimator_search(titanic_data_sets, False)
    credit_boosted_tree_grid, c_depths, c_acc_train, c_acc_val = boosted_tree_estimator_search(credit_data_sets, False)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_depths, t_acc_train, label='training')
    ax1.plot(t_depths, t_acc_val, label='validation')
    ax1.set_title('Titanic Boosted Tree Count vs. Accuracy')
    ax1.set_xlabel('Number of Estimators')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(c_depths, c_acc_train, label='training')
    ax2.plot(c_depths, c_acc_val, label='validation')
    ax2.set_title('Credit Boosted Tree Count  vs. Accuracy')
    ax2.set_xlabel('Number of Estimators')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    f.savefig('boost_est_score.png')


    # Dataset Size Scores
    titanic_sizes, t_scores = boosted_tree_subset_vals(titanic_data_sets, 150)
    credit_sizes, c_scores = boosted_tree_subset_vals(credit_data_sets, 40)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(titanic_sizes, t_scores)
    ax1.set_title('Titanic Subset Size vs. Validation Score')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')

    ax2.plot(credit_sizes, c_scores)
    ax2.set_title('Credit Subset Size vs. Validation Score')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Validation Accuracy')

    f.savefig('boosted_subset_score.png')

boosted_tree_viz(titanic_data_sets, credit_data_sets)


# tuned models
tree = DecisionTreeClassifier(max_depth=1, random_state=0)
boosted_tree_titanic = AdaBoostClassifier(base_estimator=tree, n_estimators=150, random_state=0)
boosted_tree_titanic.fit(titanic_train_X, titanic_train_y)
titanic_score = boosted_tree_titanic.score(titanic_test_X, titanic_test_y)
titanic_grid = titanic_boosted_tree_grid.score(titanic_test_X, titanic_test_y)

tree = DecisionTreeClassifier(max_depth=1, random_state=0)
boosted_tree_credit = AdaBoostClassifier(base_estimator=tree, n_estimators=40, random_state=0)
boosted_tree_credit.fit(titanic_train_X, titanic_train_y)
boosted_tree_credit.fit(credit_train_X, credit_train_y)
credit_score = boosted_tree_credit.score(credit_test_X, credit_test_y)
credit_grid = credit_boosted_tree_grid.score(credit_test_X, credit_test_y)

print("boosted_tree test scores")

print("Titanic")
print("Grid: ")
print(str(titanic_grid))
print("Tuned: ")
print(str(titanic_score))

print("Credit")
print("Grid: ")
print(str(credit_grid))
print("Tuned: ")
print(str(credit_score))


print("Train time")
print(timeit.timeit(lambda: boosted_tree_titanic.fit(titanic_train_X, titanic_train_y), number=100)/100)
print("Score time")
print(timeit.timeit(lambda: boosted_tree_titanic.score(titanic_test_X, titanic_test_y), number=100)/100)
# endregion

# region Neural Network


def mlp_grid_search(data_sets, prints=False, activation='tanh', static_width=-1):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    depths = np.arange(5) + 1
    if static_width == -1:
        widths = 20 * (np.arange(5) + 1)
        sizes = [tuple([x]*y) for x in widths for y in depths]
    else:
        sizes = [tuple([static_width]*d) for d in depths]

    best_size = -1
    best_acc = -1
    best_model = None
    acc_train = []
    acc_val = []
    for size in sizes:
        mlp = MLPClassifier(hidden_layer_sizes=size, activation=activation, random_state=0, max_iter=500)
        mlp.fit(X_train, y_train)

        if prints:
            print("Size: " + str(size) + " Complete")

        train_score = mlp.score(X_train, y_train)
        acc_train.append(train_score)
        val_score = mlp.score(X_val, y_val)
        acc_val.append(val_score)

        if val_score > best_acc:
            best_model = mlp
            best_size = size
            best_acc = val_score

    if prints:
        print("best size: " + str(best_size))
        print("score: " + str(best_acc))

    return best_model, depths, acc_train, acc_val


titanic_mlp_grid, t_depths, t_acc_train, t_acc_val = mlp_grid_search(titanic_data_sets, True)
credit_mlp_grid, c_depths, c_acc_train, c_acc_val = mlp_grid_search(credit_data_sets, True)


# Charts
def mlp_subset_vals(data_sets):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    sizes = ((np.arange(6) + 1)/6 * len(X_train)).astype(int)
    scores = []
    for size in sizes:
        subset_X = X_train.sample(size)
        subset_y = y_train[subset_X.index]
        mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), activation='tanh', random_state=0, max_iter=1000)
        mlp.fit(subset_X, subset_y)
        score = mlp.score(X_val, y_val)
        scores.append(score)

    return sizes, scores


def mlp_viz(titanic_data_sets, credit_data_sets):

    # K scores
    titanic_mlp_grid, t_depths, t_acc_train, t_acc_val = mlp_grid_search(titanic_data_sets, False, static_width=20)
    credit_mlp_grid, c_depths, c_acc_train, c_acc_val = mlp_grid_search(credit_data_sets, False, static_width=20)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_depths, t_acc_train, label='training')
    ax1.plot(t_depths, t_acc_val, label='validation')
    ax1.set_title('Titanic MLP N-layers vs. Accuracy')
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(c_depths, c_acc_train, label='training')
    ax2.plot(c_depths, c_acc_val, label='validation')
    ax2.set_title('Credit MLP N-layers vs. Accuracy')
    ax2.set_xlabel('Number of Layers')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    f.savefig('mlp_layer_score.png')


    # Dataset Size Scores
    titanic_sizes, t_scores = mlp_subset_vals(titanic_data_sets)
    credit_sizes, c_scores = mlp_subset_vals(credit_data_sets)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(titanic_sizes, t_scores)
    ax1.set_title('Titanic Subset Size vs. Validation Score')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')

    ax2.plot(credit_sizes, c_scores)
    ax2.set_title('Credit Subset Size vs. Validation Score')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Validation Accuracy')

    f.savefig('mlp_subset_score.png')

mlp_viz(titanic_data_sets, credit_data_sets)


# tuned models
mlp_titanic = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20), activation='tanh', random_state=0, max_iter=1000)
mlp_titanic.fit(titanic_train_X, titanic_train_y)
titanic_score = mlp_titanic.score(titanic_test_X, titanic_test_y)
titanic_grid = titanic_mlp_grid.score(titanic_test_X, titanic_test_y)

mlp_credit = MLPClassifier(hidden_layer_sizes=(60), activation='tanh', random_state=0, max_iter=1000)
mlp_credit.fit(credit_train_X, credit_train_y)
credit_score = mlp_credit.score(credit_test_X, credit_test_y)
credit_grid = credit_mlp_grid.score(credit_test_X, credit_test_y)

print("mlp test scores")

print("Titanic")
print("Grid: ")
print(str(titanic_grid))
print("Tuned: ")
print(str(titanic_score))

print("Credit")
print("Grid: ")
print(str(credit_grid))
print("Tuned: ")
print(str(credit_score))


print("Train time")
print(timeit.timeit(lambda: mlp_credit.fit(titanic_train_X, titanic_train_y), number=1))
print("Score time")
print(timeit.timeit(lambda: mlp_credit.score(titanic_test_X, titanic_test_y), number=1))

# endregion

# region SVM

def svm_degree_search(data_sets, prints=False):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    degrees = np.arange(10) + 1

    best_deg = -1
    best_acc = -1
    best_model = None
    acc_train = []
    acc_val = []
    for degree in degrees:
        svm = SVC(kernel='poly', degree=degree, random_state=0)
        svm.fit(X_train, y_train)

        train_score = svm.score(X_train, y_train)
        acc_train.append(train_score)
        val_score = svm.score(X_val, y_val)
        acc_val.append(val_score)

        if val_score > best_acc:
            best_model = svm
            best_deg = degree
            best_acc = val_score

    if prints:
        print("best depth: " + str(best_deg))
        print("score: " + str(best_acc))

    return best_model, degrees, acc_train, acc_val


titanic_svm_grid, t_depths, t_acc_train, t_acc_val = svm_degree_search(titanic_data_sets, True)
credit_svm_grid, c_depths, c_acc_train, c_acc_val = svm_degree_search(credit_data_sets, True)


# Charts
def svm_subset_vals(data_sets, degree):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    sizes = ((np.arange(19) + 1)/19 * len(X_train)).astype(int)
    scores = []
    for size in sizes:
        subset_X = X_train.sample(size)
        subset_y = y_train[subset_X.index]
        svm = SVC(kernel='poly', degree=degree, random_state=0)
        svm.fit(subset_X, subset_y)
        score = svm.score(X_val, y_val)
        scores.append(score)

    return sizes, scores


def svm_viz(titanic_data_sets, credit_data_sets):

    # K scores
    titanic_svm_grid, t_depths, t_acc_train, t_acc_val = svm_degree_search(titanic_data_sets, False)
    credit_svm_grid, c_depths, c_acc_train, c_acc_val = svm_degree_search(credit_data_sets, False)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_depths, t_acc_train, label='training')
    ax1.plot(t_depths, t_acc_val, label='validation')
    ax1.set_title('Titanic SVM polynomial kernel degree vs. Accuracy')
    ax1.set_xlabel('Kernel Poly Degree')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(c_depths, c_acc_train, label='training')
    ax2.plot(c_depths, c_acc_val, label='validation')
    ax2.set_title('Credit SVM polynomial kernel degree vs. Accuracy')
    ax2.set_xlabel('Kernel Poly Degree')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    f.savefig('svm_degree_score.png')


    # Dataset Size Scores
    titanic_sizes, t_scores = svm_subset_vals(titanic_data_sets, 2)
    credit_sizes, c_scores = svm_subset_vals(credit_data_sets, 5)

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(titanic_sizes, t_scores)
    ax1.set_title('Titanic Subset Size vs. Validation Score')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')

    ax2.plot(credit_sizes, c_scores)
    ax2.set_title('Credit Subset Size vs. Validation Score')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Validation Accuracy')

    f.savefig('svm_subset_score.png')

svm_viz(titanic_data_sets, credit_data_sets)


# tuned models
svm_titanic = SVC(kernel='poly', degree=2, random_state=0)
svm_titanic.fit(titanic_train_X, titanic_train_y)
titanic_score = svm_titanic.score(titanic_test_X, titanic_test_y)
titanic_grid = titanic_svm_grid.score(titanic_test_X, titanic_test_y)

svm_credit = SVC(kernel='poly', degree=5, random_state=0)
svm_credit.fit(credit_train_X, credit_train_y)
credit_score = svm_credit.score(credit_test_X, credit_test_y)
credit_grid = credit_svm_grid.score(credit_test_X, credit_test_y)

print("svm test scores")

print("Titanic")
print("Grid: ")
print(str(titanic_grid))
print("Tuned: ")
print(str(titanic_score))

print("Credit")
print("Grid: ")
print(str(credit_grid))
print("Tuned: ")
print(str(credit_score))


print("Train time")
print(timeit.timeit(lambda: svm_credit.fit(titanic_train_X, titanic_train_y), number=100)/100)
print("Score time")
print(timeit.timeit(lambda: svm_credit.score(titanic_test_X, titanic_test_y), number=100)/100)

# endregion