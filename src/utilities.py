import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import igraph
import time
import pandas as pd
import pickle
import os
import logging
from joblib import Parallel, delayed


def data_gen(data_settings, embedding='None'):

    train_perc = 0.5
    val_perc = 0.2

    # temp = sio.loadmat('datasets/ml100k.mat')
    if data_settings['dataset_idx'] == 1:
        temp = sio.loadmat('datasets/ml100kSparse.mat')
        full_matrix = temp['fullMatrix'].astype(float)
    elif data_settings['dataset_idx'] == 2:
        temp = sio.loadmat('datasets/ml1mSparse.mat')
        full_matrix = temp['fullMatrix'].astype(float)
    elif data_settings['dataset_idx'] == 3:
        temp = sio.loadmat('datasets/jester1Sparse.mat')
        full_matrix = temp['jester1Sparse'].astype(float)
    elif data_settings['dataset_idx'] == 4:
        temp = sio.loadmat('datasets/jester2Sparse.mat')
        full_matrix = temp['jester2Sparse'].astype(float)
    elif data_settings['dataset_idx'] == 5:
        temp = sio.loadmat('datasets/jester3Sparse.mat')
        full_matrix = temp['jester3Sparse'].astype(float)
    elif data_settings['dataset_idx'] == 6:
        temp = sio.loadmat('datasets/sushi.mat')
        full_matrix = temp['M'].astype(float)
        train_perc = 0.35
        val_perc = 0.35
    elif data_settings['dataset_idx'] == 7:
        # from skmultilearn.dataset import load_from_arff
        # import arff
        # arff.DENSE
        # path_to_arff_file = 'datasets/multilabel/' + data_settings['dataset'] + '/bibtex.arff'
        # label_count = 159
        # label_location = "end"
        # arff_file_is_sparse = False
        # X, y = load_from_arff(
        #     path_to_arff_file,
        #     label_count=label_count,
        #     label_location=label_location,
        #     load_sparse=arff_file_is_sparse
        # )

        file_path =  "datasets/multilabel/" + data_settings['dataset'] + "/bibtex.pckl"

        # pickle.dump([X, y], open(file_path, "wb"))

        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 8:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        # from sklearn.preprocessing import normalize
        #
        # X = normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
        
        from sklearn.preprocessing import scale

        X = lil_matrix(scale(X.toarray(), axis=0))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 9:
        from skmultilearn.dataset import load_from_arff
        import arff
        dname = data_settings['dataset']
        path_to_arff_file = 'datasets/multilabel/' + dname + '/' + dname + '.arff'
        label_count = 174
        label_location = "end"
        arff_file_is_sparse = False
        X, y = load_from_arff(
            path_to_arff_file,
            label_count=label_count,
            label_location=label_location,
            load_sparse=arff_file_is_sparse
        )

        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'

        pickle.dump([X, y], open(file_path, "wb"))

        # X, y = pickle.load(open(file_path, "rb"))
    elif data_settings['dataset_idx'] == 10:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 11:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 12:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 13:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 14:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 15:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2
    elif data_settings['dataset_idx'] == 12:
        dname = data_settings['dataset']
        file_path = "datasets/multilabel/" + dname + '/' + dname + '.pckl'
        X, y = pickle.load(open(file_path, "rb"))

        train_perc = 0.6
        val_perc = 0.2



    n_rows, n_cols = y.shape

    shuffled_rows = np.random.permutation(n_rows)
    tr_idxs = shuffled_rows[:int(round(n_rows*train_perc))]
    val_idxs = shuffled_rows[len(tr_idxs):len(tr_idxs)+int(round(n_rows*val_perc))]
    test_idxs = shuffled_rows[len(tr_idxs)+len(val_idxs):]

    tr_matrix = y[tr_idxs, :]
    val_matrix = y[val_idxs, :]
    test_matrix = y[test_idxs, :]

    X = lil_matrix(np.append(X.toarray(), np.ones(X.shape[0]).reshape(X.shape[0], 1), axis=1))

    side_info_tr = X[tr_idxs, :]
    side_info_val = X[val_idxs, :]
    side_info_test = X[test_idxs, :]

    n_rows, n_cols = tr_matrix.shape
    data_settings['n_rows'] = n_rows
    data_settings['n_cols'] = n_cols
    #############
    #############

    data = {}

    data['side_info'] = side_info_tr.toarray()
    data['side_info_val'] = side_info_val.toarray()
    data['side_info_test'] = side_info_test.toarray()
    data_settings['n_sideinfo'] = side_info_tr.shape[1]

    data['mask'] = np.ones(tr_matrix.shape)
    data['tr_matrix'] = tr_matrix
    data['val_matrix'] = val_matrix
    data['test_matrix'] = test_matrix

    # Y_tr_embedded, mask_tr_embedded = pairwise_embedding(tr_matrix, data_settings)
    Y_tr_embedded, mask_embedded = fscore_embedding(tr_matrix)
    data['Y_tr_embedded'] = Y_tr_embedded
    # data['mask_tr_embedded'] = mask_embedded
    data['mask_tr_embedded'] = np.ones(mask_embedded.shape)

    data_settings['n_rows_embedded'] = n_rows
    data_settings['n_cols_embedded'] = data['Y_tr_embedded'].shape[1]

    data['data_settings'] = data_settings

    return data, data_settings


def fscore_embedding(Y):
    n_rows, n_cols = Y.shape

    t = time.time()

    Y_embedded = lil_matrix((n_rows, int(n_cols * n_cols + 1)))
    mask_embedded = lil_matrix((n_rows, int(n_cols * n_cols + 1)))
    for c_row in range(n_rows):

        # t = time.time()
        # row_sum = np.sum(Y[c_row, :])
        # # slow method
        # A = np.zeros((n_cols, n_cols))
        # for j in range(n_cols):
        #     non_zero_check = Y[c_row, j] != 0
        #     for k in range(n_cols):
        #         for l in range(n_cols):
        #             if non_zero_check and row_sum == l:
        #                 Pjl = 1
        #                 A[j, k] = A[j, k] + Pjl / (l + (k + 1))
        #             else:
        #                 pass
        # Y_embedded[c_row, :-1] = np.ndarray.flatten(A)
        # print(time.time() - t)

        #TODO Double check flatten row-wise stuff
        ###################

        A = np.zeros((n_cols,n_cols))
        support_indices = np.nonzero(Y[c_row,:])[1]
        support_size = len(support_indices)

        if support_size > 0:
            A[:,support_indices] = 1 / (support_size + np.array(range(n_cols)).reshape(n_cols, 1) + 1)
        else:
            Y_embedded[c_row, -1] = 1

        Y_embedded[c_row,:-1] = A.flatten()

        if c_row % 1000 == 0:
            print('embedding row: %5d | time: %f' % (c_row, time.time() - t))
            logging.info('embedding row: %5d | time: %f', c_row, time.time() - t)

        # separate case for y = 0
        if np.sum(np.abs(Y[c_row,:])) == 0:
            Y_embedded[c_row, -1] = 1

    mask_embedded[np.nonzero(Y_embedded)] = 1

    return Y_embedded, mask_embedded


def pairwise_embedding(Y, data_settings):
    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']

    mask_embedded = lil_matrix((n_rows, int(n_cols * (n_cols - 1) / 2)))
    Y_embedded = lil_matrix((n_rows, int(n_cols * (n_cols - 1) / 2)))

    t = time.time()

    for c_row in range(n_rows):

        temp_matrix = Y[c_row, :].toarray() - Y[c_row, :].toarray().T
        # outer product of the support
        # temp_mask = abs(np.sign(Y[c_row, :].toarray()) * np.sign(Y[c_row, :].toarray().T))
        temp_mask = np.zeros(temp_matrix.shape)
        temp_mask[np.where(temp_matrix)] = 1
        temp_matrix = temp_mask * temp_matrix

        iu = np.triu_indices(n_cols, 1)

        temp_mask_vect = temp_mask[iu]
        temp_matrix_vect = temp_matrix[iu]

        ###############
        a = np.where(temp_matrix_vect)[0]
        b = np.where(temp_mask_vect)[0]
        if len(a) != len(b):
            print('fucked embedding: ' + str(c_row))
            k = 1

        Y_embedded[c_row, temp_mask_vect>0 ] = temp_matrix_vect[temp_mask_vect>0]
        mask_embedded[c_row, temp_mask_vect>0] = temp_mask_vect[temp_mask_vect>0]

        # Y_embedded[:, temp_mask_vect>0 ] = temp_matrix_vect[temp_mask_vect>0]
        # mask_embedded[:, temp_mask_vect>0] = temp_mask_vect[temp_mask_vect>0]

        # break
        if c_row % 100 == 0:
            print('embedding row: %5d | time: %f' % (c_row, time.time() - t))
            logging.info('embedding row: %5d | time: %f', c_row, time.time() - t)

    return Y_embedded, mask_embedded


def get_side_info(data, data_settings):
    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']

    # load up user and item side information
    side_info = np.random.randn(n_rows, 5)

    data_settings['n_sideinfo'] = side_info.shape[1]

    data['side_info'] = side_info
    return data


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def decoding(row, data_settings):
    tt = time.time()
    n_cols = data_settings['n_cols']

    i_upper = np.triu_indices(n_cols, 1)
    adjacency_matrix = np.zeros((n_cols, n_cols))
    adjacency_matrix[i_upper] = row

    a = pd.DataFrame(adjacency_matrix.tolist())

    # Get the values as np.array, it's more convenient.
    A = a.values

    A_neg = np.zeros(A.shape)
    A_neg[np.where(A < 0)] = A[np.where(A < 0)]

    A[np.where(A < 0)] = 0

    i_lower = np.tril_indices(n_cols, -1)
    A[i_lower] = np.abs(A_neg.T[i_lower])
    # print('top search set time: %7.2f' % (time.time() - tt))

    # A = np.array([[0, 2, 0, 0, 0, 0, 0],
    #               [0, 0, 5, 0, 0, 0, 0],
    #               [0, 0, 0, 3, 1, 0, 0],
    #               [8, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 4],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               ])

    g = igraph.Graph.Adjacency((A > 0).tolist())
    tt = time.time()
    g.es['weight'] = A[A.nonzero()]
    # print('graph creation time: %7.2f' % (time.time() - tt))

    # print(g.es['weight'][:3])
    # logging.info(g.es['weight'][:3])
    tt = time.time()
    idx_to_drop = g.feedback_arc_set(weights=g.es['weight'])

    # print(idx_to_drop[:3])
    # logging.info(idx_to_drop[:3])

    g.delete_edges(idx_to_drop)

    adjacency_matrix = pd.DataFrame(g.get_adjacency(attribute='weight').data).values

    ordering = g.topological_sorting(mode='OUT')
    # print('bottom part time: %7.2f' % (time.time() - tt))

    # print(ordering[:3])
    # logging.info(ordering[:3])

    return adjacency_matrix, ordering


def pairwise_loss(data, data_settings, W):
    Y_tr_embedded = data['Y_tr_embedded']
    Y_val_embedded = data['Y_val_embedded']
    Y_test_embedded = data['Y_test_embedded']
    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']
    mask_tr_embedded = data['mask_tr_embedded']
    mask_val_embedded = data['mask_val_embedded']
    mask_test_embedded = data['mask_test_embedded']

    # TODO only create the graph on these or the entire?

    # W_tr = W * mask_tr_embedded.toarray()
    # W_val = W * mask_val_embedded.toarray()
    # W_test = W * mask_test_embedded.toarray()

    def get_relative_ranking_from_ordering(ordering):
        ordering = np.array(ordering)
        ranking = np.zeros(ordering.shape)
        index_vec = np.arange(1, len(ordering) + 1)
        ranking[ordering] = index_vec
        ranking = ranking.astype(int)
        ranking = np.reshape(ranking, (len(ranking), 1))

        rel_rankings = np.sign(ranking - ranking.T)
        return rel_rankings

    def get_rel_ratings_from_true_embedding(y_embedded, rel_rankings, n_cols=n_cols):
        rel_ratings = np.zeros(rel_rankings.shape)
        Y_temp = y_embedded.toarray()
        iu = np.triu_indices(n_cols, 1)
        rel_ratings[iu] = Y_temp
        return rel_ratings

    tr_perf = 0
    val_perf = 0
    test_perf = 0

    n_rows_tr = n_rows
    n_rows_val = n_rows
    n_rows_test = n_rows
    t = time.time()
    for c_row in range(n_rows):

        adjacency_matrix, ordering = decoding(W[c_row, :] * mask_tr_embedded[c_row, :].toarray(), data_settings)
        rel_rankings = get_relative_ranking_from_ordering(ordering)
        rel_ratings = get_rel_ratings_from_true_embedding(Y_tr_embedded[c_row, :], rel_rankings)
        temp = rel_ratings * rel_rankings
        l_min = np.sum(rel_ratings * -np.sign(rel_ratings))
        l_max = np.sum(rel_ratings * np.sign(rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows_tr = n_rows_tr - 1
            # print('skipping training row %d' % c_row)
            logging.info('skipping training row %d', c_row)
        else:
            c_tr_perf = (np.sum(temp) - l_min) / (l_max - l_min)
            tr_perf = tr_perf + c_tr_perf

        #

        adjacency_matrix, ordering = decoding(W[c_row, :] * mask_val_embedded[c_row, :].toarray(), data_settings)
        rel_rankings = get_relative_ranking_from_ordering(ordering)
        rel_ratings = get_rel_ratings_from_true_embedding(Y_val_embedded[c_row, :], rel_rankings)
        temp = rel_ratings * rel_rankings
        l_min = np.sum(rel_ratings * -np.sign(rel_ratings))
        l_max = np.sum(rel_ratings * np.sign(rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows_val = n_rows_val - 1
            # print('skipping val row %d' % c_row)
            logging.info('skipping val row %d', c_row)
        else:
            c_val_perf = (np.sum(temp) - l_min) / (l_max - l_min)
            val_perf = val_perf + c_val_perf

        #

        adjacency_matrix, ordering = decoding(W[c_row, :] * mask_test_embedded[c_row, :].toarray(), data_settings)
        rel_rankings = get_relative_ranking_from_ordering(ordering)
        rel_ratings = get_rel_ratings_from_true_embedding(Y_test_embedded[c_row, :], rel_rankings)
        temp = rel_ratings * rel_rankings
        l_min = np.sum(rel_ratings * -np.sign(rel_ratings))
        l_max = np.sum(rel_ratings * np.sign(rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows_test = n_rows_test - 1
            print('skipping test row %d' % c_row)
            logging.info('skipping test row %d', c_row)
        else:
            c_test_perf = (np.sum(temp) - l_min) / (l_max - l_min)
            test_perf = test_perf + c_test_perf

        if c_row % 100 == 0:
            print('row: %4d | time lapsed: %7.2f' % (c_row, time.time() - t))
            logging.info('row: %4d | time lapsed: %7.2f', c_row, time.time() - t)

    tr_perf = tr_perf / n_rows_tr
    val_perf = val_perf / n_rows_val
    test_perf = test_perf / n_rows_test

    return tr_perf, val_perf, test_perf


def pairwise_loss_separate(Y_embedded, mask_embedded, data_settings, W):
    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']

    def get_relative_ranking_from_ordering(ordering):
        ordering = np.array(ordering)
        ranking = np.zeros(ordering.shape)
        index_vec = np.arange(1, len(ordering) + 1)
        ranking[ordering] = index_vec
        ranking = ranking.astype(int)
        ranking = np.reshape(ranking, (len(ranking), 1))

        rel_rankings = np.sign(ranking - ranking.T)
        return rel_rankings

    def get_rel_ratings_from_true_embedding(y_embedded, rel_rankings, n_cols=n_cols):
        rel_ratings = np.zeros(rel_rankings.shape)
        Y_temp = y_embedded.toarray()
        iu = np.triu_indices(n_cols, 1)
        rel_ratings[iu] = Y_temp
        return rel_ratings

    perf = 0

    n_rows_counter = n_rows
    t = time.time()
    for c_row in range(n_rows):

        adjacency_matrix, ordering = decoding(W[c_row, :] * mask_embedded[c_row, :].toarray(), data_settings)
        rel_rankings = get_relative_ranking_from_ordering(ordering)
        rel_ratings = get_rel_ratings_from_true_embedding(Y_embedded[c_row, :], rel_rankings)
        temp = rel_ratings * rel_rankings
        l_min = np.sum(rel_ratings * -np.sign(rel_ratings))
        l_max = np.sum(rel_ratings * np.sign(rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows_counter = n_rows_counter - 1
            logging.info('skipping training row %d', c_row)
        else:
            c_perf = (np.sum(temp) - l_min) / (l_max - l_min)
            perf = perf + c_perf

        if c_row % 100 == 0:
            print('row: %4d | time lapsed: %7.2f' % (c_row, time.time() - t))
            logging.info('row: %4d | time lapsed: %7.2f', c_row, time.time() - t)

    perf = perf / n_rows_counter

    return perf


def fscore_from_embeddings(Y_true, side_info, data_settings, W):

    n_rows = Y_true.shape[0]
    # if data_settings['embedding'] != 'SELF':
    Y_embedded_pred = side_info @ W
    # else:
    #     Y_embedded_pred = W

    score = 0

    t = time.time()
    for c_row in range(n_rows):

        Y_pred_vec = decoding_fscore_embeddings(Y_embedded_pred[c_row, :], data_settings)
        Y_true_vec = Y_true[c_row, :].toarray().ravel()

        # tp = len(np.nonzero(Y_pred_vec * Y_true_vec)[0])
        # tp_fp = len(np.nonzero(Y_pred_vec)[0])
        # tp_fn = len(np.nonzero(Y_true_vec)[0])
        # new_fscore = 2 * tp / (tp_fp + tp_fn)

        from sklearn.metrics import f1_score
        new_fscore = f1_score(Y_true_vec, Y_pred_vec)

        score = score + new_fscore

        if c_row % 1000 == 0:
            print('row: %4d | time lapsed: %7.2f' % (c_row, time.time() - t))
            logging.info('row: %4d | time lapsed: %7.2f', c_row, time.time() - t)

    score = score / n_rows

    return score


def decoding_fscore_embeddings(row, data_settings):

    n_cols = data_settings['n_cols']

    A = np.reshape(row[:-1], [n_cols, n_cols])

    running_max = row[-1]
    running_fstar = []
    for k in range(n_cols):

        # sorted_col = np.sort(A[:, k])[::-1]
        # sorted_col_idx = np.argsort(A[:, k])[::-1]

        sorted_row = np.sort(A[k, :])[::-1]
        sorted_row_idx = np.argsort(A[k, :])[::-1]

        temp = np.sum(sorted_row[:k+1])

        if running_max < temp:
            running_max = temp
            running_fstar = sorted_row_idx[:k+1]

    y = np.zeros(n_cols)
    if len(running_fstar) > 0:
        y[running_fstar] = 1

    return y


def perf_check(W, data):

    train_idxs = data['train_idxs']
    val_idxs = data['val_idxs']
    test_idxs = data['test_idxs']
    Ytr = data['tr_matrix'].toarray()
    Yval = data['val_matrix'].toarray()
    Ytest = data['test_matrix'].toarray()
    n_rows = W.shape[0]

    tr_perf = [0] * n_rows
    val_perf = [0] * n_rows
    test_perf = [0] * n_rows
    for c_row in range(n_rows):
        tr_perf[c_row] = ndcg_score(Ytr[c_row, train_idxs[c_row]], W[c_row, train_idxs[c_row]])
        val_perf[c_row] = ndcg_score(Yval[c_row, val_idxs[c_row]], W[c_row, val_idxs[c_row]])
        test_perf[c_row] = ndcg_score(Ytest[c_row, test_idxs[c_row]], W[c_row, test_idxs[c_row]])

    tr_perf = np.mean(tr_perf)
    val_perf = np.mean(val_perf)
    test_perf = np.mean(test_perf)

    return tr_perf, val_perf, test_perf


def straight_pairwise_loss(data, data_settings, W):
    Y_tr_embedded = data['Y_tr_embedded']
    Y_val_embedded = data['Y_val_embedded']
    Y_test_embedded = data['Y_test_embedded']
    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']

    from scipy.stats import rankdata

    tr_perf = 0
    val_perf = 0
    test_perf = 0
    for c_row in range(n_rows):
        # retrieve rank from ratings and compute the diff matrix
        predicted_ratings = W[c_row, :]
        ranking = rankdata(np.abs(predicted_ratings - np.max(predicted_ratings)), method='ordinal')
        ranking = np.reshape(ranking, (len(ranking), 1))
        # rankings and ratings have flipped order so we take the -1
        rel_rankings = -1 * np.sign(ranking - ranking.T)

        rel_ratings = np.zeros(rel_rankings.shape)
        Y_temp = Y_tr_embedded[c_row, :].toarray()
        iu = np.triu_indices(n_cols, 1)
        rel_ratings[iu] = Y_temp

        temp = rel_ratings * rel_rankings
        l_min = np.sum(rel_ratings * -np.sign(rel_ratings))
        l_max = np.sum(rel_ratings * np.sign(rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows = n_rows - 1
            print('skipping row %d' % c_row)
            logging.info('skipping row %d', c_row)
            continue
        else:
            c_tr_perf = (np.sum(temp) - l_min) / (l_max - l_min)
        tr_perf = tr_perf + c_tr_perf

    tr_perf = tr_perf / n_rows
    val_perf = tr_perf
    test_perf = tr_perf

    return tr_perf, val_perf, test_perf


def save_results(results, data_settings, training_settings):
    filename = training_settings['filename']
    foldername = training_settings['foldername']

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    f = open(foldername + '/' + filename + ".pckl", 'wb')
    pickle.dump(results, f)
    pickle.dump(data_settings, f)
    pickle.dump(training_settings, f)
    f.close()

