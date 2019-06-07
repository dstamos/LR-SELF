import scipy.io as sio
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import igraph
import time
import pandas as pd
from src.side_info_accessories import extract_side_info


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DataHandler:
    def __init__(self, settings):
        self.n_rows = None
        self.n_cols = None
        self.tr_matrix = None
        self.tr_matrix_mask = None
        self.val_matrix = None
        self.test_matrix = None
        self.test_matrix = None
        self.tr_embedded = None
        self.tr_embedded_mask = None
        self.n_rows_embedded = None
        self.n_cols_embedded = None
        self.n_side_info = None
        self.side_info = None

        temp = sio.loadmat('datasets/' + settings.dataset + '.mat')
        full_matrix = temp['fullMatrix'].astype(float)

        n_rows, n_cols = full_matrix.shape
        self.n_rows = n_rows
        self.n_cols = n_cols

        tr_matrix = lil_matrix((n_rows, n_cols))
        val_matrix = lil_matrix((n_rows, n_cols))
        test_matrix = lil_matrix((n_rows, n_cols))
        train_idxs = [0] * n_rows
        val_idxs = [0] * n_rows
        test_idxs = [0] * n_rows

        t = time.time()
        for c_row in range(n_rows):
            all_ratings = np.nonzero(full_matrix[c_row,:])[1]
            all_ratings_shuffled = np.random.permutation(all_ratings)
            n_all_ratings = len(all_ratings_shuffled)

            n_train = int(np.floor(settings.train_perc * n_all_ratings))
            n_val = int(np.ceil(settings.val_perc * n_all_ratings))

            train_idx = all_ratings_shuffled[:n_train]
            val_idx = all_ratings_shuffled[n_train+1:n_train+n_val]
            test_idx = all_ratings_shuffled[n_train + n_val + 1:]

            train_idxs[c_row] = train_idx
            val_idxs[c_row] = val_idx
            test_idxs[c_row] = test_idx

            tr_matrix[c_row, train_idx] = full_matrix[c_row, train_idx]
            val_matrix[c_row, val_idx] = full_matrix[c_row, val_idx]
            test_matrix[c_row, test_idx] = full_matrix[c_row, test_idx]

            if c_row % 500 == 0:
                print('sampling row: %5d | time: %f' % (c_row, time.time() - t))

        side_info = np.zeros((n_rows, n_cols + 1))
        side_info[:, :-1] = tr_matrix.toarray()
        side_info = side_info / np.tile(np.sqrt(1 + np.sum(tr_matrix.toarray()**2, 1)).reshape(n_rows, 1), (1, n_cols+1))
        n_side_info = side_info.shape[1]

        tr_embedded, tr_embedded_mask = pairwise_embedding(tr_matrix, n_rows, n_cols)
        tr_embedded = tr_embedded
        tr_embedded_mask = tr_embedded_mask
        n_rows_embedded = n_rows
        n_cols_embedded = tr_embedded.shape[1]

        tr_matrix_mask = tr_matrix != 0
        tr_matrix_mask.astype(np.int)
        val_matrix = val_matrix
        test_matrix = test_matrix

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.tr_matrix = tr_matrix
        self.tr_matrix_mask = tr_matrix_mask
        self.val_matrix = val_matrix
        self.test_matrix = test_matrix
        self.test_matrix = test_matrix
        self.tr_embedded = tr_embedded
        self.tr_embedded_mask = tr_embedded_mask
        self.n_rows_embedded = n_rows_embedded
        self.n_cols_embedded = n_cols_embedded
        self.n_side_info = n_side_info
        self.side_info = side_info


def pairwise_embedding(y, n_rows, n_cols):

    mask_embedded = lil_matrix((n_rows, int((n_cols * (n_cols - 1) / 2))))
    y_embedded = lil_matrix((n_rows, int((n_cols * (n_cols - 1) / 2))))

    t = time.time()

    for c_row in range(n_rows):

        temp_matrix = y[c_row, :].toarray() - y[c_row, :].toarray().T
        temp_mask = np.zeros(temp_matrix.shape)
        temp_mask[np.where(temp_matrix)] = 1
        temp_matrix = temp_mask * temp_matrix

        iu = np.triu_indices(n_cols, 1)

        temp_mask_vect = temp_mask[iu]
        temp_matrix_vect = temp_matrix[iu]

        y_embedded[c_row, temp_mask_vect > 0] = temp_matrix_vect[temp_mask_vect > 0]
        mask_embedded[c_row, temp_mask_vect > 0] = temp_mask_vect[temp_mask_vect > 0]

        if c_row % 100 == 0:
            print('embedding row: %5d | time: %f' % (c_row, time.time() - t))

    return y_embedded, mask_embedded


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
