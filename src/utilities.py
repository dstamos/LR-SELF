import scipy.io as sio
import numpy as np
from scipy.sparse import lil_matrix
import igraph
import time
import pandas as pd


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
        full_matrix = temp['full_matrix'].astype(float)

        n_rows, n_cols = full_matrix.shape
        self.n_rows = n_rows
        self.n_cols = n_cols

        tr_matrix = lil_matrix((n_rows, n_cols))
        val_matrix = lil_matrix((n_rows, n_cols))
        test_matrix = lil_matrix((n_rows, n_cols))
        train_idxs = [0] * n_rows
        val_idxs = [0] * n_rows
        test_idxs = [0] * n_rows

        for c_row in range(n_rows):
            all_ratings = np.nonzero(full_matrix[c_row:])[1]
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
            print('embedding row: %5d | time: %8.3f' % (c_row, time.time() - t))

    return y_embedded, mask_embedded


def decoding(row, n_cols):

    i_upper = np.triu_indices(n_cols, 1)
    adjacency_matrix = np.zeros((n_cols, n_cols))
    adjacency_matrix[i_upper] = row

    a_matrix = adjacency_matrix

    a_neg = np.zeros(a_matrix.shape)
    a_neg[np.where(a_matrix < 0)] = a_matrix[np.where(a_matrix < 0)]

    a_matrix[np.where(a_matrix < 0)] = 0

    i_lower = np.tril_indices(n_cols, -1)
    a_matrix[i_lower] = np.abs(a_neg.T[i_lower])

    g = igraph.Graph.Adjacency((a_matrix > 0).tolist())
    g.es['weight'] = a_matrix[a_matrix.nonzero()]

    idx_to_drop = g.feedback_arc_set(weights=g.es['weight'])

    g.delete_edges(idx_to_drop)

    adjacency_matrix = pd.DataFrame(g.get_adjacency(attribute='weight').data).values

    ordering = g.topological_sorting(mode='OUT')

    return adjacency_matrix, ordering


def pairwise_loss_separate(y, mask_embedded, data, w_matrix):
    n_rows = data.n_rows
    n_cols = data.n_cols

    def get_relative_ranking_from_ordering(ordering):
        ordering = np.array(ordering)
        ranking = np.zeros(ordering.shape)
        index_vec = np.arange(1, len(ordering) + 1)
        ranking[ordering] = index_vec
        ranking = ranking.astype(int)
        ranking = np.reshape(ranking, (len(ranking), 1))

        rel_rankings = np.sign(ranking - ranking.T)
        return rel_rankings

    def get_rel_ratings_from_true_embedding(y_embedded, rel_rankings):
        rel_ratings = np.zeros(rel_rankings.shape)
        y_temp = y_embedded.toarray()
        iu = np.triu_indices(n_cols, 1)
        rel_ratings[iu] = y_temp
        return rel_ratings

    perf = 0

    n_rows_counter = n_rows
    t = time.time()
    for c_row in range(n_rows):

        adjacency_matrix, curr_ordering = decoding(w_matrix[c_row, :] * mask_embedded[c_row, :].toarray(), n_cols)
        curr_rel_rankings = get_relative_ranking_from_ordering(curr_ordering)
        curr_rel_ratings = get_rel_ratings_from_true_embedding(y[c_row, :], curr_rel_rankings)
        temp = curr_rel_ratings * curr_rel_rankings
        l_min = np.sum(curr_rel_ratings * -np.sign(curr_rel_ratings))
        l_max = np.sum(curr_rel_ratings * np.sign(curr_rel_ratings))

        if (l_min == 0) and (l_max == 0):
            n_rows_counter = n_rows_counter - 1
        else:
            c_perf = (np.sum(temp) - l_min) / (l_max - l_min)
            perf = perf + c_perf

        if c_row % 100 == 0:
            print('row: %4d | time lapsed: %7.2f' % (c_row, time.time() - t))

    perf = perf / n_rows_counter

    return perf
