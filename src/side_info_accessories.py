import numpy as np
import warnings
import scipy.io as sio
import math


def extract_side_info(mat):
    n_rows, n_cols = mat.shape
    # kNN item-based features
    item_similarity, r_item_pred = cosine_similarity_items(mat)
    user_similarity, r_user_pred = cosine_similarity_users(mat)

    #######################################################
    temp = sio.loadmat('datasets/' + 'ml100kSparse_seed_' + str(1) + '.mat')
    tr_matrix = temp['Mtr'].astype(float)
    val_matrix = temp['Mva'].astype(float)
    test_matrix = temp['Mts'].astype(float)
    side_info = temp['X'].astype(float)

    import matplotlib.pyplot as plt
    plt.imshow(side_info.toarray())
    plt.pause(0.1)
    #######################################################

    features = []

    knn_item_based_1 = np.zeros((n_cols, n_rows))
    knn_item_based_2 = np.zeros(n_cols)
    movie_features = np.zeros((n_cols, 15))

    for movie_index in range(n_cols):
        h1, h2 = knn_item_based_features(item_similarity, r_item_pred, movie_index)
        knn_item_based_1[movie_index, :] = h1
        knn_item_based_2[movie_index] = h2

        temp_item = item_based_stat_features(tr_matrix, movie_index)
        movie_features[movie_index, :] = temp_item

    # for each user we create a query
    for user_index in range(n_rows):
        knn_user_based_1, knn_user_based_2 = knn_user_based_features(user_similarity, r_user_pred, user_index)

        # get the movies actually rated by the user
        movie_list = np.where(mat[user_index, :].toarray() != 0)[1]

        # get the ratings for this user
        r = mat[user_index, movie_list].toarray()

        # get the feature vectors
        q = []

        # user-based statistical features
        temp_user_feature = user_based_stat_features(mat, user_index)

        # for each movie we build a feature vector
        for movie_index in range(n_cols):
            # create the feature vector for the user/movie pair
            temp_f = []

            # user features
            temp_item = temp_user_feature
            temp_item = temp_item / np.linalg.norm(temp_item)
            temp_f = temp_f + list(temp_item)

            # movie features
            temp_item = movie_features[movie_index, :]
            temp_item = temp_item / np.linalg.norm(temp_item)
            temp_f = temp_f + list(temp_item)

            # knn movie features
            temp_item = [knn_item_based_1[movie_index, user_index], knn_item_based_2[movie_index]]
            temp_item = temp_item / np.linalg.norm(temp_item)
            temp_f = temp_f + list(temp_item)

            # knn user features
            temp_item = [knn_user_based_1[movie_index], knn_user_based_1[movie_index]]
            temp_item = temp_item / np.linalg.norm(temp_item)
            temp_f = temp_f + list(temp_item)

            temp_f = [0 if math.isnan(i) else i for i in temp_f]

            # create the matrix of documents
            q.append(temp_f)

            if movie_index % 10 == 0:
                print('user: %5d | item: %4d' % (user_index, movie_index))
        features.append(q)
    return features


def cosine_similarity_users(mat):
    # http://proceedings.mlr.press/v18/xie12a/xie12a.pdf
    # Weights based on popularity equation (2)
    frequency_in_dataset = np.sum(mat != 0, axis=1)
    w_weights = np.array(1 / np.log2(3 + frequency_in_dataset)).ravel()

    # weighted cosine similarity equation (3)
    temp_mask = mat != 0
    temp_mat = np.multiply(temp_mask.toarray(), np.reshape(w_weights, [len(w_weights), 1]))
    nominator = temp_mat @ temp_mat.T
    denominator = np.outer(np.sqrt(np.sum(temp_mat, 1)), np.sqrt(np.sum(temp_mat, 1)))
    denominator[denominator == 0] = 10**-2
    s_weighted_cosine_similarity = nominator / denominator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_pred = (s_weighted_cosine_similarity @ mat) / np.reshape(np.sum(s_weighted_cosine_similarity, 0), [len(s_weighted_cosine_similarity), 1])

    # new matrix of ratings on which we do kNN equation (4)
    row_norms = np.sqrt(np.nansum(r_pred, 1))

    r_pred[np.isnan(r_pred)] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_cosine_similarity = (r_pred @ r_pred.T) / (row_norms * row_norms.T)
    s_cosine_similarity[np.isnan(s_cosine_similarity)] = 0
    s_cosine_similarity = s_cosine_similarity - np.diag(np.diag(s_cosine_similarity))
    return s_cosine_similarity, r_pred


def cosine_similarity_items(mat):
    # http://proceedings.mlr.press/v18/xie12a/xie12a.pdf
    # Weights based on popularity equation (2)
    frequency_in_dataset = np.sum(mat != 0, axis=0)
    w_weights = np.array(1 / np.log2(3 + frequency_in_dataset)).ravel()

    # weighted cosine similarity equation (3)
    temp_mask = mat != 0
    temp_mat = np.multiply(temp_mask.toarray(), w_weights)
    nominator = temp_mat.T @ temp_mat
    denominator = np.outer(np.sqrt(np.sum(temp_mat, 0)), np.sqrt(np.sum(temp_mat, 0)))
    denominator[denominator == 0] = 10**-2
    s_weighted_cosine_similarity = nominator / denominator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_pred = (mat * s_weighted_cosine_similarity) / np.sum(s_weighted_cosine_similarity, 1)

    # new matrix of ratings on which we do kNN equation (4)
    row_norms = np.sqrt(np.nansum(r_pred, 0))

    r_pred[np.isnan(r_pred)] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_cosine_similarity = (r_pred.T @ r_pred) / (row_norms.T * row_norms)
    s_cosine_similarity[np.isnan(s_cosine_similarity)] = 0
    s_cosine_similarity = s_cosine_similarity - np.diag(np.diag(s_cosine_similarity))
    return s_cosine_similarity, r_pred


def user_based_stat_features(mat, user_index):
    # http://proceedings.mlr.press/v18/xie12a/xie12a.pdf
    # Table 4
    mat = mat.toarray()
    n_features = 14
    user_based_features = np.zeros(n_features)
    max_rating = np.max(mat)

    observed_ratings = mat[user_index, np.where(mat[user_index, :] != 0)[0]]
    n_movies_rated = len(observed_ratings)

    user_based_features[0] = np.mean(observed_ratings)
    user_based_features[1] = len(np.where(mat[user_index, :] > np.floor(0.8 * max_rating))[0]) / n_movies_rated
    user_based_features[2] = np.quantile(observed_ratings, 0.01)
    user_based_features[3] = np.quantile(observed_ratings, 0.05)
    user_based_features[4] = np.quantile(observed_ratings, 0.25)
    user_based_features[5] = np.quantile(observed_ratings, 0.50)
    user_based_features[6] = np.quantile(observed_ratings, 0.75)
    user_based_features[7] = np.quantile(observed_ratings, 0.95)
    user_based_features[8] = np.quantile(observed_ratings, 0.99)
    user_based_features[9] = np.std(observed_ratings)
    user_based_features[10] = user_based_features[0] - (user_based_features[9] / np.sqrt(n_movies_rated))
    user_based_features[11] = user_based_features[0] + (user_based_features[9] / np.sqrt(n_movies_rated))
    user_based_features[12] = np.max(observed_ratings)
    user_based_features[13] = np.min(observed_ratings)
    return user_based_features


def item_based_stat_features(mat, movie_index):
    # http://proceedings.mlr.press/v18/xie12a/xie12a.pdf
    # Table 3
    mat = mat.toarray()
    n_features = 15
    item_based_features = np.zeros(n_features)
    max_rating = np.max(mat)

    observed_ratings = mat[np.where(mat[:, movie_index] != 0)[0], movie_index]
    n_users_rated = len(observed_ratings)

    item_based_features[0] = np.count_nonzero(observed_ratings)
    if item_based_features[0] == 0:
        return item_based_features
    item_based_features[1] = np.max(observed_ratings)
    item_based_features[2] = np.min(observed_ratings)
    item_based_features[3] = np.mean(observed_ratings)
    item_based_features[4] = np.std(observed_ratings)
    item_based_features[5] = item_based_features[3] - (item_based_features[4] / np.sqrt(item_based_features[0]))
    item_based_features[6] = item_based_features[3] + (item_based_features[4] / np.sqrt(item_based_features[0]))
    item_based_features[7] = np.quantile(observed_ratings, 0.01)
    item_based_features[8] = np.quantile(observed_ratings, 0.05)
    item_based_features[9] = np.quantile(observed_ratings, 0.25)
    item_based_features[10] = np.quantile(observed_ratings, 0.5)
    item_based_features[11] = np.quantile(observed_ratings, 0.75)
    item_based_features[12] = np.quantile(observed_ratings, 0.95)
    item_based_features[13] = np.quantile(observed_ratings, 0.99)
    item_based_features[14] = len(np.where(mat[:, movie_index] > np.floor(0.8 * max_rating))[0]) / n_users_rated
    return item_based_features


def knn_item_based_features(similarity_mat, r_pred, movie_index):
    k = 20
    argmax = np.argsort(similarity_mat[movie_index, :])[::-1]
    most_similar_indexes = argmax[:k]

    knn_item_based_1 = np.mean(r_pred[:, most_similar_indexes], 1)
    knn_item_based_2 = np.mean(similarity_mat[movie_index, most_similar_indexes])

    return knn_item_based_1, knn_item_based_2


def knn_user_based_features(similarity_mat, r_pred, user_index):
    k = 20
    argmax = np.argsort(similarity_mat[user_index, :])[::-1]
    most_similar_indexes = argmax[:k]

    knn_user_based_1 = np.mean(r_pred[most_similar_indexes, :], 0)
    knn_user_based_2 = np.mean(similarity_mat[user_index, most_similar_indexes])

    return knn_user_based_1, knn_user_based_2
