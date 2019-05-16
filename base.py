from src.utilities import data_gen, pairwise_loss, pairwise_loss_separate, \
    save_results, straight_pairwise_loss, pairwise_embedding, fscore_embedding, fscore_from_embeddings
from src.optimisation import fista_matrix, mf_cong_grad, frob_gd, self_carlo_vector_valued
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import logging


def main(data, data_settings, training_settings, param1):
    print('main')
    logging.info("main")



    # W = fista_matrix(data, data_settings, training_settings, param1)
    if data_settings['embedding'] == 'SELF':
        W, results = self_carlo_vector_valued(data, data_settings, training_settings, param1)
    else:
        W, results = mf_cong_grad(data, data_settings, training_settings, param1, data_settings['embedding'])


    # W = frob_gd(data, data_settings, training_settings, param1)

    return W, results


if __name__ == "__main__":

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        dataset_idx = int(sys.argv[2])
        lambda_idx = int(sys.argv[3])
        embedding = int(sys.argv[4])
        rank = int(sys.argv[5])
    else:
        seed = 1
        dataset_idx = 14
        lambda_idx = 4
        embedding = 5    # 1 pairwise, 2 None, 3 sideInfo
        rank = 5

    # param1_range = [10 ** float(i) for i in np.linspace(-2, 3.2, 30)] base
    # param1_range = [10 ** float(i) for i in np.linspace(-6, 2, 30)]  # SELF
    param1_range = [10 ** float(i) for i in np.linspace(-6, 4, 30)]  # multi-labelstuff
    # param1_range = [10 ** float(i) for i in np.linspace(-10, 6, 5)]  # multi-labelstuff test stuff
    # param1_range = [10 ** float(i) for i in np.linspace(-1, 2, 30)]
    # param1_range = [10 ** float(i) for i in np.linspace(1, 7, 30)]  # sideinfo

    data_settings = {}
    data_settings['seed'] = seed
    if embedding == 1:
        data_settings['embedding'] = 'pairwise'
    elif embedding == 2:
        data_settings['embedding'] = 'None'
    elif embedding == 3:
        data_settings['embedding'] = 'sideInfo'
    elif embedding == 4:
        data_settings['embedding'] = 'SELF'
    elif embedding == 5:
        data_settings['embedding'] = 'multilabel_fscore'

    data_settings['dataset_idx'] = dataset_idx
    if dataset_idx == 1:
        data_settings['dataset'] = 'ml100k'
    elif dataset_idx == 2:
        data_settings['dataset'] = 'ml1m'
    elif dataset_idx == 3:
        data_settings['dataset'] = 'jester1'
    elif dataset_idx == 4:
        data_settings['dataset'] = 'jester2'
    elif dataset_idx == 5:
        data_settings['dataset'] = 'jester3'
    elif dataset_idx == 6:
        data_settings['dataset'] = 'sushi'
    elif dataset_idx == 7:
        data_settings['dataset'] = 'bibtex'
    elif dataset_idx == 8:
        data_settings['dataset'] = 'birds'
    elif dataset_idx == 9:
        data_settings['dataset'] = 'CAL500'
    elif dataset_idx == 10:
        data_settings['dataset'] = 'corel5k'
    elif dataset_idx == 11:
        data_settings['dataset'] = 'enron'
    elif dataset_idx == 12:
        data_settings['dataset'] = 'mediamill'
    elif dataset_idx == 13:
        data_settings['dataset'] = 'medical'
    elif dataset_idx == 14:
        data_settings['dataset'] = 'scene'
    elif dataset_idx == 15:
        data_settings['dataset'] = 'yeast'

    np.random.seed(data_settings['seed'])

    filename = "seed_" + str(seed) + '-lambda_' + str(param1_range[lambda_idx]) + '-rank_' + str(rank)
    foldername = 'results/' + data_settings['dataset'] + '/' + data_settings['embedding']

    # old_stdout = sys.stdout
    # log_file = open(data_settings['dataset'] + "-seed_" + str(seed) + '-lambda_' +
    #                 str(param1_range[lambda_idx]) + ".log", "w")
    # sys.stdout = log_file

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename='logs/' + data_settings['dataset'] + '-' + data_settings['embedding'] +
                                 "-seed_" + str(seed) + '-lambda_' +
                                 str(param1_range[lambda_idx]) + ""
                                                                 ""
                                                                 ".log",
                        level=logging.INFO)
    logging.info("Starting log")

    training_settings = {}
    training_settings['param1_range'] = param1_range
    training_settings['filename'] = filename
    training_settings['foldername'] = foldername

    training_settings['rank'] = rank

    data, data_settings = data_gen(data_settings, data_settings['embedding'])

    all_tr_perf = [None] * len(param1_range)
    all_val_perf = [None] * len(param1_range)
    all_test_perf = [None] * len(param1_range)
    best_val = -10**10

    # for idx, _ in enumerate(param1_range):
    idx = lambda_idx

    param1 = param1_range[idx]
    print('Working on param1: %20.15f' % training_settings['param1_range'][idx])
    logging.info('Working on param1: %20.15f', training_settings['param1_range'][idx])




    # Y = data['tr_matrix']
    # from src.utilities import decoding_fscore_embeddings
    #
    # Y_embed, _ = fscore_embedding(Y)
    # Y_recovered = np.zeros((Y.shape))
    # for c_row in range(Y_embed.shape[0]):
    #     Y_recovered[c_row, :] = decoding_fscore_embeddings(Y_embed[c_row, :].toarray().ravel(), data_settings)
    #
    # print('True:')
    # print(Y[0:16, :].toarray())
    #
    # print(' ')
    # print('Recovered:')
    # print(Y_recovered[0:16, :].astype(int))



    W, results = main(data, data_settings, training_settings, param1)

    # import pickle
    # f = open('temp_results_investigating_error.pckl', 'wb')
    # pickle.dump(W, f)
    # pickle.dump(results, f)
    # f = open('temp_results_investigating_error.pckl', 'rb')
    # W = pickle.load(f)
    # results = pickle.load(f)
    # f.close()

    # time.sleep(1)

    tr_perf = fscore_from_embeddings(data['tr_matrix'], data['side_info'], data_settings, W)
    val_perf = fscore_from_embeddings(data['val_matrix'], data['side_info_val'], data_settings, W)
    test_perf = fscore_from_embeddings(data['test_matrix'], data['side_info_test'], data_settings, W)






    # if data_settings['embedding'] == 'None':
    #     tr_perf, val_perf, test_perf = straight_pairwise_loss(data, data_settings, W)
    # else:
    #     # tr_perf, val_perf, test_perf = pairwise_loss(data, data_settings, W)
    #
    #     tr_perf = pairwise_loss_separate(data['Y_tr_embedded'], data['mask_tr_embedded'], data_settings, W)
    #
    #     Y_embedded, mask_embedded = pairwise_embedding(data['val_matrix'], data_settings)
    #     val_perf = pairwise_loss_separate(Y_embedded, mask_embedded, data_settings, W)
    #     Y_embedded, mask_embedded = pairwise_embedding(data['test_matrix'], data_settings)
    #     test_perf = pairwise_loss_separate(Y_embedded, mask_embedded, data_settings, W)

    print('training error: %7.5f' % tr_perf)
    print('validation error: %7.5f' % val_perf)
    print('test error: %7.5f' % test_perf)
    logging.info('training error: %7.5f', tr_perf)
    logging.info('validation error: %7.5f', val_perf)
    logging.info('test error: %7.5f', test_perf)

    all_tr_perf[idx] = tr_perf
    all_val_perf[idx] = val_perf
    all_test_perf[idx] = test_perf
    if val_perf > best_val:
        best_val = val_perf
        best_W = W

    results['tr_perf'] = tr_perf
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf

    save_results(results, data_settings, training_settings)

    # sys.stdout = old_stdout
    # log_file.close()

    print("done")
    logging.info("done")