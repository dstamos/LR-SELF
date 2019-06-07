from src.utilities import Struct, DataHandler, pairwise_loss_separate, pairwise_embedding
from src.optimisation import mf_conjugate_grad
import numpy as np


if __name__ == "__main__":

    seed = 10
    regul_param = 10
    rank = 10
    train_perc = 0.5
    val_perc = 0.2

    np.random.seed(seed)

    settings_dict = {'seed': seed,
                     'embedding': 'sideInfo',  # 'pairwise', 'sideInfo'
                     'dataset': 'ml100k',  # ml100k, jester1, jester2, jester3, sushi
                     'regul_param': regul_param,
                     'rank': rank,
                     'train_perc': train_perc,
                     'val_perc': val_perc}

    settings = Struct(**settings_dict)
    data = DataHandler(settings)

    w_matrix = mf_conjugate_grad(data, settings)
    print(w_matrix)
    print(w_matrix.shape)
    exit()
    tr_perf = pairwise_loss_separate(data['Y_tr_embedded'], data['mask_tr_embedded'], data_settings, w_matrix)

    Y_embedded, mask_embedded = pairwise_embedding(data['val_matrix'], data_settings)
    val_perf = pairwise_loss_separate(Y_embedded, mask_embedded, data_settings, w_matrix)
    Y_embedded, mask_embedded = pairwise_embedding(data['test_matrix'], data_settings)
    test_perf = pairwise_loss_separate(Y_embedded, mask_embedded, data_settings, w_matrix)
