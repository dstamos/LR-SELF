from src.utilities import Struct, DataHandler, pairwise_loss_separate, pairwise_embedding
from src.optimisation import mf_conjugate_grad
import numpy as np


if __name__ == "__main__":

    seed = 10
    regul_param = 10
    rank = 10
    train_perc = 0.5
    val_perc = 0.2
    opt_tol = 1e-03
    dataset = 'sushi'  # ml100k, jester1, jester2, jester3, sushi

    np.random.seed(seed)

    settings_dict = {'seed': seed,
                     'dataset': dataset,
                     'regul_param': regul_param,
                     'rank': rank,
                     'train_perc': train_perc,
                     'val_perc': val_perc,
                     'opt_tol': opt_tol}

    settings = Struct(**settings_dict)
    data = DataHandler(settings)

    w_matrix = mf_conjugate_grad(data, settings)

    tr_perf = pairwise_loss_separate(data.tr_embedded, data.tr_embedded_mask, data, w_matrix)

    y_embedded, mask_embedded = pairwise_embedding(data.val_matrix, data.n_rows, data.n_cols)
    val_perf = pairwise_loss_separate(y_embedded, mask_embedded, data, w_matrix)

    y_embedded, mask_embedded = pairwise_embedding(data.test_matrix, data.n_rows, data.n_cols)
    test_perf = pairwise_loss_separate(y_embedded, mask_embedded, data, w_matrix)

    print('tr perf: %7.4f | val perf: %7.4f | test perf: %7.4f' % (tr_perf, val_perf, test_perf))
