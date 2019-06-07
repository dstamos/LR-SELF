import time
import numpy as np
import scipy as sp
from numpy.linalg import norm
from scipy.sparse.linalg import norm as norm_sparse
from scipy.optimize import fmin_cg
from scipy.sparse import csr_matrix


def mf_conjugate_grad(data, settings):

    n_cols = data.n_cols_embedded
    n_side_info = data.n_side_info
    mask = data.tr_embedded_mask
    tr_embedded = data.tr_embedded
    rank = settings.rank

    a_factor = np.random.randn(n_side_info, rank)
    b_factor = np.random.randn(n_cols, rank)

    idx_mask = sp.sparse.find(mask)
    y_tr_vec = tr_embedded[idx_mask[0], idx_mask[1]]
    temp_bucket = {'y_tr_vec': y_tr_vec,
                   'idx_mask': idx_mask,
                   'side_info': data.side_info,
                   'A': a_factor,
                   'B': b_factor,
                   'temp_iter': 0,
                   'loss_part': loss_part_fun(data.side_info @ a_factor, b_factor, tr_embedded, mask, idx_mask)}

    conv_tol = 10**-3
    curr_cost = 10**10
    opt_tol = 1e-03
    t = time.time()

    while True:
        prev_cost = curr_cost

        a = a_factor.flatten()
        b = b_factor.flatten()
        x = np.append(a, b)

        x = fmin_cg(mf_obj_wrapper, x, mf_grad_wrapper,
                    args=(data, settings, temp_bucket),
                    disp=True,
                    retall=False,
                    full_output=False,
                    maxiter=10000,
                    gtol=opt_tol)

        a_factor = np.reshape(x[:n_side_info * rank], (n_side_info, rank))
        b_factor = np.reshape(x[n_side_info * rank:], (n_cols, rank))

        curr_cost = mf_obj(a_factor, b_factor, tr_embedded, mask, idx_mask, settings.regul_param, temp_bucket)
        diff = abs(prev_cost - curr_cost) / prev_cost
        print('rank: %4d | diff: %14.6f | time: %f' % (rank, diff, time.time() - t))
        if diff < conv_tol:
            break

    w_matrix = data.side_info @ a_factor @ b_factor.T

    return w_matrix


def mf_obj(a_factor, b_factor, y_matrix, mask, idx_mask, param1, temp_bucket):
    a_prev = temp_bucket['A']
    b_prev = temp_bucket['B']

    if np.all(a_prev == a_factor) and np.all(b_prev == b_factor):
        loss_part = temp_bucket['loss_part']
    else:
        side_info = temp_bucket['side_info']
        loss_part = loss_part_fun(side_info @ a_factor, b_factor, y_matrix, mask, idx_mask)

        temp_bucket['loss_part'] = loss_part
        temp_bucket['A'] = a_factor
        temp_bucket['B'] = b_factor

    obj = 0.5 * norm_sparse(loss_part, 'fro') ** 2 + 0.5 * param1 * (norm(a_factor, 'fro') ** 2 + norm(b_factor, 'fro') ** 2)

    return obj


def mf_obj_wrapper(x, *args):

    data, settings, temp_bucket = args

    rank = settings.rank

    idx_mask = temp_bucket['idx_mask']

    n_cols = data.n_cols_embedded
    mask = data.tr_embedded_mask
    tr_embedded_mask = data.tr_embedded
    n_side_info = data.n_side_info

    a_factor = np.reshape(x[:n_side_info*rank], (n_side_info, rank))
    b_factor = np.reshape(x[n_side_info*rank:], (n_cols, rank))

    obj = mf_obj(a_factor, b_factor, tr_embedded_mask, mask, idx_mask, settings.regul_param, temp_bucket)

    temp_bucket['temp_iter'] = temp_bucket['temp_iter'] + 1
    if (temp_bucket['temp_iter'] % 1 == 0) or (temp_bucket['temp_iter'] == 1):
        print('total obj computations: %5d | obj (in or out of line search): %20.6f' % (temp_bucket['temp_iter'], obj))

    return obj


def mf_grad(a_factor, b_factor, y_matrix, mask, idx_mask, param1, side_info, temp_bucket):

    a_prev = temp_bucket['A']
    b_prev = temp_bucket['B']

    if np.all(a_prev == a_factor) and np.all(b_prev == b_factor):
        loss_part = temp_bucket['loss_part']
    else:
        loss_part = loss_part_fun(side_info @ a_factor, b_factor, y_matrix, mask, idx_mask)
        temp_bucket['loss_part'] = loss_part
        temp_bucket['A'] = a_factor
        temp_bucket['B'] = b_factor

    grad_wrt_a = -side_info.T @ (loss_part @ b_factor) + param1 * a_factor
    grad_wrt_b = -loss_part.T @ (side_info @ a_factor) + param1 * b_factor

    grad_wrt_a = grad_wrt_a.ravel()
    grad_wrt_b = grad_wrt_b.ravel()
    grad = np.append(grad_wrt_a, grad_wrt_b)

    return grad


def mf_grad_wrapper(x, *args):
    data, settings, temp_bucket = args
    rank = settings.rank

    idx_mask = temp_bucket['idx_mask']

    n_cols = data.n_cols_embedded
    n_side_info = data.n_side_info
    side_info = data.side_info
    mask = data.tr_embedded_mask
    tr_embedded = data.tr_embedded

    a_factor = np.reshape(x[:n_side_info*rank], (n_side_info, rank))
    b_factor = np.reshape(x[n_side_info*rank:], (n_cols, rank))

    grad = mf_grad(a_factor, b_factor, tr_embedded, mask, idx_mask, settings.regul_param, side_info, temp_bucket)

    return grad


def loss_part_fun(a_factor, b_factor, y_matrix, mask, idx_mask):
    n_rows, n_cols = mask.shape

    n_total = len(idx_mask[0])
    w_bar_vec = np.empty(n_total)
    idx = 0
    while idx < n_total:
        chunk = np.min([1000000, n_total-idx])
        w_bar_vec[idx:idx+chunk] = np.sum(a_factor[idx_mask[0][idx:idx + chunk], :].T *
                                          b_factor[idx_mask[1][idx:idx + chunk], :].T, axis=0)
        idx = idx + chunk

    w_matrix = csr_matrix((w_bar_vec, (idx_mask[0], idx_mask[1])), shape=(n_rows, n_cols))
    loss_part = y_matrix - w_matrix

    return loss_part
