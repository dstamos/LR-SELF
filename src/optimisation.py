import time
import numpy as np
import scipy as sp
from numpy.linalg import norm
from scipy.sparse.linalg import norm as norm_sparse
from scipy.optimize import fmin_cg
from scipy.sparse import csr_matrix


def mf_conjugate_grad(data, settings):

    n_cols = data.n_cols_embedded
    n_sideinfo = data.n_sideinfo
    mask = data.mask_tr_embedded
    y_tr_embedded = data.Y_tr_embedded
    rank = settings.rank

    a_factor = np.random.randn(n_sideinfo, rank)
    b_factor = np.random.randn(n_cols, rank)

    idx_mask = sp.sparse.find(mask)
    y_tr_vec = y_tr_embedded[idx_mask[0], idx_mask[1]]
    temp_bucket = {'y_tr_vec': y_tr_vec,
                   'idx_mask': idx_mask,
                   'side_info': side_info,
                   'A': a_factor,
                   'B': b_factor}

    print('about to compute the loss part for the first time')
    temp_bucket['loss_part'] = loss_part_fun(data.side_info @ a_factor, b_factor, y_tr_embedded, mask, idx_mask)

    print('loss part computed')

    conv_tol = 10**-3
    curr_cost = 10**10
    opt_tol = 1e-03
    t = time.time()

    while True:
        prev_cost = curr_cost

        a = a_factor.flatten()
        b = b_factor.flatten()
        x = np.append(a, b)

        print('time pre opt: %f' % (time.time() - t))

        x = fmin_cg(mf_obj_wrapper, x, mf_grad_wrapper,
                    args=(data, settings),
                    disp=True,
                    retall=False,
                    full_output=False,
                    maxiter=10000,
                    gtol=opt_tol)

        a_factor = np.reshape(x[:n_sideinfo * rank], (n_sideinfo, rank))
        b_factor = np.reshape(x[n_sideinfo * rank:], (n_cols, rank))

        curr_cost = mf_obj(a_factor, b_factor, y_tr_embedded, mask, y_tr_vec, idx_mask, settings.regul_param, temp_bucket)
        diff = abs(prev_cost - curr_cost) / prev_cost
        print('rank: %4d | diff: %12.6f | time: %f' % (rank, diff, time.time() - t))
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

    data, data_settings, param1 = args

    rank = data_settings['rank']
    embedding = data_settings['embedding']

    # TODO fix
    y_tr_vec = data['y_tr_vec']
    idx_mask = data['idx_mask']

    if embedding == 'None':
        n_rows = data_settings['n_rows']
        n_cols = data_settings['n_cols']
        mask = data['mask']
        Y = data['tr_matrix']
    elif embedding == 'pairwise':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']
    elif embedding == 'sideInfo':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        n_sideinfo = data_settings['n_sideinfo']
        side_info = data['side_info']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']

    temp_iter = data_settings['temp_iter']

    if embedding == 'sideInfo':
        A = np.reshape(x[:n_sideinfo*rank], (n_sideinfo, rank))
        B = np.reshape(x[n_sideinfo*rank:], (n_cols, rank))
    else:
        A = np.reshape(x[:n_rows*rank], (n_rows, rank))
        B = np.reshape(x[n_rows*rank:], (n_cols, rank))

    if embedding == 'sideInfo':
        obj = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)
    else:
        obj = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)

    temp_iter = temp_iter + 1
    data_settings['temp_iter'] = temp_iter
    if (temp_iter % 1 == 0) or (temp_iter == 1):
        print('obj computations: %5d | obj: %14.6f' % (temp_iter, obj))

    return obj


def mf_grad(A, B, Y, mask, y_tr_vec, idx_mask, param1, embedding, side_info, data):

    A_prev = data['A']
    B_prev = data['B']

    if np.all(A_prev == A) and np.all(B_prev == B):
        print('same A and B in the grad')
        loss_part = data['loss_part']
    else:
        if embedding == 'sideInfo':
            loss_part = loss_part_fun(side_info @ A, B, Y, mask, idx_mask)
        else:
            loss_part = loss_part_fun(A, B, Y, mask, idx_mask)
        data['loss_part'] = loss_part

        data['A'] = A
        data['B'] = B

    if embedding == 'sideInfo':
        grad_wrt_A = -side_info.T @ (loss_part @ B) + param1 * A
        grad_wrt_B = -loss_part.T @ (side_info @ A) + param1 * B
    else:
        grad_wrt_A = -loss_part @ B + param1*A
        grad_wrt_B = -loss_part.T @ A + param1*B

    grad_wrt_a = grad_wrt_A.ravel()
    grad_wrt_b = grad_wrt_B.ravel()
    grad = np.append(grad_wrt_a, grad_wrt_b)

    return grad


def mf_grad_wrapper(x, *args):
    data, data_settings, param1 = args
    rank = data_settings['rank']
    embedding = data_settings['embedding']

    y_tr_vec = data['y_tr_vec']
    idx_mask = data['idx_mask']

    if embedding == 'None':
        n_rows = data_settings['n_rows']
        n_cols = data_settings['n_cols']
        side_info = None
        mask = data['mask']
        Y = data['tr_matrix']
    elif embedding == 'pairwise':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        side_info = None
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']
    elif embedding == 'sideInfo':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        n_sideinfo = data_settings['n_sideinfo']
        side_info = data['side_info']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']


    if embedding == 'sideInfo':
        A = np.reshape(x[:n_sideinfo*rank], (n_sideinfo, rank))
        B = np.reshape(x[n_sideinfo*rank:], (n_cols, rank))
    else:
        A = np.reshape(x[:n_rows*rank], (n_rows, rank))
        B = np.reshape(x[n_rows*rank:], (n_cols, rank))

    grad = mf_grad(A, B, Y, mask, y_tr_vec, idx_mask, param1, embedding, side_info, data)
    return grad


def loss_part_fun(A, B, Y, mask, idx_mask):
    n_rows, n_cols = mask.shape
    # A_bar = A[idx_mask[0], :].T
    # B_bar = B[idx_mask[1], :].T
    # W_bar = A_bar * B_bar

    # t = time.time()
    # w_bar_vec = np.sum(A[idx_mask[0], :].T * B[idx_mask[1], :].T, axis=0)
    # print(time.time() - t)


    # t = time.time()
    n_total = len(idx_mask[0])
    w_bar_vec = np.empty(n_total)
    idx = 0
    while idx < n_total:
        chunk = np.min([1000000, n_total-idx])
        # print('s: %5d | e: %5d' %(idx, idx+chunk))
        w_bar_vec[idx:idx+chunk] = np.sum(A[idx_mask[0][idx:idx+chunk], :].T *
                                           B[idx_mask[1][idx:idx+chunk], :].T, axis=0)
        idx = idx + chunk
    # print(time.time() - t)



    W = csr_matrix((n_rows, n_cols))

    W[idx_mask[0], idx_mask[1]] = w_bar_vec

    loss_part = Y - W

    # loss_part = mask.multiply(Y - A @ B.T)

    return loss_part