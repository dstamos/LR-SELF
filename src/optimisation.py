import time
import numpy as np
import logging
import scipy as sp
from numpy.linalg import norm
from scipy.sparse.linalg import norm as norm_sparse
from scipy.optimize import fmin_cg
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from src.utilities import save_results


def fista_matrix(data, data_settings, training_settings, param1):

    Y = data['tr_matrix']
    train_idxs = data['train_idxs']

    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']
    alpha = csr_matrix((n_rows, n_cols))
    W = csr_matrix((n_rows, n_cols))

    Lipschitz = 1

    penalty = lambda x: param1 * norm(x, ord='nuc')
    prox = lambda x: np.sign(x) * np.maximum(abs(x) - param1 / Lipschitz, 0)
    loss = lambda x: 1/2 * norm(Y - x, ord='fro')**2
    grad = lambda x: -(Y - x)

    curr_iter = 0
    diff = 10**8
    conv_tol = 10**-5
    curr_cost = loss(W) + penalty(W)
    theta = 1
    objectives = []

    mask = Y != 0
    mask.astype(np.int)

    method = 'fista'

    t = time.time()
    while (curr_iter < 10**5) and (diff > conv_tol):
        curr_iter = curr_iter + 1
        prev_cost = curr_cost

        if method == 'fista':
            prev_W = alpha

            step_size = (1 / Lipschitz)
            search_point = alpha - step_size * grad(mask*alpha)

            U, s, Vt = np.linalg.svd(search_point)
            s = prox(s)
            S = csr_matrix((U.shape[0], Vt.shape[1]))
                # np.zeros((U.shape[0], Vt.shape[1]))
            S[:len(s), :len(s)] = np.diag(s)
            W = U @ S @ Vt

            theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
            rho = 1 - theta + np.sqrt(1 - theta)
            alpha = rho * W - (rho - 1) * prev_W

            curr_cost = loss(mask*alpha) + penalty(alpha)
        else:
            step_size = (1 / Lipschitz)
            search_point = W - step_size * grad(mask*W)

            U, s, Vt = np.linalg.svd(search_point)
            s = prox(s)
            S = csr_matrix((U.shape[0], Vt.shape[1]))
            S[:len(s), :len(s)] = np.diag(s)
            W = U @ S @ Vt

            curr_cost = loss(mask*W) + penalty(W)

        objectives.append(curr_cost)
        diff = abs(prev_cost - curr_cost) / prev_cost


        if (time.time() - t > 0):
            t = time.time()
            print('iter: %6d | cost: %20.8f ~ tol: %18.15f | step: %12.10f' % (curr_iter, curr_cost, diff, step_size))
    print('iter: %6d | cost: %20.8f ~ tol: %18.15f | step: %12.10f' % (curr_iter, curr_cost, diff, step_size))

    return W


def frob_gd(data, data_settings, training_settings, param1):

    Y = data['tr_matrix']
    mask = Y != 0
    mask.astype(np.int)

    n_rows = data_settings['n_rows']
    n_cols = data_settings['n_cols']
    W = np.zeros((n_rows, n_cols))

    obj = lambda x: 0.5 * norm(mask*(Y - x), ord='fro')**2 + 0.5 * param1 * norm(x, ord='fro')**2
    grad = lambda x: mask*(x - Y) + param1 * x

    curr_iter = 0
    diff = 10**8
    conv_tol = 10**-8
    curr_cost = obj(W)
    objectives = []

    t = time.time()
    while (curr_iter < 10**5) and (diff > conv_tol):
        curr_iter = curr_iter + 1
        prev_cost = curr_cost
        prev_W =  W

        step_size = 10
        grad_W = grad(prev_W)
        W = prev_W - step_size * grad_W
        curr_cost = obj(W)

        while curr_cost > prev_cost - 0.5 * step_size * norm(prev_W, ord='fro')**2:
            step_size = 0.5 * step_size
            W = prev_W - step_size * grad_W
            curr_cost = obj(W)

        objectives.append(curr_cost)
        diff = abs(prev_cost - curr_cost) / prev_cost

        if (time.time() - t > 0):
            t = time.time()
            print('iter: %6d | cost: %20.15f ~ tol: %18.15f | step: %12.10f' % (curr_iter, curr_cost, diff, step_size))
    print('iter: %6d | cost: %20.15f ~ tol: %18.15f | step: %12.10f' % (curr_iter, curr_cost, diff, step_size))

    return W


def self_carlo_mf(data, data_settings, training_settings, param1):
    n_rows = data_settings['n_rows_embedded']
    n_cols = data_settings['n_cols_embedded']
    n_sideinfo = data_settings['n_sideinfo']
    mask = data['mask_tr_embedded']
    Y = data['Y_tr_embedded']
    side_info = data['side_info']
    mask = data['mask']
    tr_matrix = data['tr_matrix']

    W = csr_matrix((n_rows, n_cols))

    K = (side_info @ side_info.T)
    n_users = data_settings['n_rows']
    n_items = data_settings['n_cols']

    W = np.zeros((n_rows, n_cols))
    tt = time.time()
    c_col = -1
    for c_i_item in range(n_items):
        for c_j_item in range(c_i_item+1, n_items):
            # non_zero_i = sp.sparse.csr_matrix.nonzero(mask[:, c_i_item])[0]
            # non_zero_j = sp.sparse.csr_matrix.nonzero(mask[:, c_j_item])[0]

            non_zero_i = np.nonzero(mask[:, c_i_item])[0]
            non_zero_j = np.nonzero(mask[:, c_j_item])[0]


            tr_indeces = np.intersect1d(non_zero_i, non_zero_j)

            tmpY = tr_matrix[tr_indeces, c_i_item] - tr_matrix[tr_indeces, c_j_item]

            if len(tr_indeces) < tr_matrix.shape[1]:
                tmpK = K[np.ix_(tr_indeces, tr_indeces)]
                tmpC = np.linalg.solve(tmpK + len(tr_indeces) * param1 * np.eye(tmpK.shape[1]), tmpY.toarray())
                tmpG = K[:, tr_indeces] @ tmpC
            else:
                tmpX = side_info[tr_indeces, :]
                try:
                    tmpW = np.linalg.solve(tmpX.T @ tmpX + len(tr_indeces) * param1 * np.eye(tmpX.shape[1]), (tmpX.T @ tmpY).toarray())
                except:
                    tmpW = np.linalg.solve(tmpX.T @ tmpX + len(tr_indeces) * param1 * np.eye(tmpX.shape[1]), (tmpX.T @ tmpY))
                tmpG = side_info @ tmpW
            c_col = c_col + 1
            print('(%5d, %5d) | W col: %5d | time: %7.2f' % (c_i_item, c_j_item, c_col, time.time() - tt))
            W[:, c_col] = -tmpG.ravel()

    results = {}

    print(np.linalg.norm(W, ord='fro'))
    print(np.linalg.norm(W, ord='nuc'))

    return W, results


def self_carlo_vector_valued(data, data_settings, training_settings, param1):
    n_rows = data_settings['n_rows_embedded']
    n_cols = data_settings['n_cols_embedded']
    n_sideinfo = data_settings['n_sideinfo']
    Y = data['Y_tr_embedded']
    side_info = data['side_info']
    tr_matrix = data['tr_matrix']

    W = np.zeros((n_sideinfo, n_cols))

    # K = (side_info @ side_info.T)
    n_users = data_settings['n_rows']
    n_items = data_settings['n_cols']

    X = side_info

    W = np.linalg.solve(X.T @ X + n_rows * param1 * np.eye(X.shape[1]), (X.T @ Y))



    #
    #
    # tt = time.time()
    # c_col = -1
    # for c_i_item in range(n_items):
    #     for c_j_item in range(c_i_item+1, n_items):
    #         # non_zero_i = sp.sparse.csr_matrix.nonzero(mask[:, c_i_item])[0]
    #         # non_zero_j = sp.sparse.csr_matrix.nonzero(mask[:, c_j_item])[0]
    #
    #         non_zero_i = np.nonzero(mask[:, c_i_item])[0]
    #         non_zero_j = np.nonzero(mask[:, c_j_item])[0]
    #
    #
    #         tr_indeces = np.intersect1d(non_zero_i, non_zero_j)
    #
    #         tmpY = tr_matrix[tr_indeces, c_i_item] - tr_matrix[tr_indeces, c_j_item]
    #
    #         if len(tr_indeces) < tr_matrix.shape[1]:
    #             tmpK = K[np.ix_(tr_indeces, tr_indeces)]
    #             tmpC = np.linalg.solve(tmpK + len(tr_indeces) * param1 * np.eye(tmpK.shape[1]), tmpY.toarray())
    #             tmpG = K[:, tr_indeces] @ tmpC
    #         else:
    #             tmpX = side_info[tr_indeces, :]
    #             try:
    #                 tmpW = np.linalg.solve(tmpX.T @ tmpX + len(tr_indeces) * param1 * np.eye(tmpX.shape[1]), (tmpX.T @ tmpY).toarray())
    #             except:
    #                 tmpW = np.linalg.solve(tmpX.T @ tmpX + len(tr_indeces) * param1 * np.eye(tmpX.shape[1]), (tmpX.T @ tmpY))
    #             tmpG = side_info @ tmpW
    #         c_col = c_col + 1
    #         print('(%5d, %5d) | W col: %5d | time: %7.2f' % (c_i_item, c_j_item, c_col, time.time() - tt))
    #         W[:, c_col] = -tmpG.ravel()

    results = {}

    print(np.linalg.norm(W, ord='fro'))
    print(np.linalg.norm(W, ord='nuc'))

    return W, results



def mf_cong_grad(data, data_settings, training_settings, param1, embedding=None):

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
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']
    elif embedding == 'multilabel_fscore':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        n_sideinfo = data_settings['n_sideinfo']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']

    # rank = np.min([n_rows, n_cols])

    # for jester
    # rank = 20
    # for ml100k
    # rank = 20
    # rank = 50
    rank = training_settings['rank']

    if data_settings['dataset_idx'] == 1:
        rank = rank
    elif data_settings['dataset_idx'] == 2:
        rank = rank
    elif data_settings['dataset_idx'] == 3:
        rank = rank
    elif data_settings['dataset_idx'] == 4:
        rank = rank
    elif data_settings['dataset_idx'] == 5:
        rank = rank
    elif data_settings['dataset_idx'] == 6:
        rank = rank
    elif data_settings['dataset_idx'] == 7:
        rank = rank

    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
        # todo generalise when you have side info for both
        A = np.random.randn(n_sideinfo, rank)
        B = np.random.randn(n_cols, rank)
    else:
        A = np.random.randn(n_rows, rank)
        B = np.random.randn(n_cols, rank)

    idx_mask = sp.sparse.find(mask)
    y_tr_vec = Y[idx_mask[0], idx_mask[1]]
    data['y_tr_vec'] = y_tr_vec
    data['idx_mask'] = idx_mask
    data['A'] = A
    data['B'] = B

    print('about to compute the loss part for the first time')
    logging.info('about to compute the loss part for the first time')
    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
        data['loss_part'] = loss_part_fun(data['side_info'] @ A, B, Y, mask, idx_mask)
    else:
        data['loss_part'] = loss_part_fun(A, B, Y, mask, idx_mask)

    print('loss part computed')
    logging.info('loss part computed')

    diff = 10**10
    conv_tol = 10**-3
    curr_cost = 10**10
    t = time.time()

    while True:
        data_settings['temp_iter'] = 0
        prev_cost = curr_cost

        data_settings['rank'] = rank
        a = A.flatten()
        b = B.flatten()
        x = np.append(a, b)

        if embedding == 'None':
            opt_tol = 1e-05
        else:
            if data_settings['dataset_idx'] == 1:
                opt_tol = 1e-02
                # opt_tol = 1e-04
            elif data_settings['dataset_idx'] == 2:
                opt_tol = 1e-01
            elif data_settings['dataset_idx'] == 3:
                opt_tol = 1e-03
            elif data_settings['dataset_idx'] == 4:
                opt_tol = 1e-03
            elif data_settings['dataset_idx'] == 5:
                opt_tol = 1e-03
            elif data_settings['dataset_idx'] == 6:
                opt_tol = 1e-05
            else:
                opt_tol = 1e-05

        print('time pre opt: %f' % (time.time() - t))
        logging.info('time pre opt: %f', time.time() - t)


        # from scipy.optimize import approx_fprime
        #
        # def numerical_grad(x, f, *args):
        #     data, data_settings, param1 = args
        #     numel = len(x)
        #     I = np.eye(numel)
        #     eps = 1e-8
        #     grad = np.zeros(numel)
        #     for i in range(numel):
        #         eps_vec = I[:, i] * eps
        #         grad[i] = (f(x + eps_vec, data, data_settings, param1) - f(x - eps_vec, data, data_settings, param1)) / (2 * eps)
        #     return grad
        #
        # my_grad = mf_grad_wrapper(x, data, data_settings, param1)
        # scipy_grad = approx_fprime(x, mf_obj_wrapper, np.sqrt(np.finfo(np.float).eps), data, data_settings, param1)
        # numerical_shit = numerical_grad(x, mf_obj_wrapper, data, data_settings, param1)
        #
        # import matplotlib.pyplot as plt
        # plt.stem((my_grad - scipy_grad) / scipy_grad)
        # plt.pause(1)
        #
        # print(my_grad[-20:])
        # print(scipy_grad[-20:])
        # print(numerical_shit[-20:])
        #
        # print(my_grad[:20])
        # print(scipy_grad[:20])
        # print(numerical_shit[:20])


        # n = size(numgrad);
        # I = eye(n, n);
        # for i = 1:size(numgrad)
        # eps_vec = I(:, i) *eps;
        # numgrad(i) = (J(theta + eps_vec) - J(theta - eps_vec)) / (2 * eps);
        # end

        x = fmin_cg(mf_obj_wrapper, x, mf_grad_wrapper,
                    args=(data, data_settings, param1),
                    disp=True,
                    retall=False,
                    full_output=False,
                    maxiter=500000,
                    gtol=opt_tol)

        if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
            A = np.reshape(x[:n_sideinfo * rank], (n_sideinfo, rank))
            B = np.reshape(x[n_sideinfo * rank:], (n_cols, rank))
        else:
            A = np.reshape(x[:n_rows*rank], (n_rows, rank))
            B = np.reshape(x[n_rows*rank:], (n_cols, rank))

        if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
            curr_cost = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)
        else:
            curr_cost = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)
        diff = abs(prev_cost - curr_cost) / prev_cost
        print('rank: %4d | diff: %12.6f | time: %f' % (rank, diff, time.time() - t))
        logging.info('rank: %4d | diff: %12.6f | time: %f', rank, diff, time.time() - t)

        results = {}
        # results['A'] = A
        # results['B'] = B
        # save_results(results, data_settings, training_settings)

        break

        # rank_step = 100
        # loss = - 0.5 * mask.multiply(Y - A @ B.T)
        # u, s, vt = svds(loss, k=rank_step)
        # if (rank <= np.max([n_rows, n_cols]) and (diff > conv_tol)):
        # if (rank <= 50) and (diff > conv_tol):
        #     rank = np.min([rank + rank_step, np.max([n_rows, n_cols])])
        #     A_added = u
        #     B_added = vt.T
        #
        #     starting_cost = curr_cost
        #     curr_cost = 10**10
        #     step_size = 100
        #     while starting_cost < curr_cost:
        #         A_temp = np.append(A, step_size * A_added, axis=1)
        #         B_temp = np.append(B, -step_size * B_added, axis=1)
        #
        #         curr_cost = mf_obj(A_temp, B_temp, Y, mask, param1)
        #         cost_change = starting_cost - curr_cost
        #         print('descent step: %10.5f | change: %15.3f' % (step_size, cost_change))
        #         logging.info('descent step: %10.5f | change: %15.3f',
        #                                           step_size, cost_change)
        #         step_size = 0.5 * step_size
        #     A = A_temp
        #     B = B_temp
        # else:
        #     break
    if embedding == 'sideInfo':
        W = data['side_info'] @ A @ B.T
    else:
        W = A @ B.T

    print(np.linalg.norm(W, ord='fro'))
    print(np.linalg.norm(W, ord='nuc'))

    return W, results


def mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data):
    # obj = 0.5 * norm_sparse(Y - mask.multiply(A @ B.T), 'fro')**2 + 0.5 * param1 * (norm(A, 'fro')**2 + norm(B, 'fro')**2)

    A_prev = data['A']
    B_prev = data['B']

    if np.all(A_prev == A) and np.all(B_prev == B):
        loss_part = data['loss_part']
    else:
        if data['data_settings']['embedding'] == 'sideInfo' or data['data_settings']['embedding'] == 'multilabel_fscore':
            side_info = data['side_info']
            loss_part = loss_part_fun(side_info @ A, B, Y, mask, idx_mask)
        else:
            loss_part = loss_part_fun(A, B, Y, mask, idx_mask)
        data['loss_part'] = loss_part
        data['A'] = A
        data['B'] = B

        # A_bar = A[idx_mask[0], :].T
        # B_bar = B[idx_mask[1], :]
        # W_bar = A_bar * B_bar.T
        # w_bar_vec = np.sum(W_bar, axis=0)

    obj = 0.5 * norm_sparse(loss_part, 'fro') ** 2 + 0.5 * param1 * (norm(A, 'fro') ** 2 + norm(B, 'fro') ** 2)

    return obj


def mf_obj_wrapper(x, *args):

    # tt = time.time()


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
    elif embedding == 'multilabel_fscore':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        n_sideinfo = data_settings['n_sideinfo']
        side_info = data['side_info']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']

    temp_iter = data_settings['temp_iter']

    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
        A = np.reshape(x[:n_sideinfo*rank], (n_sideinfo, rank))
        B = np.reshape(x[n_sideinfo*rank:], (n_cols, rank))
    else:
        A = np.reshape(x[:n_rows*rank], (n_rows, rank))
        B = np.reshape(x[n_rows*rank:], (n_cols, rank))

    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
        obj = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)
    else:
        obj = mf_obj(A, B, Y, mask, y_tr_vec, idx_mask, param1, data)

    temp_iter = temp_iter + 1
    data_settings['temp_iter'] = temp_iter
    if (temp_iter % 1 == 0) or (temp_iter == 1):
        print('obj computations: %5d | obj: %14.6f' % (temp_iter, obj))
        logging.info('obj computations: %5d | obj: %14.6f', temp_iter, obj)

    return obj


def mf_grad(A, B, Y, mask, y_tr_vec, idx_mask, param1, embedding, side_info, data):

    A_prev = data['A']
    B_prev = data['B']

    if np.all(A_prev == A) and np.all(B_prev == B):
        print('same A and B in the grad')
        loss_part = data['loss_part']
    else:
        if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
            loss_part = loss_part_fun(side_info @ A, B, Y, mask, idx_mask)
        else:
            loss_part = loss_part_fun(A, B, Y, mask, idx_mask)
        data['loss_part'] = loss_part

        data['A'] = A
        data['B'] = B

    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
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
    elif embedding == 'multilabel_fscore':
        n_rows = data_settings['n_rows_embedded']
        n_cols = data_settings['n_cols_embedded']
        n_sideinfo = data_settings['n_sideinfo']
        side_info = data['side_info']
        mask = data['mask_tr_embedded']
        Y = data['Y_tr_embedded']


    if embedding == 'sideInfo' or embedding == 'multilabel_fscore':
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