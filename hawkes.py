

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from scipy.optimize import minimize
from scipy.optimize import check_grad
import scipy
import time
from queue import PriorityQueue
import metrics
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.cluster import SpectralClustering


class TSHawkes:
    """
    para:
    b:array
    alpha:matrix
    q:array of array
    p:array of array
    beta:matrix of array

    r:dim of p,q
    delta:kernel's para

    member:
    events:numpy 2darray of (u,i,t)
    T:end time
    numUser:user's number
    numItem:item's number
    numRecords:record's number

    Intermediate variable：
    E_u_i:all (u,i) sum arrays
    U_I_last:last (u,i) item index in events
    u_i:set of all (u,i) pair

    E_u:all item sum arrays
    U_last:last i item index in events

    Ru:
    u_cluster:user in which cluster
    cluster_u:cluser with its users
    R_u:user's neighborhood

    events:pandas DF of (u,i,t)
    T:end time
    """

    def __init__(self, events, T, delta1, delta2, r, penalty_bu, penalty_bi, penalty_p, penalty_q, penalty_beta,
                 u_cluster, cluster_num, item_cluster, test_datasets, if_print=False, validate=False):
        self.last_time = time.time()
        self.events = events
        self.numRecords = self.events.shape[0]
        self.numUser = max(self.events[:, 0]) + 1
        self.numItem = max(self.events[:, 1]) + 1
        self.T = T
        self.delta1 = delta1
        self.delta2 = delta2
        self.r = r
        self.if_print = if_print
        self.validate = validate
        self.penalty_bi = penalty_bi
        self.penalty_bu = penalty_bu
        self.penalty_p = penalty_p
        self.penalty_q = penalty_q
        self.penalty_beta = penalty_beta

        self.bu = np.random.uniform(0, 1, size=(self.numUser,))
        self.bi = np.random.uniform(0, 1, size=(self.numItem,))
        self.q = np.random.uniform(0, 1, size=(self.numItem, self.r))
        self.p = np.random.uniform(0, 1, size=(self.numUser, self.r))
        self.beta = np.random.uniform(0, 1, size=(cluster_num.shape[0], max(cluster_num)))

        self.bu_zero = np.full((self.numUser,), 2.0)
        self.bi_zero = np.full((self.numItem,), 2.0)
        self.q_zero = np.full((self.numItem, self.r), 2.0)
        self.p_zero = np.full((self.numUser, self.r), 2.0)
        self.beta_zero = np.full((cluster_num.shape[0], max(cluster_num)), 2.0)

        self.u_cluster = u_cluster
        self.cluster_num = cluster_num
        self.item_cluster = item_cluster
        self.test_datasets = test_datasets

        self.list_size = [5, 10] + [int(self.numItem * ratio) for ratio in [0.001, 0.0025, 0.005, 0.01]]    # 推荐长度列表

        if self.if_print:
            print('__init__ ', time.time() - self.last_time, 's')
            self.last_time = time.time()

        self.calculateE()

    def kernel_g(self, t, delta):
        """
        核
        :param t:
        :param delta:
        :return:
        """
        return np.exp(-delta * t)

    def integral_g(self, superscript, delta):
        """
        核
        :param superscript:
        :param delta:
        :return:
        """
        return (1 - np.exp(-delta * superscript)) / delta

    def update_calculate(self, i, np_i, t, E, last, front, E_last, delta):
        if front[np_i]:
            E_last[i] = (E_last[last[np_i]] + 1) * self.kernel_g(t - self.events[last[np_i]][2], delta)
            if t > self.events[last[np_i]][2]:
                E[i] = E_last[i]
            else:
                E[i] = E[last[np_i]]
        else:
            front[np_i] = True
        last[np_i] = i

    def init_term_beta(self):
        """
         这几个矩阵存的是intensity公式里的最后一项对应的值，这里进行初始化，更新在get_ith_intensity里，值是随时间变化的
        """
        self.term_beta_last_time = np.zeros((self.numItem, max(self.cluster_num)))

        self.term_beta_p = np.zeros((self.numItem, max(self.cluster_num), self.r))
        self.term_beta_last_p = np.zeros((self.numItem, max(self.cluster_num), self.r))

        self.term_beta_q = np.zeros((self.numItem, max(self.cluster_num), self.r))
        self.term_beta_last_q = np.zeros((self.numItem, max(self.cluster_num), self.r))

    def calculateE(self):
        if self.if_print:
            print('calculateE ', end="")

        self.init_term_beta()

        self.E_u_i_1 = np.zeros(self.numRecords)
        self.U_I_last_1 = np.zeros((self.numUser, self.numItem), dtype=np.int)
        self.U_I_front_1 = np.zeros((self.numUser, self.numItem), dtype=np.bool)
        self.E_u_i_last_1 = np.zeros(self.numRecords)

        self.E_u_i_2 = np.zeros(self.numRecords)
        self.U_I_last_2 = np.zeros((self.numUser, self.numItem), dtype=np.int)
        self.U_I_front_2 = np.zeros((self.numUser, self.numItem), dtype=np.bool)
        self.E_u_i_last_2 = np.zeros(self.numRecords)

        self.sum_integral_g_u_i_2 = np.zeros((self.numUser, self.numItem))

        self.user_items = set()

        for i, event in enumerate(self.events):
            self.update_calculate(i, (event[0], event[1]), event[2], self.E_u_i_1, self.U_I_last_1, self.U_I_front_1,
                                  self.E_u_i_last_1, self.delta1)
            self.update_calculate(i, (event[0], event[1]), event[2], self.E_u_i_2, self.U_I_last_2, self.U_I_front_2,
                                  self.E_u_i_last_2, self.delta2)

            self.sum_integral_g_u_i_2[event[0], event[1]] += self.integral_g(self.T - event[2], self.delta2)

            self.user_items.add((event[0], event[1]))

        if self.if_print:
            print(time.time() - self.last_time, 's')
            self.last_time = time.time()

    def getIntensity(self, u, i, t, disp=False):
        """
        for debug use
        :param u:
        :param i:
        :param t:
        :param disp:
        :return:
        """

        sum_E_u_i = 0
        sum_E_beta_u_ = np.zeros(self.r)

        if i >= self.numItem or u >= self.numUser:
            return 0

        video_type = self.item_cluster[i]
        user_video_cluster = self.u_cluster[u][video_type]
        cur_beta = self.beta[video_type, user_video_cluster]

        if self.U_I_front_1[u, i]:
            sum_E_u_i = (self.E_u_i_last_1[self.U_I_last_1[u, i]] + 1) * self.kernel_g(
                t - self.events[self.U_I_last_1[u, i]][2], self.delta1)

        sum_E_beta_u_ = self.term_beta_last_p[i, user_video_cluster] * self.kernel_g(
            t - self.term_beta_last_time[i, user_video_cluster], self.delta2)

        if disp:
            print(self.bu[u], '+', self.bi[i], ' + np.dot( ', self.q[i], ' , ', self.p[u], ' )* ', sum_E_u_i, ' + ',
                  cur_beta, ' * np.dot( ', self.q[i], ' , ', sum_E_beta_u_, ' )')
        return self.bu[u] + self.bi[i] + (1 - cur_beta) * np.dot(self.q[i], self.p[u]) * sum_E_u_i + cur_beta * np.dot(
            self.q[i], sum_E_beta_u_)

    def user_rank_items(self, k, t):
        """
        取每个用户最大的前k个intensity
        :param k:
        :param t:
        :return:
        """
        users_test = set(self.test_datasets[0].unique())
        bi_ranks = np.flip(self.bi.argsort()[-k:], 0)
        users_train = set(self.events.T[0])

        def rank_each_user(u_array):
            u = u_array[0]
            ranks = np.full(k, -1, dtype=np.int)
            if u not in users_test:
                return ranks
            if u not in users_train:
                return bi_ranks
            min_heap = PriorityQueue()
            for i in range(self.numItem):
                video_type = self.item_cluster[i]
                user_video_cluster = self.u_cluster[u][video_type]
                cur_beta = self.beta[video_type, user_video_cluster]
                sum_E_beta_u_ = self.term_beta_last_p[i, user_video_cluster] * self.kernel_g(
                    t - self.term_beta_last_time[i, user_video_cluster], self.delta2)
                sum_E_u_i = 0
                if self.U_I_front_1[u, i]:
                    sum_E_u_i = (self.E_u_i_last_1[self.U_I_last_1[u, i]] + 1) * self.kernel_g(
                        t - self.events[self.U_I_last_1[u, i]][2], self.delta1)
                intensity = self.bu[u] + self.bi[i] + (1 - cur_beta) * np.dot(self.q[i], self.p[
                    u]) * sum_E_u_i + cur_beta * np.dot(self.q[i], sum_E_beta_u_)
                min_heap.put((intensity, i))
                if min_heap.qsize() > k:
                    min_heap.get()
            for i in range(k - 1, -1, -1):
                if min_heap.empty():
                    break
                intensity_i = min_heap.get()
                ranks[i] = intensity_i[1]
            return ranks

        return np.apply_along_axis(rank_each_user, 1, np.arange(self.numUser).reshape((self.numUser, 1)))

    def getIntensity_ith_event(self, i, disp=False):            # 优化过程中计算intensity
        event = self.events[i]

        video_type = self.item_cluster[event[1]]
        user_video_cluster = self.u_cluster[event[0]][video_type]
        cur_beta = self.beta[video_type, user_video_cluster]

        kernel_plus = self.kernel_g(event[2] - self.term_beta_last_time[event[1], user_video_cluster], self.delta2)
        self.term_beta_last_p[event[1], user_video_cluster] *= kernel_plus
        self.term_beta_last_q[event[1], user_video_cluster] *= kernel_plus
        if self.term_beta_last_time[event[1], user_video_cluster] < event[2]:
            self.term_beta_p[event[1], user_video_cluster] = self.term_beta_last_p[event[1], user_video_cluster]
            self.term_beta_q[event[1], user_video_cluster] = self.term_beta_last_q[event[1], user_video_cluster]

        self.term_beta_last_p[event[1], user_video_cluster] += self.p[event[0]]
        self.term_beta_last_q[event[1], user_video_cluster] += cur_beta * self.q[event[1]]
        self.term_beta_last_time[event[1], user_video_cluster] = event[2]

        intensity = self.bu[event[0]] + self.bi[event[1]] + (1 - cur_beta) * np.dot(self.q[event[1]], self.p[
            event[0]]) * self.getSum_E_u_i_1(i) + cur_beta * np.dot(self.q[event[1]],
                                                                    self.term_beta_p[event[1], user_video_cluster])

        if disp:
            print(self.bu[event[0]], '+', self.bi[event[1]], ' + np.dot( ', self.q[event[1]], ' , ', self.p[event[0]],
                  ' )* ', self.getSum_E_u_i(i), ' + ',
                  cur_beta * np.dot(self.q[event[i]], self.term_beta_p[event[1], user_video_cluster]))

        return intensity

    def x_to_para(self, x):
        begin = 0

        end = begin + np.cumprod(self.bu.shape)[-1]
        self.bu = x[begin: end].reshape(self.bu.shape)
        self.bu_zero[self.bu <= 0] = self.bu_zero[self.bu <= 0] / 2
        self.bu[self.bu <= 0] = self.bu_zero[self.bu <= 0]
        begin = end

        end = begin + np.cumprod(self.bi.shape)[-1]
        self.bi = x[begin: end].reshape(self.bi.shape)
        self.bi_zero[self.bi <= 0] = self.bi_zero[self.bi <= 0] / 2
        self.bi[self.bi <= 0] = self.bi_zero[self.bi <= 0]
        begin = end

        end = begin + np.cumprod(self.q.shape)[-1]
        self.q = x[begin: end].reshape(self.q.shape)
        self.q_zero[self.q <= 0] = self.q_zero[self.q <= 0] / 2
        self.q[self.q <= 0] = self.q_zero[self.q <= 0]
        begin = end

        end = begin + np.cumprod(self.p.shape)[-1]
        self.p = x[begin: end].reshape(self.p.shape)
        self.p_zero[self.p <= 0] = self.p_zero[self.p <= 0] / 2
        self.p[self.p <= 0] = self.p_zero[self.p <= 0]
        begin = end

        end = begin + np.cumprod(self.beta.shape)[-1]
        self.beta = x[begin: end].reshape(self.beta.shape)
        self.beta_zero[self.beta <= 0] = self.beta_zero[self.beta <= 0] / 2
        self.beta[self.beta <= 0] = self.beta_zero[self.beta <= 0]
        begin = end

    def cluster_Ru(self):
        if self.if_print:
            print('cluster_Ru', end="")

        self.cluster_u = [[] for i in range(self.cluster_num.shape[0])]
        for i, u_cluster_i in enumerate(self.u_cluster.T):
            self.cluster_u[i] = [[] for j in range(self.cluster_num[i])]
            for u, c in enumerate(u_cluster_i):
                self.cluster_u[i][c].append(u)
            print(end='cluster size : ')
            for c in range(self.cluster_num[i]):
                print(c, len(self.cluster_u[i][c]), end=' | ')
                self.cluster_u[i][c] = np.array(self.cluster_u[i][c])
            print()

        if self.if_print:
            print(time.time() - self.last_time, 's')
            self.last_time = time.time()

    def likelihood(self, x, disp=False):    # 计算目标函数
        if self.if_print:
            print('likelihood ', end="")

        self.x_to_para(x)

        intensities = 0
        sum_q_p = 0

        self.init_term_beta()
        for i, event in enumerate(self.events):
            intensities += math.log(self.getIntensity_ith_event(i))
            video_type = self.item_cluster[event[1]]
            cur_beta = self.beta[video_type, self.u_cluster[event[0]][video_type]]
            sum_q_p += np.dot(self.q[event[1]], self.p[event[0]]) * self.integral_g(self.T - event[2], self.delta1) * (
                        1 - cur_beta)

        term_tail = 0
        self.sum_R_u_2 = np.zeros(self.beta.shape)

        for u, i in self.user_items:
            video_type = self.item_cluster[i]
            user_cluster = self.u_cluster[u][video_type]
            self.sum_R_u_2[video_type, user_cluster] += np.dot(self.q[i], self.p[u] * self.sum_integral_g_u_i_2[u, i])

        for i in range(self.cluster_num.shape[0]):
            for j in range(self.cluster_num[i]):
                term_tail += self.cluster_u[i][j].shape[0] * self.beta[i][j] * self.sum_R_u_2[i][j]

        sumB = self.T * (np.sum(self.bi) * self.numUser + np.sum(self.bu) * self.numItem)

        if self.if_print:
            print(time.time() - self.last_time, 's')
            self.last_time = time.time()

        tmp_obj = -intensities + sumB + sum_q_p + term_tail     ### intensity 不是已经计算过了，那么这个tmp_obj是做什么的
        if disp:
            print(-intensities, '+', sumB, '+', sum_q_p, '+', term_tail, \
                  ' + ', self.penalty_bu, '*', np.sum(self.bu * self.bu), \
                  ' + ', self.penalty_bi, '*', np.sum(self.bi * self.bi), \
                  ' + ', self.penalty_p, '*', np.sum(self.p * self.p), \
                  ' + ', self.penalty_q, '*', np.sum(self.q * self.q), \
                  ' + ', self.penalty_beta, '*', np.sum(self.beta * self.beta)
                  )
            print(tmp_obj)
        self.obj_pre.append(tmp_obj)
        return tmp_obj \
               + self.penalty_bi * np.sum(self.bi * self.bi) + self.penalty_bu * np.sum(self.bu * self.bu) \
               + self.penalty_p * np.sum(self.p * self.p) + self.penalty_q * np.sum(
            self.q * self.q) + self.penalty_beta * np.sum(self.beta * self.beta)

    def likelihoodGradient(self, x):
        if self.if_print:
            print('likelihoodGradient ', end="")

        bu_grad = np.zeros(self.bu.shape)
        bi_grad = np.zeros(self.bi.shape)
        p_grad = np.zeros(self.p.shape)
        q_grad = np.zeros(self.q.shape)
        beta_grad = np.zeros(self.beta.shape)

        R_p_grad = np.zeros(self.beta.shape + (self.r,))
        R_q_grad = np.zeros(self.beta.shape + (self.r,))

        self.init_term_beta()
        for i, event in enumerate(self.events):
            intensity = self.getIntensity_ith_event(i)
            intensity_ = 1 / intensity

            video_type = self.item_cluster[event[1]]
            user_cluster = self.u_cluster[event[0]][video_type]
            cur_beta = self.beta[video_type, user_cluster]

            bu_grad[event[0]] += intensity_
            bi_grad[event[1]] += intensity_
            p_grad[event[0]] += self.q[event[1]] * (
                        intensity_ * self.getSum_E_u_i_1(i) - self.integral_g(self.T - event[2], self.delta1)) * (
                                            1 - cur_beta)

            R_p_grad[video_type, user_cluster] += self.term_beta_q[event[1], user_cluster] * intensity_
            R_q_grad[video_type, user_cluster] += cur_beta * self.term_beta_p[event[1], user_cluster] * intensity_

            q_grad[event[1]] += self.p[event[0]] * self.getSum_E_u_i_1(i) * (1 - cur_beta) * intensity_ - self.p[
                event[0]] * self.integral_g(self.T - event[2], self.delta1) * (1 - cur_beta)
            beta_grad[video_type, user_cluster] += (-np.dot(self.q[event[1]], self.p[event[0]]) * self.getSum_E_u_i_1(
                i) + np.dot(self.q[event[1]], self.term_beta_p[event[1], user_cluster])) * intensity_ \
                                                   + np.dot(self.q[event[1]], self.p[event[0]]) * self.integral_g(
                self.T - event[2], self.delta1)

        sumQ = sum(self.q)
        bu_grad -= self.T * self.numItem
        bi_grad -= self.T * self.numUser

        sum_R_u_p_grad = np.zeros(self.beta.shape + (3,))
        sum_R_u_q_grad = np.zeros(self.beta.shape + (3,))
        for u, i in self.user_items:
            video_type = self.item_cluster[i]
            user_cluster = self.u_cluster[u][video_type]
            sum_R_u_p_grad[video_type, user_cluster] += self.q[i] * self.sum_integral_g_u_i_2[u, i]
            sum_R_u_q_grad[video_type, user_cluster] += self.p[u] * self.sum_integral_g_u_i_2[u, i]

        for user, item in self.user_items:
            i = self.item_cluster[item]
            j = self.u_cluster[user][video_type]
            p_grad[user] += R_p_grad[i][j]
            q_grad[user] += R_q_grad[i][j]
            p_grad[user] -= self.cluster_u[i][j].shape[0] * self.beta[i][j] * sum_R_u_p_grad[i][j]
            q_grad[item] -= self.cluster_u[i][j].shape[0] * self.beta[i][j] * sum_R_u_q_grad[i][j]

        for i in range(self.cluster_num.shape[0]):
            for j in range(self.cluster_num[i]):
                R_u_num = self.cluster_u[i][j].shape[0]
                beta_grad[i][j] -= R_u_num * self.sum_R_u_2[i][j]

        if self.if_print:
            print(time.time() - self.last_time, 's')
            self.last_time = time.time()

        bu_grad -= self.penalty_bu * 2 * self.bu
        bi_grad -= self.penalty_bi * 2 * self.bi
        p_grad -= self.penalty_p * 2 * self.p
        q_grad -= self.penalty_q * 2 * self.q
        beta_grad -= self.penalty_beta * 2 * self.beta

        print(-np.concatenate([bu_grad.ravel(), bi_grad.ravel(), q_grad.ravel(), p_grad.ravel(), beta_grad.ravel()]))

        return -np.concatenate([bu_grad.ravel(), bi_grad.ravel(), q_grad.ravel(), p_grad.ravel(), beta_grad.ravel()])

    def validate_test(self, disp=True):
        re = self.user_rank_items(max(self.list_size), self.T)
        for idx, i in enumerate(self.list_size):
            ranks = re[:, 0:i]
            hit_r = metrics.hit_rate(ranks, self.test_datasets)
            self.hitRate[self.T][idx].append(hit_r)
            if disp:
                print("hit citutation: ", end="")
                print(hit_r, end=",")
        if disp:
            print()
        return self.hitRate[self.T]

    def likelihood_(self, disp=False):
        return self.likelihood(
            np.concatenate([self.bu.ravel(), self.bi.ravel(), self.q.ravel(), self.p.ravel(), self.beta.ravel()]), disp)

    def getSum_E_u_i_1(self, i):
        return self.E_u_i_1[i]

    def fit_with_BFGS(self):
        self.hitRate = [[[] for j in range(len(self.list_size))] for i in range(self.T + 1)]
        self.sum_hitrate = []
        self.obj = []
        self.obj_pre = []

        self.cluster_Ru()

        if self.if_print:
            print('fit init ', end="")

        x0 = np.concatenate([self.bu.ravel(), self.bi.ravel(), self.q.ravel(), self.p.ravel(), self.beta.ravel()])
        min_bnd = np.cumprod(self.bu.shape)[-1]
        beta_bnd = np.cumprod(self.beta.shape)[-1]
        bnds = [(0, None) for i in range(min_bnd)] + [(0, None) for i in range(min_bnd, len(x0) - beta_bnd)] + [(0, 1) for i in range(len(x0) - beta_bnd,len(x0))]    ### 这是做什么？把三个部分的（0，1）拼接起来

        if self.if_print:
            print(time.time() - self.last_time, 's')
            self.last_time = time.time()

        iteration = [0]

        def call_back(x):
            obj_ = self.likelihood(x, True)
            self.obj.append(obj_)
            print('iter ', iteration, ' likelihood ', obj_, ' time ', time.time() - self.last_time, 's')
            self.last_time = time.time()
            iteration[0] = iteration[0] + 1
            self.backup_x = x
            if self.validate:
                self.validate_test()

        print('iter  [-1]  likelihood  ', self.likelihood(x0))
        self.validate_test()
        res = minimize(self.likelihood,     # 目标函数
                       x0,
                       method='L-BFGS-B',
                       jac=self.likelihoodGradient,     # 调用梯度函数？
                       bounds=bnds,
                       options={'disp': False, 'maxcor': 10, 'ftol': 1e-9, 'gtol': 1e-5, 'maxfun': 15000, 'maxiter': 100,
                                'maxls': 20},
                       callback=call_back,       # 回调函数
                       )
        print('likelihood end with iteration', iteration[0] - 1, '. likelihood', self.likelihood(res.x))
        self.x_to_para(res.x)


