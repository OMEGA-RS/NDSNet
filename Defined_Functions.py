import cv2
import os
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage import measure
from sklearn import metrics


# Initialize
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Create directory
def create_folder(path):
    if not os.path.exists(os.path.join(path, 'exp1')):
        os.makedirs(os.path.join(path, 'exp1'))
        return os.path.join(path, 'exp1')
    else:
        i = 2
        while os.path.exists(os.path.join(path, f'exp{i}')):
            i += 1
        new_folder = os.path.join(path, f'exp{i}')
        os.makedirs(new_folder)
        return new_folder


# Generating a difference image
def ND(im1, im2):
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    H, W = im1.shape

    # logarithmic transformation
    E = np.exp(0)
    im1_t = np.log(im1 + E)
    im2_t = np.log(im2 + E)

    # gaussian filter
    im1_g = gaussian_filter(im1_t, 0.5)
    im2_g = gaussian_filter(im2_t, 0.5)

    # neighborhood differencing
    patchsize = 3
    flag = patchsize // 2

    X1 = cv2.copyMakeBorder(im1_g, flag, flag, flag, flag, cv2.BORDER_DEFAULT)
    X2 = cv2.copyMakeBorder(im2_g, flag, flag, flag, flag, cv2.BORDER_DEFAULT)

    NFC_DI = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            im1_block = X1[i:i + patchsize, j:j + patchsize]
            im2_block = X2[i:i + patchsize, j:j + patchsize]

            v1 = im1_block.flatten()
            v2 = im2_block.flatten()

            NFC_DI[i, j] = np.sqrt(np.square(v1 - v2).sum(axis=0))

    im = (normalization(NFC_DI) * 255).astype(np.uint8)

    return im


def CalcuateC(U, I, V):
    c = len(V)
    for i in range(c):
        V[i] = np.sum(np.sum(U[:, :, i] ** 2 * I)) / np.sum(np.sum(U[:, :, i] ** 2))
    return V


def CalcuateG(U, V, flag, I):
    global d
    if flag == 3:
        d = np.array([[1 / (np.sqrt(2) + 1), 1 / 2, 1 / (np.sqrt(2) + 1)],
                      [1 / 2, 0, 1 / 2],
                      [1 / (np.sqrt(2) + 1), 1 / 2, 1 / (np.sqrt(2) + 1)]])
    elif flag == 5:
        d = np.array([[1 / (np.sqrt(8) + 1), 1 / (np.sqrt(5) + 1), 1 / 3, 1 / (np.sqrt(5) + 1), 1 / (np.sqrt(8) + 1)],
                      [1 / (np.sqrt(5) + 1), 1 / (np.sqrt(2) + 1), 1 / 2, 1 / (np.sqrt(2) + 1), 1 / (np.sqrt(5) + 1)],
                      [1 / 3, 1 / 2, 0, 1 / 2, 1 / 3],
                      [1 / (np.sqrt(5) + 1), 1 / (np.sqrt(2) + 1), 1 / 2, 1 / (np.sqrt(2) + 1), 1 / (np.sqrt(5) + 1)],
                      [1 / (np.sqrt(8) + 1), 1 / (np.sqrt(5) + 1), 1 / 3, 1 / (np.sqrt(5) + 1), 1 / (np.sqrt(8) + 1)]])
    c = len(V)
    G = np.ones_like(U)
    for i in range(c):
        temp = (1 - U[:, :, i]) ** 2 * ((I - V[i]) ** 2)
        G[:, :, i] = convolve(temp, d, mode='constant')
    return G


def CalcuateU(V, G, I):
    U = np.ones_like(G)
    c = len(V)
    all_ = np.ones_like(G)
    for i in range(c):
        all_[:, :, i] = (I - V[i]) ** 2 + G[:, :, i]
    all_[all_ == 0] = 0.01

    for i in range(c):
        temp = 0
        for j in range(c):
            temp += all_[:, :, i] / all_[:, :, j]
        temp[temp == 0] = 0.01
        U[:, :, i] = 1 / temp

    temp = np.sum(U, axis=2)
    temp = np.repeat(temp[:, :, np.newaxis], U.shape[2], axis=2)
    U = U / temp
    return U


def defuzzy(U, V):
    U = np.array(U)
    V = np.array(V)
    I = np.ones((U.shape[0], U.shape[1]))
    c = len(V)
    Vlabel = np.ones_like(V)

    if c == 2:
        Vlabel[0] = 0
        Vlabel[1] = 255
    elif c == 3:
        Vlabel[0] = 0
        Vlabel[1] = 128
        Vlabel[2] = 255

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            index = np.argmax(U[i, j])
            I[i, j] = Vlabel[index]

    return I


def rerangeuc(U, V):
    # c = len(V)
    index = np.argsort(V)
    for i in range(len(U)):
        for j in range(len(U[i])):
            U[i][j] = U[i][j][index]
    V = V[index]
    return U, V


def Initialization(I, c):
    I = np.uint8(I)
    counts, x = np.histogram(I, bins=range(257))
    I = I.astype(float)
    m, n = I.shape
    Ufcm = [[np.zeros(c) for _ in range(n)] for _ in range(m)]
    e = 0.001
    V1 = np.zeros(c)  # V1是旧的聚类中心
    V2 = np.zeros(c)  # V2是新的聚类中心
    U1 = np.zeros((c, 256))
    m1 = 2  # 确定加权指数m
    num = 0  # 初始迭代次数
    V1[0] = 50  # 初始化聚类中心V
    V1[1] = 150

    flag = 1
    while flag == 1 and num < 500:
        for j in range(256):
            for i in range(c):
                s = 0
                if x[j] - V1[i] == 0:
                    U1[i, j] = 1
                    U1[:i, j] = 0
                    U1[i + 1:, j] = 0
                    break
                else:
                    for k in range(c):
                        s += (x[j] - V1[k]) ** (2 / (m1 - 1))
                    U1[i, j] = s / (x[j] - V1[i]) ** (2 / (m1 - 1))

        for j in range(256):
            temp = U1[:, j].sum()
            U1[:, j] /= temp

        for i in range(c):
            sum_ = 0
            sum1 = 0
            for j in range(256):
                sum_ += counts[j] * U1[i, j] ** m1
                sum1 += x[j] * counts[j] * U1[i, j] ** m1
            V2[i] = sum1 / sum_

        num += 1
        if np.max(np.abs(V2 - V1)) < e:
            flag = 0
        else:
            V1 = V2

    for i in range(m):
        for j in range(n):
            for k in range(256):
                if x[k] == I[i, j]:
                    Ufcm[i][j] = U1[:, k]

    return Ufcm, V2


def FLICM(I, Cluster_n):
    I = np.uint8(I)
    m, n = I.shape
    c = Cluster_n
    flag = 1
    Ufcm, Vfcm = Initialization(I, c)  # FCM
    Ufcm, Vfcm = rerangeuc(Ufcm, Vfcm)
    I = I.astype(float)
    U = np.ones((m, n, c))
    for i in range(m):
        for j in range(n):
            for k in range(c):
                U[i, j, k] = Ufcm[i][j][k]
    V = Vfcm
    if flag == 1:
        flag = 3
    elif flag == 2:
        flag = 5
    else:
        raise ValueError('Here is an error!')

    for _ in range(100):
        V = CalcuateC(U, I, V)
        G = CalcuateG(U, V, flag, I)
        U = CalcuateU(V, G, I)

    for i in range(m):
        for j in range(n):
            for k in range(c):
                Ufcm[i][j][k] = U[i, j, k]

    Vfcm = V
    Ufcm, Vfcm = rerangeuc(Ufcm, Vfcm)

    Iindex = np.uint8(defuzzy(Ufcm, Vfcm))

    return Iindex

def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new


def calculate_metric(ref_map, change_map, name):
    ref_label, map_label = ref_map.flatten(), change_map.flatten()
    confusion_matrix = metrics.confusion_matrix(ref_label, map_label)

    tmp_FP = confusion_matrix[0, 1]
    tmp_FN = confusion_matrix[1, 0]
    tmp_OE = len(ref_label) - np.trace(confusion_matrix)
    tmp_OA = np.trace(confusion_matrix) / len(ref_label)
    tmp_KC = metrics.cohen_kappa_score(ref_label, map_label)

    message = ""
    message += f"{name} results ==>\n"

    message += "FP is : {:d}\n".format(tmp_FP)
    message += "FN is : {:d}\n".format(tmp_FN)
    message += "OE is : {:d}\n".format(tmp_OE)
    message += "OA is : {:f}\n".format(tmp_OA)
    message += "KC is : {:f}\n".format(tmp_KC)

    print(message)

    return message
