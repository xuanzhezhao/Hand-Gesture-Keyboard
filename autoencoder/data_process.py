import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re
import random
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def load_data():
    df_A = pd.read_table('sign/a.txt', header=None, sep=',')
    A = np.array(df_A)
    df_B = pd.read_table('sign/b.txt', header=None, sep=',')
    B = np.array(df_B)
    df_C = pd.read_table('sign/c.txt', header=None, sep=',')
    C = np.array(df_C)
    df_D = pd.read_table('sign/d.txt', header=None, sep=',')
    D = np.array(df_D)
    df_E = pd.read_table('sign/e.txt', header=None, sep=',')
    E = np.array(df_E)
    df_F = pd.read_table('sign/f.txt', header=None, sep=',')
    F = np.array(df_F)
    df_G = pd.read_table('sign/g.txt', header=None, sep=',')
    G = np.array(df_G)
    df_H = pd.read_table('sign/h.txt', header=None, sep=',')
    H = np.array(df_H)
    df_I = pd.read_table('sign/i.txt', header=None, sep=',')
    I = np.array(df_I)
    df_J = pd.read_table('sign/j.txt', header=None, sep=',')
    J = np.array(df_J)
    df_K = pd.read_table('sign/k.txt', header=None, sep=',')
    K = np.array(df_K)
    df_L = pd.read_table('sign/l.txt', header=None, sep=',')
    L = np.array(df_L)
    df_M = pd.read_table('sign/m.txt', header=None, sep=',')
    M = np.array(df_M)
    df_N = pd.read_table('sign/n.txt', header=None, sep=',')
    N = np.array(df_N)
    df_O = pd.read_table('sign/o.txt', header=None, sep=',')
    O = np.array(df_O)
    df_P = pd.read_table('sign/p.txt', header=None, sep=',')
    P = np.array(df_P)
    df_Q = pd.read_table('sign/q.txt', header=None, sep=',')
    Q = np.array(df_Q)
    df_R = pd.read_table('sign/r.txt', header=None, sep=',')
    R = np.array(df_R)
    df_S = pd.read_table('sign/s.txt', header=None, sep=',')
    S = np.array(df_S)
    df_T = pd.read_table('sign/t.txt', header=None, sep=',')
    T = np.array(df_T)
    df_U = pd.read_table('sign/u.txt', header=None, sep=',')
    U = np.array(df_U)
    df_V = pd.read_table('sign/v.txt', header=None, sep=',')
    V = np.array(df_V)
    df_W = pd.read_table('sign/w.txt', header=None, sep=',')
    W = np.array(df_W)
    df_X = pd.read_table('sign/x.txt', header=None, sep=',')
    X = np.array(df_X)
    df_Y = pd.read_table('sign/y.txt', header=None, sep=',')
    Y = np.array(df_Y)
    df_Z = pd.read_table('sign/z.txt', header=None, sep=',')
    Z = np.array(df_Z)

    a = len(A)
    b = len(B)
    c = len(C)
    d = len(D)
    e = len(E)
    f = len(F)
    g = len(G)
    h = len(H)
    i = len(I)
    j = len(J)
    k = len(K)
    l = len(L)
    m = len(M)
    n = len(N)
    o = len(O)
    p = len(P)
    q = len(Q)
    r = len(R)
    s = len(S)
    t = len(T)
    u = len(U)
    v = len(V)
    w = len(W)
    x = len(X)
    y = len(Y)
    z = len(Z)






    df = df_A.append(df_B)
    df = df.append(df_C)
    df = df.append(df_D)
    df = df.append(df_E)
    df = df.append(df_F)
    df = df.append(df_G)
    df = df.append(df_H)
    df = df.append(df_I)
    df = df.append(df_J)
    df = df.append(df_K)
    df = df.append(df_L)
    df = df.append(df_M)
    df = df.append(df_N)
    df = df.append(df_O)
    df = df.append(df_P)
    df = df.append(df_Q)
    df = df.append(df_R)
    df = df.append(df_S)
    df = df.append(df_T)
    df = df.append(df_U)
    df = df.append(df_V)
    df = df.append(df_W)
    df = df.append(df_X)
    df = df.append(df_Y)
    df = df.append(df_Z)


    df = df.drop(df.columns[-1], axis=1)
    data = pd.DataFrame(df).to_numpy()
    print(type(data))
    print(data.shape)
    class_a = [0 for i in range(len(A))]
    class_b = [1 for i in range(len(B))]
    class_c = [2 for i in range(len(C))]
    class_d = [3 for i in range(len(D))]
    class_e = [4 for i in range(len(E))]
    class_f = [5 for i in range(len(F))]
    class_g = [6 for i in range(len(G))]
    class_h = [7 for i in range(len(H))]
    class_i = [8 for i in range(len(I))]
    class_j = [9 for i in range(len(J))]
    class_k = [10 for i in range(len(K))]
    class_l = [11 for i in range(len(L))]
    class_m = [12 for i in range(len(M))]
    class_n = [13 for i in range(len(N))]
    class_o = [14 for i in range(len(O))]
    class_p = [15 for i in range(len(P))]
    class_q = [16 for i in range(len(Q))]
    class_r = [17 for i in range(len(R))]
    class_s = [18 for i in range(len(S))]
    class_t = [19 for i in range(len(T))]
    class_u = [20 for i in range(len(U))]
    class_v = [21 for i in range(len(V))]
    class_w = [22 for i in range(len(W))]
    class_x = [23 for i in range(len(X))]
    class_y = [24 for i in range(len(Y))]
    class_z = [25 for i in range(len(Z))]

    y_label = np.append(class_a, class_b)
    y_label = np.append(y_label, class_c)
    y_label = np.append(y_label, class_d)
    y_label = np.append(y_label, class_e)
    y_label = np.append(y_label, class_f)
    y_label = np.append(y_label, class_g)
    y_label = np.append(y_label, class_h)
    y_label = np.append(y_label, class_i)
    y_label = np.append(y_label, class_j)
    y_label = np.append(y_label, class_k)
    y_label = np.append(y_label, class_l)
    y_label = np.append(y_label, class_m)
    y_label = np.append(y_label, class_n)
    y_label = np.append(y_label, class_o)
    y_label = np.append(y_label, class_p)
    y_label = np.append(y_label, class_q)
    y_label = np.append(y_label, class_r)
    y_label = np.append(y_label, class_s)
    y_label = np.append(y_label, class_t)
    y_label = np.append(y_label, class_u)
    y_label = np.append(y_label, class_v)
    y_label = np.append(y_label, class_w)
    y_label = np.append(y_label, class_x)
    y_label = np.append(y_label, class_y)
    y_label = np.append(y_label, class_z)
    y_label = y_label.reshape(3927, 1)
    print(type(y_label))
    print(y_label.shape)
    return data, y_label, a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z

def normalize_data(data):
    normalized_data=data
    max=np.max(abs(data),axis=0)
    for i in range (0,len(data)):
        normalized_data[i,:]=normalized_data[i,:]/max
    return normalized_data

"""
if __name__=="__main__":
    data, label=load_data()
    norm_data=normalize_data(data)
    print(data.shape)
"""