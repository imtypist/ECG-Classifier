# -*- coding:utf8 -*-
from scipy.io import loadmat
import os
import sys
import datetime

import pywt
import seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import joblib

from biosppy.signals import ecg
import logging
import time
import math

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC

from sklearn import neighbors

# 项目目录
project_path = "./"
# 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model."

# 测试集在数据集中所占的比例
RATIO = 0.3

# 选择导联
DAOLIAN = [1,6,10] # II,V1,V5
# DAOLIAN = [i for i in range(12)]


label_id = {
    '窦性心律_左室肥厚伴劳损':0,
    '窦性心动过缓':1,
    '窦性心动过速':2,
    '窦性心律_不完全性右束支传导阻滞':3,
    '窦性心律_电轴左偏':4,
    '窦性心律_提示左室肥厚':5,
    '窦性心律_完全性右束支传导阻滞':6,
    '窦性心律_完全性左束支传导阻滞':7,
    '窦性心律_左前分支阻滞':8,
    '正常心电图':9
}

# 小波去噪预处理
def denoise(data, isPlot=False):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    if isPlot is True:
        plt.plot(data)
        plt.plot(rdata)
        plt.show()

    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(dir_name, number, X_data, Y_data):
    # 读取心电数据记录
    # print("正在读取 " + number + " 号心电数据...")
    record = loadmat("./ECGTrainData/Train/" + dir_name + "/" + number)['Beats'][0]
    for i in range(record['beatNumber'][0][0][0]):
        # 获取心电数据记录中R波的位置和对应的标签
        Rlocation = int(record['rPeak'][0][0][i]/2) # 降采样
        Rclass = record['label'][0][0]
        # 按label_id转换为0-9
        label = label_id[Rclass]

        if Rlocation < 100: # 过滤不符合条件的数据
            continue

        data = record['beatData'][0][0][0].T
        rdata = []
        # 降采样 1000hz -> 500hz
        for j in DAOLIAN:
            tdata = []
            for k in range(len(data[j])):
                if (k < len(data[j])-1) and (k % 2 == 0):
                    tdata.append((data[j][k]+data[j][k+1])/2)
                elif k == len(data[j])-1:
                    tdata.append(data[j][k])
            rdata.append(tdata)

        # X_data在R波前后截取长度为300的数据点
        rdata = np.array(rdata)
        x_train = rdata[:,Rlocation - 99:Rlocation + 201]
        if len(x_train[0]) < 300:
            continue

        # 300*len(DAOLIAN)
        x_train = x_train.flatten()

        X_data.append(x_train)
        Y_data.append(label)

    return


# 加载数据集并进行预处理
def loadData():
    trainDirSet = label_id.keys()

    dataSet = []
    labelSet = []
    for dir_name in trainDirSet:
        print("正在读取 " + dir_name + " 类别下的心电数据...")
        for root, dirs, files in os.walk("./ECGTrainData/Train/" + dir_name):
            for number in files:
                getDataSet(dir_name, number, dataSet, labelSet)


    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300*len(DAOLIAN))
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300*len(DAOLIAN)].reshape(-1, 300*len(DAOLIAN), 1)
    Y = train_ds[:, 300*len(DAOLIAN)]

    # 测试集及其标签集
    # shuffle_index = np.random.permutation(len(X))
    # test_length = int(RATIO * len(shuffle_index))
    # test_index = shuffle_index[:test_length]
    # train_index = shuffle_index[test_length:]
    # X_test, Y_test = X[test_index], Y[test_index]
    # X_train, Y_train = X[train_index], Y[train_index]
    # return X_train, Y_train, X_test, Y_test
    return X,Y


# 构建CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300*len(DAOLIAN), 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,10 个节点
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return newModel


def trainModel(method):
    if os.path.exists(model_path + method):
        print("There exists a " + method + " model file, please remove it before training a new one.")
    else:
        # X_train,Y_train为所有的数据集和标签集
        # X_test,Y_test为拆分的测试集和标签集
        X,Y = loadData()
        X_train = None
        X_test = None
        Y_train = None
        Y_test = None
        if method == "cnn":
            # define 10-fold cross validation test harness
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            max_acc = 0
            cvscores = []
            for train, test in kfold.split(X,Y):
                X_train = X[train]
                Y_train = Y[train]
                X_test = X[test]
                Y_test = Y[test]
                # 构建CNN模型
                model = buildModel()
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                # model.summary()
                # 定义TensorBoard对象
                # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                # 训练与验证
                model.fit(X_train, Y_train, epochs=30,
                          batch_size=64,
                          verbose=0,
                          # validation_split=RATIO,
                          # callbacks=[tensorboard_callback]
                          )
                # evaluate the model
                scores = model.evaluate(X_test, Y_test, verbose=0)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                cvscores.append(scores[1] * 100)
                if scores[1] > max_acc:
                    max_acc = scores[1]
                    model.save(filepath=model_path + method)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

            # 预测
            Y_pred = np.argmax(model.predict(X_test), axis=-1)
            # 绘制混淆矩阵
            plotHeatMap(Y_test, Y_pred)
            # # 用所有数据重新train一次model
            # model = buildModel()
            # model.compile(optimizer='adam',
            #               loss='sparse_categorical_crossentropy',
            #               metrics=['accuracy'])
            # model.summary()
            # model.fit(X, Y, epochs=30,
            #               batch_size=64,
            #               # validation_split=RATIO,
            #               # callbacks=[tensorboard_callback]
            #               )
            # model.save(filepath=model_path + method)
        elif method == "svm" or method == "knn":
            # 利用小波变换提取特征
            features = []
            X = X.reshape(-1,300*len(DAOLIAN))
            for data in X:
                cA5 = pywt.wavedec(data=data, wavelet='db6', level=5)[0]
                features.append(cA5)
                # print(len(cA5)) # 38

            X = np.array(features).reshape(-1,len(cA5))

            print(method + " training and testing...")
            # define 10-fold cross validation test harness
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            max_acc = 0
            cvscores = []
            for train, test in kfold.split(X,Y):
                X_train = X[train]
                Y_train = Y[train]
                X_test = X[test]
                Y_test = Y[test]

                # min_max_scaler = preprocessing.MinMaxScaler() 
                # X_train = min_max_scaler.fit_transform(X_train)
                # X_test = min_max_scaler.transform(X_test)

                #模型训练及预测
                # tic=time.time()
                if method == "svm":
                    model = SVC(kernel='rbf', C=2, gamma='scale')
                else:
                    model = neighbors.KNeighborsClassifier()
                model.fit(X_train,Y_train)
                # evaluate the model
                scores = model.score(X_test, Y_test)
                print("%s: %.2f%%" % ("acc", scores*100))
                cvscores.append(scores * 100)
                if scores > max_acc:
                    max_acc = scores
                    joblib.dump(model, model_path + method)
                # toc=time.time()
                # print("Elapsed time is %f sec."%(toc-tic))

            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
            Y_pred = model.predict(X_test)
            print(Y_pred)
            print(Y_test)
            plotHeatMap(Y_test, Y_pred)
    return


# 混淆矩阵
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred, labels=[i for i in range(10)])
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # 绘图
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_beats(data_path):
    data = loadmat("./ECGTestData/ECGTestData/" + data_path)['data'].T # II导
    data = data[1:,:] # 去除第一行index
    beats_matrix = []
    logging.info("--------------------------------------------------")
    logging.info("载入信号-%s, 长度 = %d " % (data_path, len(data[0])))
    min_beats = math.inf
    for i in DAOLIAN:
        signal = data[i]
        fs = 500  # 信号采样率 500 Hz
        # logging.info("调用 hamilton_segmenter 进行R波检测 ...")
        # tic = time.time()
        rpeaks = ecg.hamilton_segmenter(signal, sampling_rate=fs)
        # toc = time.time()
        # logging.info("完成. 用时: %f 秒. " % (toc - tic))
        rpeaks = rpeaks[0]

        # heart_rate = 60 / (np.mean(np.diff(rpeaks)) / fs)
        # np.diff 计算相邻R峰之间的距离分别有多少个采样点，np.mean求平均后，除以采样率，
        # 将单位转化为秒，然后计算60秒内可以有多少个RR间期作为心率
        # logging.info("平均心率: %.3f / 分钟." % (heart_rate))

        win_before = 0.2
        win_after = 0.4
        # logging.info("根据R波位置截取心拍, 心拍前窗口：%.3f 秒 ~ 心拍后窗口：%.3f 秒 ..." % (win_before, win_after)) # rpeak全部落在100处
        # tic = time.time()
        beats, rpeaks_beats = ecg.extract_heartbeats(signal, rpeaks, fs, win_before, win_after)
        # toc = time.time()
        # logging.info("完成. 用时: %f 秒." % (toc - tic))
        # logging.info("共截取到 %d 个心拍, 每个心拍长度为 %d 个采样点" % (beats.shape[0], beats.shape[1]))

        # plt.figure()
        # plt.grid(True)
        # for i in range(beats.shape[0]):
        #     plt.plot(beats[i])
        # plt.title(data_path)
        # plt.show()

        beats_matrix.append(beats)
        if len(beats) < min_beats:
            min_beats = len(beats)

    test_data = []
    for i in range(min_beats):
        tdata = []
        for j in range(len(DAOLIAN)):
            rbeats = denoise(beats_matrix[j][i])
            tdata.append(rbeats)
        test_data.append(tdata)

    test_data = np.array(test_data).flatten().reshape(-1, 300*len(DAOLIAN), 1)
    return test_data


def predict(test_data, model, method):
    if method == "cnn":
        pred = np.argmax(model.predict(test_data), axis=-1)
    elif method == "svm" or method == "knn":
        test_data = test_data.reshape(-1,300*len(DAOLIAN))
        features = []
        for data in test_data:
            cA5 = pywt.wavedec(data=data, wavelet='db6', level=5)[0]
            features.append(cA5)
            # print(len(cA5)) # 38

        test_data = np.array(features).reshape(-1,len(cA5))

        # min_max_scaler = preprocessing.MinMaxScaler()
        # test_data = min_max_scaler.fit_transform(test_data)
        pred = model.predict(test_data)

    majorCategory = {}
    for cate in pred:
        cate = int(cate)
        if cate not in majorCategory:
            majorCategory[cate] = 1
        else:
            majorCategory[cate] += 1
    majorCategory = sorted(majorCategory.items(),key = lambda x:x[1],reverse = True)
    print(majorCategory)
    return str(majorCategory[0][0])


def testData(method):
    f = open("pred_results_" + method + ".csv", "w+")
    f.write("id,categories\n")

    if method == "cnn":
        model = tf.keras.models.load_model(filepath=model_path + method)
    elif method == "svm" or method == "knn":
        model = joblib.load(model_path + method)

    for root, dirs, files in os.walk("./ECGTestData/ECGTestData/"):
        for data in files:
            test_data = extract_beats(data)
            category = predict(test_data, model, method)
            f.write(data.split(".")[0] + "," + category + "\n")

    f.close()
    return

if __name__ == '__main__':
    try:
        method = sys.argv[1]
    except Exception as e:
        print("You need to give a method to train ECG signal (options: cnn, svm, knn)")
        exit(0)

    if method == "cnn":
        trainModel("cnn")
        testData("cnn")
    elif method == "svm":
        trainModel("svm")
        testData("svm")
    elif method == "knn":
        trainModel("knn")
        testData("knn")
    else:
        print("This method is not supported, you can choose one of [cnn, svm, knn].")