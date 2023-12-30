import numpy as np
import pandas as pd
import sklearn
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model


# 这个文件用于数据集的读取及处理

# 读取文件，返回dataframe
def read_file(filename):
    # 数据导入
    df = pd.read_csv(filename, encoding="UTF-8")
    # 设置标签
    df.columns = ['Label', 'Source', 'Domain']  # 为每一行命名
    # 去掉来源，并对域名进行清洗，留下可识别的部分
    # 将合法域名标签改为1，dga的改为0
    dict_label = {'legit': 0, 'dga': 1}
    L = df['Label'].map(dict_label)
    # 对于域名也进行处理，留下中间的可识别部分作为n-gram的输入
    domain = np.array(df['Domain'])
    D = []
    for url in domain:
        # 删除.后的所有数据
        url = url.split(".", 1)[0]
        D.append(url)
    col_2 = ['Domain']
    D = pd.DataFrame(D, columns=col_2)
    df_n = pd.concat([D, L], axis=1)  # axis=1 横向合并

    return df_n


# 将域名向量化（已修改）
def domain_to_vector(list_Domain, list_Label):
    # list_Label = dataset['Label'].tolist()      # 域名标签列表
    # list_Domain = dataset['Domain'].tolist()    # 域名列表

    # 使用set()构造一个不重复的字典，并将域名里的字符换为其中对应的值，完成特征向量的构造
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(list_Domain)))}
    # 使用构造好的字典对域名进行处理
    list_data = [[valid_chars[y] for y in x] for x in list_Domain]
    # 以里面域名最大长度构造特征，小于最大长度的用0填充
    max_data_len = len(max(list_Domain, key=len, default=''))
    data_vector = sequence.pad_sequences(list_data, maxlen=max_data_len)
    # 将标签列表转化为ndarray
    list_label = np.array(list_Label)
    # 返回域名向量,标签列表,域名最大长度，域名字符对应的字典
    return data_vector, list_label, max_data_len, valid_chars


# 分割数据集以及测试集 73开
def data_split(data_vector, list_label, split_size):
    # 使用sklearn里的库函数进行份分割数据集
    train_data, test_data, train_label, test_label = train_test_split(data_vector, list_label, test_size=split_size)
    return train_data, test_data, train_label, test_label

# 绘制模型图
def plot_MyModel(model, savePath):
    plot_model(model, to_file=savePath, show_shapes=True, show_layer_names=True, rankdir='TB')
    plt.figure(figsize=(10, 10))
    img = plt.imread(savePath)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# 绘制混淆矩阵图像
def plot_confusion_metrics(history, savePath):

    plt.subplot(2, 2, 1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.7)
    plt.plot(history.epoch, history.history['loss'], color='red', label='Train')
    plt.plot(history.epoch, history.history['val_' + 'loss'], color='orange', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([0, 0.8])
    plt.legend()
    plt.savefig(savePath)

    metrics = ['accuracy', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(2, 2, n + 2)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.7)
        plt.plot(history.epoch, history.history[metric], color='red', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric], color='orange', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.ylim([0.6, 1])
        plt.legend()
        plt.savefig(savePath)

