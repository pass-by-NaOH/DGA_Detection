import json

import keras
import tensorflow as tf
from tensorflow_core.python.keras.layers import MaxPooling1D, Concatenate, Softmax, Lambda, \
    GlobalAveragePooling1D, BatchNormalization, Flatten
from tensorflow_core.python.keras.layers import Embedding, LSTM, Dropout, Dense, Activation, Conv1D
from tensorflow_core.python.keras import Model, Input
import pickle

# cnn_At_lstm模型
def cnn_At_lstm(max_features, max_data_len):
    # 输入层
    input_layer = Input(shape=(max_data_len,), dtype='int32')

    # 词嵌入
    embed_layer = Embedding(input_dim=max_features, output_dim=128, input_length=max_data_len)(input_layer)


    # 卷积-LSTM层，2、3、4代表2、3、4gram
    conv1 = Conv1D(128, 2, activation='relu', padding='same')(embed_layer)
    max1 = MaxPooling1D()(conv1)
    batch_normalization1 = BatchNormalization()(max1)
    lstm1 = LSTM(128)(batch_normalization1)
    attention_pre1 = Dense(1, name='attention_vec1')(lstm1)
    attention_probs1 = Softmax()(attention_pre1)
    attention_mul1 = Lambda(lambda x: x[0] * x[1])([attention_probs1, lstm1])
    dropout1 = Dropout(0.5)(attention_mul1)

    conv2 = Conv1D(128, 3, activation='relu', padding='same')(embed_layer)
    max2 = MaxPooling1D()(conv2)
    batch_normalization2 = BatchNormalization()(max2)
    lstm2 = LSTM(128)(batch_normalization2)
    attention_pre2 = Dense(1, name='attention_vec2')(lstm2)
    attention_probs2 = Softmax()(attention_pre2)
    attention_mul2 = Lambda(lambda x: x[0] * x[1])([attention_probs2, lstm2])
    dropout2 = Dropout(0.5)(attention_mul2)

    conv3 = Conv1D(128, 4, activation='relu', padding='same')(embed_layer)
    max3 = MaxPooling1D()(conv3)
    batch_normalization3 = BatchNormalization()(max3)
    lstm3 = LSTM(128)(batch_normalization3)
    attention_pre3 = Dense(1, name='attention_vec3')(lstm3)
    attention_probs3 = Softmax()(attention_pre3)
    attention_mul3 = Lambda(lambda x: x[0] * x[1])([attention_probs3, lstm3])
    dropout3 = Dropout(0.5)(attention_mul3)

    # 最后将三个卷积层的结果拼在一起
    conv_layer = Concatenate(axis=-1)([dropout1, dropout2, dropout3])
    flat = Flatten()(conv_layer)
    out = Dense(1, activation='sigmoid')(flat)

    # 评价函数后得到所有的参数
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # 将整个模型拼合在一起
    model = Model(input_layer, out)
    # 模型使用二元交叉熵损失函数（二分类问题），优化器使用Adam优化器，评价函数使用“准确率”
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    # 模型信息
    model.summary()

    return model

# cnn_lstm模型
def cnn_lstm(max_features, max_data_len):
    # 输入层
    input_layer = Input(shape=(max_data_len,), dtype='int32')

    # 词嵌入
    embed_layer = Embedding(input_dim=max_features, output_dim=128, input_length=max_data_len)(input_layer)

    # 卷积层，2、3、4代表2、3、4gram
    conv1 = Conv1D(128, 2, activation='relu', padding='same')(embed_layer)
    max1 = MaxPooling1D()(conv1)
    batch_normalization1 = BatchNormalization()(max1)
    lstm1 = LSTM(128)(batch_normalization1)
    dropout1 = Dropout(0.5)(lstm1)

    conv2 = Conv1D(128, 3, activation='relu', padding='same')(embed_layer)
    max2 = MaxPooling1D()(conv2)
    batch_normalization2 = BatchNormalization()(max2)
    lstm2 = LSTM(128)(batch_normalization2)
    dropout2 = Dropout(0.5)(lstm2)

    conv3 = Conv1D(128, 4, activation='relu', padding='same')(embed_layer)
    max3 = MaxPooling1D()(conv3)
    batch_normalization3 = BatchNormalization()(max3)
    lstm3 = LSTM(128)(batch_normalization3)
    dropout3 = Dropout(0.5)(lstm3)

    # 最后将三个卷积层的结果拼在一起
    conv_layer = Concatenate(axis=-1)([dropout1, dropout2, dropout3])

    flat = Flatten()(conv_layer)
    out = Dense(1, activation='sigmoid')(flat)

    # 评价函数后得到所有的参数
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # 将整个模型拼合在一起
    model = Model(input_layer, out)
    # 模型使用二元交叉熵损失函数（二分类问题），优化器使用Adam优化器，评价函数使用“准确率”
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    # 模型信息
    model.summary()

    return model



# cnn模型
def cnn(max_features, max_data_len):
    # 输入层
    input_layer = Input(shape=(max_data_len,), dtype='int32')

    # 词嵌入
    embed_layer = Embedding(input_dim=max_features, output_dim=128, input_length=max_data_len)(input_layer)

    # 卷积层
    conv1 = Conv1D(128, 2, activation='relu', padding='same')(embed_layer)
    max1 = GlobalAveragePooling1D()(conv1)
    batch_normalization1 = BatchNormalization()(max1)
    dropout1 = Dropout(0.5)(batch_normalization1)

    conv2 = Conv1D(128, 3, activation='relu', padding='same')(embed_layer)
    max2 = GlobalAveragePooling1D()(conv2)
    batch_normalization2 = BatchNormalization()(max2)
    dropout2 = Dropout(0.5)(batch_normalization2)

    conv3 = Conv1D(128, 4, activation='relu', padding='same')(embed_layer)
    max3 = GlobalAveragePooling1D()(conv3)
    batch_normalization3 = BatchNormalization()(max3)
    dropout3 = Dropout(0.5)(batch_normalization3)

    # 最后将三个卷积层的结果拼在一起
    conv_layer = Concatenate(axis=-1)([dropout1, dropout2, dropout3])
    flat = Flatten()(conv_layer)
    out = Dense(1, activation='sigmoid')(flat)

    # 评价函数后得到所有的参数
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # 将整个模型拼合在一起
    model = Model(input_layer, out)
    # 模型使用二元交叉熵损失函数（二分类问题），优化器使用Adam优化器，评价函数使用“准确率”
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    # 模型信息
    model.summary()

    return model


# lstm模型
def lstm(max_features, max_data_len):
    # 输入层
    input_layer = Input(shape=(max_data_len,), dtype='int32')

    # 词嵌入
    embed_layer = Embedding(input_dim=max_features, output_dim=128, input_length=max_data_len)(input_layer)
    lstm = LSTM(128)(embed_layer)
    dropout = Dropout(0.5)(lstm)
    flat = Flatten()(dropout)
    out = Dense(1, activation='sigmoid')(flat)

    # 评价函数后得到所有的参数
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    model = Model(input_layer, out)
    # 模型使用二元交叉熵损失函数（二分类问题），优化器使用Adam优化器，评价函数使用“准确率”
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    # 模型信息
    model.summary()

    return model


# 保存模型
def model_save_h5(model, modelPath):
    model.save(model, modelPath)


def model_load_h5(modelPath):
    tf.keras.models.load_model(modelPath)


# 保存为json
def model_save_json(model, modelPath):
    model_json = model.to_json()
    with open(modelPath, 'w') as f:
        json.dump(model_json, f)
        print('模型的架构json文件保存完成！')


# 加载模型
def model_load_json(modelPath):
    with open(modelPath, 'r') as f:
        model_json = json.load(f)
        # 模型的加载
    model = tf.keras.models.model_from_json(model_json)
    return model


# 将训练的history保存为文本文件
def save_history(history, historyPath):
    with open(historyPath, 'wb') as file_txt:
        pickle.dump(history.history, file_txt)

# 加载history
def load_history(historyPath):
    with open(historyPath, 'rb') as file_txt:
        history = pickle.load(file_txt)
    return history