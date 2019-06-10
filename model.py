import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from corpus_process import Get_data
import numpy as np
import one_hot_services

def padding_sequence(data_list, max_len = 50, value = 0):
    '''
    将不定长的句子填充为定长句子
    :param data_list: 数据集，形如[[1,...,m],...,[i,...n]], m, n, i < vocab_size 且不一定按顺序递增或递减
    :param max_len: 最大句子长度，即填充长度
    :param value: 填充的占位符的index
    :return:
    '''
    data = keras.preprocessing.sequence.pad_sequences(
        data_list, value=value, padding='post', maxlen=max_len
    )
    return data

def lstm(class_num = 30,vocab_size = 200000, embedding_size = 200, sequence_length = 50, hidden_dim = 64):
    '''
    keras下的lstm框架
    :param class_num: 标签数量，默认30
    :param vocab_size: 词的数量
    :param embedding_size: embedding后的维度
    :param sequence_length: 序列长度
    :param hidden_dim: lstm层的维度
    :return:
    '''
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_size, input_length = sequence_length))
    model.add(layers.LSTM(hidden_dim, return_sequences = True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(class_num, activation = 'softmax'))

    return model

def early_stop(patience = 1, min_delta = 1e-2, monitor = 'loss'):
    '''
    使用early stop的方法，当loss不再下降时，停止训练
    :param patience: 当loss不再下降时继续训练的batch数量
    :param min_delta: loss的阈值，loss需要下降到该值以下
    :param monitor: 需要监视的指标，默认为loss
    :return:
    '''
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=1
        )
    ]
    return callbacks

if __name__ == '__main__':
    G = Get_data()
    texts, labels = G.work_flow(1000)
    # 获取数据
    x_train = padding_sequence(texts[int(0.2*len(texts)):])
    y_train = padding_sequence(labels[int(0.2*len(texts)):])
    x_test = padding_sequence(texts[:int(0.8*len(texts))])
    y_test = padding_sequence(labels[:int(0.8*len(texts))])
    # 获取词数量和标签数量
    vocab_num = np.max(x_train) + 1
    label_num = np.max(y_train) + 1
    # 标签转换为one-hot，tf.keras实际上支持标签为index的形式：SparseCategoricalCrossentropy，但是因为不明原因，无法使用
    y_train = [one_hot_services.convert_to_one_hot(i, label_num) for i in y_train]
    y_test = [one_hot_services.convert_to_one_hot(i, label_num) for i in y_test]
    # 转换为array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 获得model
    model = lstm(vocab_size = vocab_num, class_num = label_num, embedding_size = 200, sequence_length = 50, hidden_dim = 64)

    print(model.summary())

    # 设定early-stop为回调函数
    callbacks = early_stop()

    # 设定损失函数等
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    # 训练10个epoch，每个batch有32条数据
    history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, callbacks = callbacks)

    # 预测并打印结果
    result = model.evaluate(x_test, y_test)
    print(result)