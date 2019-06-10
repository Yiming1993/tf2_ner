# coding = 'utf-8'
import numpy as np

def make_dict(sentences, start_num = 1):
    '''
    制作词典
    :param sentences: list，词集合
    :param start_num: 最开始的index号码
    :return:
    '''
    iterT = start_num
    word_ids = {}
    for i in list(set(sentences)):
        if i not in word_ids:
            word_ids[i] = iterT
            iterT += 1
        else:
            pass
    word_ids['UNK'] = iterT + 1
    return word_ids

def word2id(cut_sentence, word_ids):
    '''
    将词转换为词典中的index
    :param cut_sentence: 一个句子
    :param word_ids: 词典
    :return:
    '''
    keys = word_ids.keys()
    # 如果在词典内，就提供index，否则，标注为UNK
    cut_sentence = [i if i in keys else 'UNK' for i in cut_sentence]
    return [word_ids[i] for i in cut_sentence]

def convert_to_one_hot(y, one_hot_dim):
    '''
    将index转换为one-hot
    :param y:
    :param one_hot_dim:
    :return:
    '''
    y = np.array(y)
    return np.eye(one_hot_dim)[y.reshape(-1)]