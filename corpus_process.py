import one_hot_services
import config
import re

class Get_data(object):
    '''
    用于对人民日报1998语料进行预处理
    '''
    def __init__(self):
        # 获得语料的路径
        self.corpus_path = config.origin_path() + '/corpus.txt'

    def get_data(self, limit = 100):
        '''
        获取语料并进行处理
        :param limit: 获取的语料条数
        :return: text:[[w1,...,wm],...,[wi,...,wn]], m, n, i < vocab_size 且不一定按顺序递增或递减
        label:[[label1,...,labell],...,[labelj,...,labelk]], l, j, k < label_size 且不一定按顺序递增或递减
        '''
        # 获取limit数量的语料
        corpus = open(self.corpus_path, 'r').readlines()[:limit]
        # 去除换行符
        corpus = [re.sub('\n','',str(i)) for i in corpus]
        # text，label分开
        texts = [self.text_label_split(i)[0] for i in corpus]
        labels = [self.text_label_split(i)[1] for i in corpus]
        # 去除text中的[
        texts = [self.format_control(i, '[') for i in texts]
        # 去除label中的]和]之后的标注
        labels = [self.hyper_format_control(i, ']') for i in labels]
        # 去除空list
        texts = [i for i in texts if i != []]
        labels = [i for i in labels if i != []]
        return texts, labels

    def format_control(self, text, control_mark):
        '''
        去除字符
        :param text:
        :param control_mark: 要被去除的字符
        :return:
        '''
        if type(text) == str:
            text = text.replace(control_mark, '')
            return text
        elif type(text) == list:
            text = [i.replace(control_mark, '') if control_mark in i else i for i in text]
            return text
        else:
            raise ValueError('controlled text type must be list or string')

    def hyper_format_control(self, text, control_mark):
        '''
        去除字符，仅保留被去除字符位置往前的数据片段
        :param text:
        :param control_mark: 要被去除的字符
        :return:
        '''
        if type(text) == str:
            text = text[:text.index(control_mark)]
            return text
        elif type(text) == list:
            text = [i[:i.index(control_mark)] if control_mark in i else i for i in text]
            return text
        else:
            raise ValueError('controlled text type must be list or string')

    def text_label_split(self, single_text, split_mark = ' '):
        '''
        分离text和label
        :param single_text: 单条数据
        :param split_mark: 按照何种字符分离
        :return: text, label
        '''
        single_text = single_text.split(split_mark)
        single_text = [i for i in single_text if i != '']
        single_text = [(i.split('/')[0], i.split('/')[1]) for i in single_text]
        text = [i[0] for i in single_text]
        label = [i[1] for i in single_text]
        return text, label

    def work_flow(self, limit = 100):
        '''
        工作流，最后从这里输出数据
        :param limit: 数据数量，默认100条
        :return:
        '''
        texts, labels = self.get_data(limit = limit)
        # 获得词集合
        word_collection = sum(texts, [])
        # 获得标签集合
        label_collection = sum(labels, [])
        # 制作字典
        word_ids = one_hot_services.make_dict(word_collection)
        label_ids = one_hot_services.make_dict(label_collection)
        # 根据字典为词和标签提供index
        texts = [one_hot_services.word2id(i, word_ids) for i in texts]
        labels = [one_hot_services.word2id(i, label_ids) for i in labels]
        return texts, labels
