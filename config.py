#coding = 'utf-8'
import os

def origin_path():
    '''
    使用config文件提供原始路径等
    :return: 运行文件所在的根目录
    '''
    origin_path = os.path.dirname(os.path.abspath(__file__))
    # print(origin_path)

    return origin_path