# coding:utf-8
# author: lu yf
# create date: 2018/2/6
import sys
sys.path.append(r'graph_embedding/RHINE/code')
import config
import models
import evaluation
import time
import warnings

warnings.filterwarnings('ignore')


def train_model(data_set, mode):
    """
    train models
    :param data_set:
    :param mode: relation categories
    :return:
    """
    con = config.Config(data_set)
    con.set_in_path("../data/" + data_set + "/")
    con.set_cls_data(data_set)

    con.set_work_threads(16)
    # 模型训练的时间
    con.set_train_times(500)
    # 训练IR关系的loss时需要的batch数量
    con.set_IRs_nbatches(100)
    # 训练AR关系的loss时需要的batch数量
    con.set_ARs_nbatches(100)
    # 优化器的learning rate
    con.set_alpha(0.005)
    # Margin-based loss的margin大小
    con.set_margin(1)
    # 应该是Embedding的维度
    con.set_dimension(200)
    # 应该是实体的负采样率
    con.set_ent_neg_rate(1)
    # 应该是关系的负采样率，负采样率为0表示不采样关系
    con.set_rel_neg_rate(0)

    con.set_opt_method("SGD")
    con.set_evaluation(True)

    export_file = "../res/" + data_set + "/models.tf"
    embedding_file = "../res/" + data_set + "/embedding.json"
    con.set_export_files(export_file)
    con.set_out_files(embedding_file)
    con.init()
    con.set_model(models.RHINE)

    linear_model, best_epoch = con.run()

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print (f'The best epoch is epoch {best_epoch}')
    exp = evaluation.Evaluation(data_set)
    emb_dict = exp.load_emb(embedding_file)
    exp.evaluation(emb_dict, 'test', linear_model)


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    data_set = 'daily_2019_valid'
    mode = 'up+pk+upk'

    print ('mode: {}'.format(mode))
    train_model(data_set, mode)
