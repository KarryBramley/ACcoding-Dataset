import argparse
import sys
import json
import os

class Config:

    def __init__(self, kvs):
        self.__dict__.update(kvs)

def parse_args(args):
    parser = argparse.ArgumentParser(prog='OJ', description='Applications for OJ Data')
    parser.add_argument('--task', type=str, default='dkvmn', help='task name which need to run.')
    parser.add_argument('--data', type=str, default='oj', help='oj data or raw data to run.')
    parser.add_argument('--config', type=str, default=r'configs/config.json', help='configuration file path')
    return parser.parse_args(args)


def parse_json_args(file_path):
    config_file = open(file_path)
    json_config = json.load(config_file)
    config_file.close()
    return json_config


def process_config(json_file):
    with open(json_file, 'r') as jf:
        config_dict = json.load(jf)

    config = Config(config_dict)

    return config, config_dict


# Knowledge Tracing tasks:
# Deep Knowledge Tracing task
def dkt_task(args, data):
    print("run DKT task with {} dataset...".format(data))
    if data == 'oj':
        # whether use data group by knowledge points
        if args.use_kp == True:
            args.f = 'data/experiment_data/DKT/oj/contestData2019_knowledges.csv'
    elif data == 'assist':
        args.model_name = 'assist_model'
        args.f = 'data/experiment_data/DKT/assistment/assistments_skill_builder_data.csv'
        args.w = 'knowledge_tracing/DKT/assist_model/weights/bestmodel'
        args.log_dir = 'knowledge_tracing/DKT/assist_model/logs'
    else:
        raise Exception("Unsupported data, DKT task only support 'oj' and 'assist' data")

    from knowledge_tracing.DKT.run_dkt import run
    run(args)


def dkvmn_task(args, data):
    print("run DKVMN task with {} dataset...".format(data))
    # default config is set with data='oj', no more need to preprocess.
    if data == 'oj':
        args.data_name = 'contest2019'
    elif data == 'assist':
        args.batch_size = 32
        args.q_embed_dim = 50
        args.qa_embed_dim = 200
        args.memory_size = 20
        args.n_question = 110
        args.seqlen = 200
        args.data_dir = 'data/experiment_data/DKVMN/assist2009_updated'
        args.data_name = 'assist2009_updated'
        args.load = 'assist2009_updated'
        args.save = 'assist2009_updated.pkl'
    elif data == 'statics':
        args.batch_size = 32
        args.q_embed_dim = 50
        args.qa_embed_dim = 100
        args.memory_size = 50
        args.n_question = 1223
        args.seqlen = 6
        args.data_dir = 'data/experiment_data/DKVMN/STATICS'
        args.data_name = 'STATICS'
        args.load = 'STATICS'
        args.save = 'STATICS.pkl'
    else:
        raise Exception("Unsupported data, DKVMN task only support 'oj', 'assist' and 'statics' data")
    from knowledge_tracing.DKVMN.main_old import dkvmn_train
    dkvmn_train(args)


def bine_task(args, data):
    print("run BiNE task with {} dataset...".format(data))
    if data == 'oj':
        root_path = "data/experiment_data/BINE/oj/"
        if args.use_daily == True:
            pre = "daily_"
        else:
            pre = ""
        args.train_data = root_path + pre + args.train_data
        args.test_data = root_path + pre + args.test_data
        args.model_name = "oj_model"
        args.vectors_u = root_path + pre + args.vectors_u
        args.vectors_v = root_path + pre + args.vectors_v
    elif data == 'dblp':
        root_path = "data/experiment_data/BINE/dblp/"
        args.train_data = root_path + args.train_data
        args.test_data = root_path  + args.test_data
        args.model_name = "dblp_model"
        args.vectors_u = root_path + args.vectors_u
        args.vectors_v = root_path + args.vectors_v
    else:
        raise Exception("Unsupported data, BiNE task only support 'oj' and 'dblp' data")
    from graph_embedding.BINE.train import bine_train
    bine_train(args)


def ige_task(args, data):
    print("run IGE task with {} dataset...".format(data))
    if data == 'dblp':
        from graph_embedding.IGE.explib.experiments.exp_ige_dblp import exp_ige_dblp
        exp_ige_dblp(args)
    elif data == 'oj':
        print("run IGE task with {} dataset...".format(data))
        from graph_embedding.IGE.explib.experiments.exp_ige_oj import exp_ige_oj
        exp_ige_oj(args)
    else:
        raise Exception("Unsupported data, IGE task only support 'oj' and 'dblp' data")


def rhine_task(args, data):
    if data == 'oj':
        from graph_embedding.RHINE.code.trainRHINE import train_model
        data_set = args.data_set
        mode = args.mode
        train_model(data_set, mode)


def transx_task(args, data):
    print("run transE, transR, transH task with {} dataset...".format(data))
    if data == 'oj':
        from graph_embedding.TRANSX.train_transx_OJ2019 import oj_transx
        oj_transx()

    elif data == 'FB15K237':
        from graph_embedding.TRANSX.examples import train_transe_FB15K237
        from graph_embedding.TRANSX.examples import train_transh_FB15K237
        from graph_embedding.TRANSX.examples import train_transr_FB15K237
    else:
        raise Exception("Unsupported data, IGE task only support 'oj' and 'dblp' data")


def main(argv):
    args = parse_args(argv[1:])
    print(args)
    config_dict = parse_json_args(args.config)
    if args.task == 'dkt':
        configs = Config(config_dict[args.task])
        dkt_task(configs, args.data)
    elif args.task == 'dkvmn':
        configs = Config(config_dict[args.task])
        dkvmn_task(configs, args.data)
    elif args.task == 'bine':
        configs = Config(config_dict[args.task])
        bine_task(configs, args.data)
    elif args.task == 'ige':
        configs = Config(config_dict[args.task])
        ige_task(configs, args.data)
    elif args.task == 'rhine':
        configs = Config(config_dict[args.task])
        rhine_task(configs, args.data)
    elif args.task == 'transx':
        configs = Config(config_dict[args.task])
        transx_task(configs, args.data)
    else:
        raise Exception("Unsupported task")

if __name__ == '__main__':
    main(sys.argv)