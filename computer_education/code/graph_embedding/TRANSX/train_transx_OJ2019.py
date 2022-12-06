import sys
sys.path.append(r'graph_embedding/TRANSX')
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransH, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os


def oj_transx():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path='./benchmarks/OJ2019/',
        nbatches=100,
        threads=8,
        sampling_mode='normal',
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path="./benchmarks/OJ2019/",
        sampling_mode='link',
        type_constrain=False)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True)

    model_e = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size())

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model_h = NegativeSampling(
        model=transh,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size()
    )

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True,
        rand_init=False)

    model_r = NegativeSampling(
        model=transr,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size()
    )

    checkpoint_path = './checkpoint/OJ2019'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    parameters_path = './result/OJ2019'
    if not os.path.exists(parameters_path):
        os.makedirs(parameters_path)

    print('='*25 + 'TransE' + '='*25)
    print('-'*20 + 'train transE model' + '-'*20)
    # pretrain transe
    trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 50, alpha = 1.0, use_gpu = False)
    trainer.run()
    transe.save_checkpoint(os.path.join(checkpoint_path, 'transe.ckpt'))
    transe_parameters = transe.get_parameters()
    transe.save_parameters(os.path.join(parameters_path, 'transe.json'))

    print('-'*20 + 'test trainsE model' + '-'*20)
    # test transe
    transe.load_checkpoint(os.path.join(checkpoint_path, 'transe.ckpt'))
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=False)
    tester.run_link_prediction(type_constrain=False)

    transe.load_checkpoint(os.path.join(checkpoint_path, 'transe.ckpt'))
    transe_parameters = transe.get_parameters()

    print('='*25 + 'TransR' + '='*25)
    print('-'*20 + 'train transR model' + '-'*20)
    # train transr
    transr.set_parameters(transe_parameters)
    trainer = Trainer(model=model_r, data_loader=train_dataloader, train_times=50, alpha=1.0, use_gpu=False)
    trainer.run()
    transr.save_checkpoint(os.path.join(checkpoint_path, 'transr.ckpt'))
    transr.save_parameters(os.path.join(parameters_path, 'transr.json'))

    print('-'*20 + 'test trainsR model' + '-'*20)
    # test transr
    transr.load_checkpoint(os.path.join(checkpoint_path, 'transr.ckpt'))
    tester = Tester(model=transr, data_loader=test_dataloader, use_gpu=False)
    tester.run_link_prediction(type_constrain=False)

    print('='*25 + 'TransH' + '='*25)
    print('-'*20 + 'train transH model' + '-'*20)
    # train transh
    trainer = Trainer(model=model_h, data_loader=train_dataloader, train_times=50, alpha=0.5, use_gpu=False)
    trainer.run()
    transh.save_checkpoint(os.path.join(checkpoint_path, 'transh.ckpt'))
    transh.save_parameters(os.path.join(parameters_path, 'transh.json'))

    print('-'*20 + 'test trainsH model' + '-'*20)
    # test the model
    transh.load_checkpoint(os.path.join(checkpoint_path, 'transh.ckpt'))
    tester = Tester(model=transh, data_loader=test_dataloader, use_gpu=False)
    tester.run_link_prediction(type_constrain=False)

if __name__ == '__main__':
    oj_transx()