import torch
import shutil
import json
import os


class BaseModel:

    def __init__(self, model, optimizer, config):
        self.config = config
        self.global_step = 0
        self.global_epoch = 0
        self.model = model
        self.optimizer = optimizer

    def save(self, is_best=False):
        # print('Saving Model...')
        data = {'epoch': self.global_epoch,
                'step': self.global_step,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()}
        fname = os.path.join(self.config.ckpt_dir,
                             'ckpt.pth.tar')
        torch.save(data, fname)
        if is_best:
            shutil.copyfile(fname, os.path.join(self.config.ckpt_dir,
                                                'best.pth.tar'))
            # print('Best Model Saved!')

        # print('Model Saved!')

    def load(self, load_best=True, load_optimizer=True):
        print('Loading Model...')
        if load_best:
            fname = os.path.join(self.config.ckpt_dir, 'best.pth.tar')
            if not os.path.exists(fname):
                fname = os.path.join(self.config.ckpt_dir, 'ckpt.pth.tar')
        else:
            fname = os.path.join(self.config.ckpt_dir, 'ckpt.pth.tar')
        if os.path.exists(fname):
            data = torch.load(fname)
            self.global_step = data['step']
            self.global_epoch = data['epoch']
            self.model.load_state_dict(data['model_state'])
            # self.model.load_state_dict({ k.replace('module.', ''):v for k,v in data['model_state'].items()})
            if load_optimizer:
                self.optimizer.load_state_dict(data['optimizer_state'])
            print('Model Loaded!')
        else:
            print('No checkpoint Found!')

    def save_embeddings(self, best_epoch, emb_x, emb_y, path=None):
        if path is None:
            path = os.path.join(self.config.ckpt_dir, 'embeddings.json')
        f = open(path, 'w')
        emb_dict = {'best_epoch': best_epoch, 'emb_x': emb_x.tolist(), 'emb_y': emb_y.tolist()}
        f.write(json.dumps(emb_dict))
        print('embeddings saved!')
        f.close()

    def load_embeddings(self, path=None):
        if path is None:
            path = os.path.join(self.config.ckpt_dir, 'embeddings.json')
        f = open(path, 'r')
        emb_dict = json.load(f)
        f.close()
        return emb_dict

    def train_step(self, *args):
        raise NotImplementedError
