from explib.trainers.base_trainer import BaseTrainer

class OjTrainer(BaseTrainer):

    def __init__(self, model, data, logger, evaluator, config):
        super().__init__(model, data, logger, config)
        self.evaluator = evaluator
        self.model.init_sampler(data.x_freqs, data.y_freqs)

    def train(self):
        print('==== Before Training ===='.format(self.model.global_epoch))
        emb_x, emb_y = self.model.get_emb(return_y=True)
        linear_model = self.model.get_linear_model()
        rec_metrics = self.evaluator.evaluate(emb_x, emb_y)
        print('='*20, 'test', '='*20)
        for k, v in rec_metrics.items():
            print('| {}: {}'.format(k, v))
        # for k, v in cls_metrics.items():
        #     print('| {}: {}'.format(k, v))
        print('=' * 50)
        super().train()

    def train_epoch(self):
        counter = cum_loss = 0.0
        for idx, batch in enumerate(self.data.iter_batch()):
            loss = self.train_step(batch)
            counter += 1
            cum_loss += loss
            c = self.model.global_step
            if c % self.config.summary_every == 0:
                print('step {:6d}:  Loss={:2.4f}'.format(c, cum_loss / counter))
                if cum_loss / counter < self.config.best_loss:
                    print('loss update: from ' + str(self.config.best_loss) + ' to ' + str(cum_loss / counter))
                    self.model.save(is_best=True)
                    print('best model saved!')
                    self.config.best_loss = loss
                cum_loss = counter = 0.0
            if c % self.config.save_every == 0:
                self.model.save()

        linear_model = self.model.get_linear_model()
        print('======= EPOCH {:3d} ======='.format(self.model.global_epoch))
        emb_x, emb_y = self.model.get_emb(return_y=True)
        train_rec = self.evaluator.evaluate(emb_x, emb_y)
        print('-'*20, 'TEST', '-'*20)
        for k, v in train_rec.items():
            print('| {}: {}'.format(k, v))

        valid_rec = self.evaluator.evaluate(emb_x, emb_y, 'valid')
        print('-'*20, 'VALID', '-'*20)
        for k, v in valid_rec.items():
            print('| {}: {}'.format(k, v))

        # 用测试集的结果来判断是否要保存
        # mean_metric = (sum(train_rec.values())+sum(train_cls.values())) / 5
        if valid_rec['MRR'] > self.config.best_metric:
            print('valid_rec mrr:', valid_rec['MRR'], 'best_metric:', self.config.best_metric)
            self.model.save_embeddings(self.model.global_epoch, emb_x, emb_y)
            self.config.best_metric = valid_rec['MRR']
        print('=' * 50)
        print()

    def train_step(self, batch):
        loss = self.model.train_step(*batch)
        return loss