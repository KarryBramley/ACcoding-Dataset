# import sys
# sys.path.append(r"graph_embedding/IGE")
import torch
from explib.dataloaders.oj_loader import OjLoader as Loader
from explib.models.ige_model import IGEModel as Model
from explib.loggers.print_logger import PrintLogger as Logger
from explib.evaluators.oj_evaluator import OjEvaluator as Evaluator
from explib.trainers.oj_trainer import OjTrainer as Trainer

from explib.utils.config import get_args, process_config
from explib.utils.io import create_dirs

def exp_ige_oj(args):
    if args.config is None:
        args.config = "explib/configs/exp_ige_oj.json"
    config = process_config(args.config)

    create_dirs([config.summary_dir, config.ckpt_dir])

    torch.manual_seed(0)

    data = Loader(config)
    model = Model(config)
    evaluator = Evaluator(config)
    logger = Logger(config)
    trainer = Trainer(model, data, logger, evaluator, config)

    trainer.train()

    print('=' * 20, 'TEST', '=' * 20)
    emb_dict = model.load_embeddings()
    test_rec = evaluator.evaluate(emb_dict['emb_x'], emb_dict['emb_y'], eval_data='test')
    print(f"""The best epoch is {emb_dict['best_epoch']}.""")
    for k, v in test_rec.items():
        print('| {}: {}'.format(k, v))
    # for k, v in test_cls.items():
    #     print('| {}: {}'.format(k, v))


if __name__ == '__main__':
    args = get_args()
    exp_ige_oj(args)


