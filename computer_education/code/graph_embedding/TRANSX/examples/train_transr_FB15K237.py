import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/FB15K237/",
	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200,
	dim_r = 200,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = False)
trainer.run()
parameters = transe.get_parameters()
parameters_path = './result'
if not os.path.exists(parameters_path):
	os.makedirs(parameters_path)
transe.save_parameters(os.path.join(parameters_path, 'transr_transe.json'))

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 50, alpha = 1.0, use_gpu = False)
trainer.run()
checkpoint_path = './checkpoint'
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)
transr.save_checkpoint(os.path.join(checkpoint_path, 'transr.ckpt'))

# test the model
transr.load_checkpoint(os.path.join(checkpoint_path, 'transr.ckpt'))
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)