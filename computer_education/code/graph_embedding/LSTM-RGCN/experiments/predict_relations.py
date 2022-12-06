from utils.misc import *
from utils.sampling import *
from torch_rgcn.utils import *
from torch_rgcn.models import RelationPredictor
from utils.data import load_link_prediction_data
import torch.nn.functional as F
import torch
import time

from utils.misc import evaluate, generate_true_dict

""" 
Relational Graph Convolution Network for relation prediction . 
Reproduced as described in https://arxiv.org/abs/1703.06103 (Section 4).
"""

# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Relation Prediction ', database='link_pred')


@ex.automain
def train(dataset,
          training,
          encoder,
          decoder,
          evaluation,
          _run):
    # DEBUG
    _run.add_artifact('./experiments/predict_links_batched.py')
    _run.add_artifact('./torch_rgcn/layers.py')
    _run.add_artifact('./torch_rgcn/models.py')
    _run.add_artifact('./torch_rgcn/utils.py')
    _run.add_artifact('./utils/data.py')
    _run.add_artifact('./utils/misc.py')

    # Set default values
    data_name = dataset["name"] if "name" in dataset else "unknown_data"
    save_dir = dataset["save_dir"] if "save_dir" in dataset else "unknown_data"
    log_file = dataset["log_file"] if "log_file" in dataset else "log"
    max_epochs = training["epochs"] if "epochs" in training else 5000
    use_cuda = training["use_cuda"] if "use_cuda" in training else False
    use_weight = training["use_weight"] if "use_weight" in training else False
    use_time = training["use_time"] if "use_time" in training else False
    graph_batch_size = training["graph_batch_size"] if "graph_batch_size" in training else None
    sampling_method = training["sampling_method"] if "sampling_method" in training else 'uniform'
    neg_sample_rate = training["negative_sampling"]["sampling_rate"] if "negative_sampling" in training else None
    head_corrupt_prob = training["negative_sampling"]["head_prob"] if "negative_sampling" in training else None
    edge_dropout = encoder["edge_dropout"] if "edge_dropout" in encoder else None
    decoder_l2_penalty = decoder["l2_penalty"] if "l2_penalty" in decoder else 0.0
    decoder_loss_expand = decoder["loss_expand"] if "loss_expand" in decoder else 0.0
    use_valid = evaluation["use_valid"] if "use_valid" in evaluation else False
    eval_every = evaluation["check_every"] if "check_every" in evaluation else 2000
    save_every = evaluation["save_every"] if "save_every" in evaluation else 100
    eval_batch_size = evaluation["batch_size"] if "batch_size" in evaluation else 16
    eval_verbose = evaluation["verbose"] if "verbose" in evaluation else False

    save_dir = 'experiments/' + save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, log_file+'.txt')
    log = open(log_file, 'w')

    # Check for available GPUs
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Note: Validation dataset will be used as test if this is not a test run
    (n2i, nodes), (r2i, relations), train, valid, test, all_triples = \
        load_link_prediction_data(name=data_name, use_weight=use_weight, use_time=use_time, use_valid=use_valid)

    num_nodes = len(n2i)
    num_relations = len(r2i)

    print('configs:', dataset, training, encoder, decoder, evaluation, file=log)
    print('num_nodes =', num_nodes, 'num_relations =', num_relations, file=log)

    test = torch.tensor(test, dtype=torch.long, device=device)  # Note: Evaluation is performed on the CPU
    valid = torch.tensor(valid, dtype=torch.long, device=device)

    if encoder["model"] == 'rgcn':
        model = RelationPredictor
    else:
        raise NotImplementedError(f'\'{encoder["model"]}\' encoder has not been implemented!')

    model = model(
        data_name=data_name,
        nnodes=num_nodes,
        nrel=num_relations,
        n2i=n2i,
        encoder_config=encoder,
        decoder_config=decoder
    )

    if use_cuda:
        model.cuda()

    if training["optimiser"]["algorithm"] == 'adam':
        optimiser = torch.optim.Adam
    elif training["optimiser"]["algorithm"] == 'adamw':
        optimiser = torch.optim.AdamW
    elif training["optimiser"]["algorithm"] == 'adagrad':
        optimiser = torch.optim.Adagrad
    elif training["optimiser"]["algorithm"] == 'sgd':
        optimiser = torch.optim.SGD
    else:
        raise NotImplementedError(f'\'{training["optimiser"]["algorithm"]}\' optimiser has not been implemented!')

    optimiser = optimiser(
        model.parameters(),
        lr=training["optimiser"]["learn_rate"],
        weight_decay=training["optimiser"]["weight_decay"]
    )

    # Edge dropout on self-loops
    if model.training and edge_dropout["self_loop_type"] != 'schlichtkrull-dropout':
        self_loop_keep_prob = 1 - edge_dropout["self_loop"]
    else:
        self_loop_keep_prob = 1
    sampling_function = select_sampling(sampling_method)

    epoch_counter = 0
    best_score = 0

    print("Start training...")
    for epoch in range(1, max_epochs + 1):
        print('='*30, f'[Epoch {epoch}] ', '='*30)

        epoch_counter += 1
        t1 = time.time()
        optimiser.zero_grad()
        model.train()

        with torch.no_grad():
            # 采样，直接用整个图或者采样batch_size大小的正样本
            print('sampling positives...')
            if graph_batch_size is None:
                # Use entire graph
                positives = train
                graph_batch_size = len(train)
                nodes_list = nodes
            else:
                # Randomly sample triples from graph
                positives = sampling_function(train, num_nodes, sample_size=graph_batch_size)
                nodes_list = extract_nodes(positives)

            positives = torch.tensor(positives, dtype=torch.long, device=device)
            sample_nodes_num = len(nodes_list)
            tuple_size = positives.size(1)

            # Generate negative samples triples for training
            # corrupt三元组来负采样，随机替换头或尾
            # .clone()[:, None, :]表示在dim=1上增加一个维度，并且在这个维度上，每个三元组复制neg_sample_rate次
            print('sampling negatives...')
            negatives = positives.clone()[:, None, :].expand(graph_batch_size, neg_sample_rate, tuple_size).contiguous()
            negatives = corrupt(negatives, num_nodes, head_corrupt_prob, device=device)
            batch_idx = torch.cat([positives, negatives], dim=0)

            # Label training data (0 for positive class and 1 for negative class)
            # 给边打标签（用于计算损失），正样本标记为1，负样本为0
            print('labeling...')
            pos_labels = torch.ones(graph_batch_size, 1, dtype=torch.float, device=device)
            neg_labels = torch.zeros(graph_batch_size * neg_sample_rate, 1, dtype=torch.float, device=device)
            train_lbl = torch.cat([pos_labels, neg_labels], dim=0).view(-1).to(device)

            graph = positives
            # Apply edge dropout
            if model.training and edge_dropout["general"] > 0.0:
                keep_prob = 1 - edge_dropout["general"]
                graph = graph[torch.randperm(graph.size(0))]
                sample_size = round(keep_prob * graph.size(0))
                graph = graph[sample_size:, :]

            # 在graph的末尾加上 交换了头和尾的三元组、子图节点列表的self-loop, [2*batch_size + sample_nodes_num, 5]
            graph_plus = add_inverse_and_self(graph, nodes_list, num_relations, self_loop_keep_prob, device)

        # Train model on training data
        print('begin forward...')
        predictions, penalty, embeddings = model(graph_plus, nodes_list, batch_idx, log=log)
        print('calculating loss...')
        loss = F.binary_cross_entropy_with_logits(predictions, train_lbl, pos_weight=torch.tensor([neg_sample_rate]))
        print('loss:', loss.item(), 'penalty:', penalty.item())
        loss = (loss * decoder_loss_expand) + (decoder_l2_penalty * penalty)
        print('new loss:', loss.item())

        t2 = time.time()
        print('backwarding...')
        loss.backward()
        print('optimising...')
        optimiser.step()
        t3 = time.time()

        _run.log_scalar("training.loss", loss.item(), step=epoch)
        print(f'[Epoch {epoch}] Loss: {loss.item():.7f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s ', file=log)

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == max_epochs:
            with torch.no_grad():
                model.eval()

                if graph_batch_size is not None:
                    graph = torch.tensor(train, dtype=torch.long)
                    graph_plus = add_inverse_and_self(graph, nodes, num_relations, self_loop_keep_prob, device)
                    embeddings = model(graph_plus, nodes, ret_embs=True, log=log, save_dir=save_dir)
                print('embeddings:', embeddings, file=log)
                valid_rec_metrics = evaluate(
                    test_set=valid,
                    embeddings=embeddings,
                    batch_size=eval_batch_size,
                    verbose=eval_verbose,
                    log=log,
                    save_dir=save_dir)

                _run.log_scalar("training.loss", loss.item(), step=epoch)

                for key, value in valid_rec_metrics.items():
                    _run.log_scalar(key, value, step=epoch)

                print(f'[Epoch {epoch}]', file=log)
                print("=" * 20, "Evaluation", "=" * 20, file=log)
                for key, value in valid_rec_metrics.items():
                    print(f'{key}: {value} \t', file=log)
                    print(f'{key}: {value} \t')

                if valid_rec_metrics['MRR'] > best_score:
                    best_score = valid_rec_metrics['MRR']
                    model.save_models(save_dir)
                    best_embeddings = embeddings
                    best_epoch = epoch

        log.flush()

    print('Training is complete!', file=log)

    with torch.no_grad():
        print("Starting final evaluation...", file=log)
        model.eval()

        test_rec_metrics = evaluate(
            test_set=test,
            embeddings=best_embeddings,
            batch_size=eval_batch_size,
            verbose=eval_verbose,
            log=log,
            save_dir=save_dir)

        for key, value in test_rec_metrics.items():
            _run.log_scalar(key, value, step=epoch_counter)

        print("=" * 20, "Final Evaluation", "=" * 20, file=log)
        print(f'[Final Scores] '
              f'Total Epoch {epoch_counter} \t Best Epoch {best_epoch}', file=log)
        for key, value in test_rec_metrics.items():
            print(f'{key}: {value} \t', file=log)

        print("=" * 20, "Final Evaluation", "=" * 20)
        print(f'[Final Scores] '
              f'Total Epoch {epoch_counter} \t Best Epoch {best_epoch}')
        for key, value in test_rec_metrics.items():
            print(f'{key}: {value} \t')

        log.flush()
