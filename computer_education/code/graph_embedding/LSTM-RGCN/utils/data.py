import rdflib as rdf
import pandas as pd
import gzip, os, pickle, tqdm
from collections import Counter
from rdflib import URIRef

S = os.sep

def locate_file(filepath):
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return directory + '/' + filepath

def st(node):
    """
    Maps an rdflib node to a unique string. We use str(node) for URIs (so they can be matched to the classes) and
    we use .n3() for everything else, so that different nodes don't become unified.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L16
    """
    if type(node) == URIRef:
        return str(node)
    else:
        return node.n3()

def add_neighbors(set, graph, node, depth=2):
    """
    Adds neighboring nodes that are n nodes away from the target label

    Source: https://github.com/pbloem/gated-rgcn/blob/f5031a3cbb485aab964aea7b6856d0584155c820/kgmodels/data.py#L29
    """
    if depth == 0:
        return

    for s, p, o in graph.triples((node, None, None)):
        set.add((s, p, o))
        add_neighbors(set, graph, o, depth=depth-1)

    for s, p, o in graph.triples((None, None, node)):
        set.add((s, p, o))
        add_neighbors(set, graph, s, depth=depth-1)

def load_strings(file):
    """ Read triples from file """
    with open(file, 'r') as f:
        return [line.split() for line in f]


def load_weighted_strings(file, use_weight=False, use_time=False):
    """ Read triples and relation weight from file """
    with open(file, 'r') as f:
        weighted_triples = []
        for line in f.readlines():
            s, r, o = line.split('\t')
            p = r
            w = 1
            t = None

            if use_weight or use_time:
                # 如果数据里有weight或者time，relation部分按照 “edge_type/weight/time”的顺序排列
                r_list = r.split('/')
                p = r_list[0]
                if use_weight:
                    w = r_list[1]
                if use_time:
                    t = r_list[-1]
            # 五元组：[s-头结点，p-关系，w-权重，t-时间，o-尾节点]
            weighted_triples.append([s, p, w, t, o])
        return weighted_triples


def load_link_prediction_data(name=None, use_weight=False, use_time=False, use_valid=False, limit=None):
    """
    Load knowledge graphs for relation Prediction  experiment.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L218

    :param name: Dataset name ('aifb', 'am', 'bgs' or 'mutag')
    :param use_test_set: If true, load the canonical test set, otherwise load validation set from file.
    :param limit: If set, only the first n triples are used.
    :return: Relation prediction test and train sets:
              - train: list of edges [subject, (predicate, weight), object]
              - test: list of edges [subject, (predicate, weight), object]
              - all_triples: sets of tuples (subject, (predicate, weight), object)
    """

    if name.lower() == 'fb15k':
        train_file = locate_file('data/fb15k/train.txt')
        val_file = locate_file('data/fb15k/valid.txt')
        test_file = locate_file('data/fb15k/test.txt')
    elif name.lower() == 'oj_fb_2019':
        train_file = locate_file('data/oj_fb_2019/train.txt')
        val_file = locate_file('data/oj_fb_2019/valid.txt')
        test_file = locate_file('data/oj_fb_2019/test.txt')
    elif name.lower() == 'oj_fb_mini':
        train_file = locate_file('data/oj_fb_mini/train.txt')
        val_file = locate_file('data/oj_fb_mini/test.txt')
        test_file = locate_file('data/oj_fb_mini/test.txt')
    else:
        raise ValueError(f'Could not find \'{name}\' dataset')

    train = load_weighted_strings(train_file, use_weight, use_time)
    val = load_weighted_strings(val_file, use_weight, use_time)
    test = load_weighted_strings(test_file, use_weight, use_time)

    if not use_valid:
        test = val

    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # Mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()

    for s, p, _, _, o in train + val + test:
        nodes.add(s)
        rels.add(p)
        nodes.add(o)

    # 转换成id的形式，保存转换的字典
    n, r = list(nodes), list(rels)
    n2i, r2i = {n: i for i, n in enumerate(nodes)}, {r: i for i, r in enumerate(rels)}

    # id形式的all_triples、train_triples、test_triples
    all_triples = set()
    for s, p, w, t, o in train + val + test:
        all_triples.add((n2i[s], r2i[p], int(w), float(t), n2i[o]))
    train = [[n2i[s], r2i[p], int(w), float(t), n2i[o]] for s, p, w, t, o in train]
    val = [[n2i[s], r2i[p], int(w), float(t), n2i[o]] for s, p, w, t, o in val]
    test = [[n2i[s], r2i[p], int(w), float(t), n2i[o]] for s, p, w, t, o in test]

    return (n2i, list(n2i.values())), (r2i, list(r2i.values())), train, val, test, all_triples
