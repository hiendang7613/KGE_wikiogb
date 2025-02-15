#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from collections import defaultdict
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import os.path as osp

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--dataset', type=str, default='ogbl-wikikg', help='dataset name, default to wikikg')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')
    parser.add_argument('--relation_type', type=str, default='all', help='1-1, 1-n, n-1, n-n')
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    print(log_file)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        print('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        writer.add_scalar("_".join([mode, metric]), metrics[metric], step)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    args.save_path = 'log/%s/%s/%s-%s/%s'%(args.dataset, args.model, args.hidden_dim, args.gamma, time.time()) if args.save_path == None else args.save_path
    writer = SummaryWriter(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    dataset = LinkPropPredDataset(name = args.dataset)

    split_dict = dataset.get_edge_split()
    nentity = dataset.graph['num_nodes']
    nrelation = int(max(dataset.graph['edge_reltype'])[0])+1

    evaluator = Evaluator(name = args.dataset)

    args.nentity = nentity
    args.nrelation = nrelation

    print('Model: %s' % args.model)
    print('Dataset: %s' % args.dataset)
    print('#entity: %d' % nentity)
    print('#relation: %d' % nrelation)

    train_triples = split_dict['train']
    print('#train: %d' % len(train_triples['head']))
    valid_triples = split_dict['valid']
    print('#valid: %d' % len(valid_triples['head']))
    test_triples = split_dict['test']
    print('#test: %d' % len(test_triples['head']))
    print('relation type %s' % args.relation_type)
    print('relation type %s' % args.relation_type)
    test_set_file = ''
    if args.relation_type == '1-1':
        test_set_file = './dataset/ogbl_wikikg/wikikg_P/1-1-id.txt'
        test_set_pre_processed = './dataset/ogbl_wikikg/wikikg_P/1-1.pt'
    elif args.relation_type == '1-n':
        test_set_file = './dataset/ogbl_wikikg/wikikg_P/1-n-id.txt'
        test_set_pre_processed = './dataset/ogbl_wikikg/wikikg_P/1-n.pt'
    elif args.relation_type == 'n-1':
        test_set_file = './dataset/ogbl_wikikg/wikikg_P/n-1-id.txt'
        test_set_pre_processed = './dataset/ogbl_wikikg/wikikg_P/n-1.pt'
    elif args.relation_type == 'n-n':
        test_set_file = './dataset/ogbl_wikikg/wikikg_P/n-n-id.txt'
        test_set_pre_processed = './dataset/ogbl_wikikg/wikikg_P/n-n.pt'

    if test_set_file != '':
        if osp.exists(test_set_pre_processed):
            test_triples = torch.load(test_set_pre_processed, 'rb')
            print("load pre processed test set")
        else:
            test_triples_new = {}
            test_triples_chosen = []
            test_triples_new['head'] = []
            test_triples_new['relation'] = []
            test_triples_new['tail'] = []
            test_triples_new['head_neg'] = []
            test_triples_new['tail_neg'] = []
            f_test = open(test_set_file, "r")
            for line in f_test:
                h, r, t = line.strip().split('\t')
                h, r, t = int(h), int(r), int(t)
                test_triples_chosen.append((h, r, t))
            f_test.close()

            for idx in range(len(test_triples['head'])):
                h, r, t = test_triples['head'][idx], test_triples['relation'][idx], test_triples['tail'][idx]
                if (h, r, t) in test_triples_chosen:
                    test_triples_new['head'].append(h)
                    test_triples_new['relation'].append(r)
                    test_triples_new['tail'].append(t)
                    test_triples_new['head_neg'].append(test_triples['head_neg'][idx])
                    test_triples_new['tail_neg'].append(test_triples['tail_neg'][idx])
            print('Saving ...')
            torch.save(test_triples_new, test_set_pre_processed, pickle_protocol=4)

            test_triples = test_triples_new
            print('#test: %d' % len(test_triples['head']))


    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    f_train = open("train.txt", "w")
    for i in tqdm(range(len(train_triples['head']))):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        train_count[(head, relation)] += 1
        train_count[(tail, -relation-1)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)
        f_train.write("\t".join([str(head), str(relation), str(tail)]) + '\n')
    f_train.close()


    """
    f_train = open("valid.txt", "w")
    for i in tqdm(range(len(valid_triples['head']))):
        head, relation, tail = valid_triples['head'][i], valid_triples['relation'][i], valid_triples['tail'][i]
        f_train.write("\t".join([str(head), str(relation), str(tail)]) + '\n')
    f_train.close()

    f_train = open("test.txt", "w")
    for i in tqdm(range(len(test_triples['head']))):
        head, relation, tail = test_triples['head'][i], test_triples['relation'][i], test_triples['tail'][i]
        f_train.write("\t".join([str(head), str(relation), str(tail)]) + '\n')
    f_train.close()
    """

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        evaluator=evaluator
    )

    print('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        print('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail, dataset.graph['edge_reltype']),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'tail-batch',
                train_count, train_true_head, train_true_tail, dataset.graph['edge_reltype']),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        print('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    print('Start Training...')
    print('init_step = %d' % init_step)
    print('batch_size = %d' % args.batch_size)
    print('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    print('hidden_dim = %d' % args.hidden_dim)
    print('gamma = %f' % args.gamma)
    print('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        print('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        print('learning_rate = %d' % current_learning_rate)

        training_logs = []

        #Training Loop
        for step in tqdm(range(init_step, args.max_steps)):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0 and step > 0: # ~ 41 seconds/saving
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Train', step, metrics, writer)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                print('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, args, edge_reltype=dataset.graph['edge_reltype'])
                log_metrics('Valid', step, metrics, writer)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        print('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, args, edge_reltype=dataset.graph['edge_reltype'])
        log_metrics('Valid', step, metrics, writer)

    if args.do_test:
        print('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, args, edge_reltype=dataset.graph['edge_reltype'])
        log_metrics('Test', step, metrics, writer)
        print(metrics)
    if args.evaluate_train:
        print('Evaluating on Training Dataset...')
        small_train_triples = {}
        indices = np.random.choice(len(train_triples['head']), args.ntriples_eval_train, replace=False)
        for i in train_triples:
            small_train_triples[i] = train_triples[i][indices]
        metrics = kge_model.test_step(kge_model, small_train_triples, args, random_sampling=True)
        log_metrics('Train', step, metrics, writer)

if __name__ == '__main__':
    main(parse_args())
