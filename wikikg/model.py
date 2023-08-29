from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

def tensor_constant(value, shape):
    """Create a tensor constant of the specified shape"""
    constant = np.ones(shape, np.float32) * value
    return torch.Tensor(constant)

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name in ['TransH']:
            self.norm_vector = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.norm_vector,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name in ['STransE']:
            self.W1 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim, self.hidden_dim))
            self.W2 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim, self.hidden_dim))
            nn.init.uniform_(
                tensor=self.W1,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.W2,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name in ['RotateCT']:
            self.b = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            nn.init.uniform_(
                tensor=self.b,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name in ['TransD']:
            self.entity_p_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_p_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.relation_p_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_p_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'PairRE', 'TransH', 'RotatEv2', 'STransE', 'RotateCT','TransD']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')

        self.evaluator = evaluator

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            sample, edge_reltype = sample
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part, edge_reltype = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)


            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
            if self.model_name == 'TransD':
                head_p = torch.index_select(
                    self.entity_p_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
    
                relation_p = torch.index_select(
                    self.relation_p_embedding,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)
    
                tail_p = torch.index_select(
                    self.entity_p_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part, edge_reltype = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            if self.model_name == 'TransD':
                head_p = torch.index_select(
                    self.entity_p_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)
    
                relation_p = torch.index_select(
                    self.relation_p_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)
    
                tail_p = torch.index_select(
                    self.entity_p_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'STransE': self.STransE,
            'TransD': self.TransD,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'PairRE': self.PairRE,
            'TransH': self.TransH,
            'RotatEv2': self.RotatEv2,
            'RotateCT': self.RotateCT,
        }


        if self.model_name in model_func:
            if self.model_name == 'TransD':
                score = model_func['TransD'](head, relation, tail, head_p, relation_p, tail_p, mode, edge_reltype)
            else:
                score = model_func[self.model_name](head, relation, tail, mode, edge_reltype)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    @staticmethod
    def l2norm_op(x):
        ones_for_sum = tensor_constant(1, (x.shape[2], x.shape[2]))
        ones_for_sum.to('cuda')
        eps = tensor_constant(1e-12, x.shape)
        return x * torch.rsqrt(torch.matmul(torch.square(x), ones_for_sum) + eps)
    
    def TransD(self, head, relation, tail, head_p, relation_p, tail_p, mode, edge_reltype):
        head = head + torch.sum(head * head_p, -1, keepdim=True) * relation_p
        tail = tail + torch.sum(tail * tail_p, -1, keepdim=True) * relation_p

        head = self.l2norm_op(head)
        relation = self.l2norm_op(relation)
        tail = self.l2norm_op(tail)
        
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE(self, head, relation, tail, mode, edge_reltype):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def STransE(self, head, relation, tail, mode, edge_reltype):
        edge_reltype = edge_reltype.squeeze(1)
        if mode == 'head-batch':
            score = torch.bmm(head, self.W1[edge_reltype]) + (relation - torch.bmm(tail, self.W2[edge_reltype]))
        else:
            score = (torch.bmm(head, self.W1[edge_reltype]) + relation) - torch.bmm(tail, self.W2[edge_reltype])

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode, edge_reltype):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode, edge_reltype):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode, edge_reltype):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def RotateCT(self, head, relation, tail, mode, edge_reltype):
        pi = 3.14159265358979323846
        # edge_reltype = edge_reltype.squeeze(1)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_b = self.b[edge_reltype]/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_b = torch.cos(phase_b)
        im_b = torch.sin(phase_b)

        if mode == 'head-batch':
            re_score = re_relation * (re_tail-re_b) + im_relation * (im_tail-im_b)
            im_score = re_relation * (im_tail-im_b) - im_relation * (re_tail-re_b)
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = (re_head-re_b) * re_relation - (im_head-im_b) * im_relation
            im_score = (re_head-re_b) * im_relation + (im_head-im_b) * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def RotatEv2(self, head, relation, tail, mode, r_norm=None):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=2)
        im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=2)

        re_score_head = re_head * re_relation_head - im_head * im_relation_head
        im_score_head = re_head * im_relation_head + im_head * re_relation_head

        re_score_tail = re_tail * re_relation_tail - im_tail * im_relation_tail
        im_score_tail = re_tail * im_relation_tail + im_tail * re_relation_tail

        re_score = re_score_head - re_score_tail
        im_score = im_score_head - im_score_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def PairRE(self, head, relation, tail, mode, edge_reltype):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransH(self, head, relation, tail, mode, edge_reltype):
        def _transfer(e, norm):
            norm = F.normalize(norm, p = 2, dim = -1)
            if e.shape[0] != norm.shape[0]:
                e = e.view(-1, norm.shape[0], e.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                e = e - torch.sum(e * norm, -1, True) * norm
                return e.view(-1, e.shape[-1])
            else:
                return e - torch.sum(e * norm, -1, True) * norm

        r_norm = self.norm_vector(relation)
        h = _transfer(h, r_norm)
        t = _transfer(t, r_norm)

        """
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        """
        
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, edge_reltype, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample, edge_reltype), mode=mode)
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model((positive_sample, edge_reltype))
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False, edge_reltype=None):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                args,
                'head-batch',
                random_sampling,
                edge_reltype=edge_reltype
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                args,
                'tail-batch',
                random_sampling,
                edge_reltype=edge_reltype
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            t1 = datetime.datetime.now().microsecond
            t3 = time.mktime(datetime.datetime.now().timetuple())
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, edge_reltype, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample, edge_reltype), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0],
                                                'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            t2 = datetime.datetime.now().microsecond
            t4 = time.mktime(datetime.datetime.now().timetuple())
            strTime = 'funtion time use:%dms' % ((t4 - t3) * 1000 + (t2 - t1) / 1000)
            print (strTime)

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
