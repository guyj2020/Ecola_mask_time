import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class TransE(torch.nn.Module):
    def __init__(self, ent_num, rel_num, emb_dim, drop_out):
        super(TransE, self).__init__()
        self.ent_embeds = nn.Embedding(ent_num, emb_dim)
        self.rel_embeds = nn.Embedding(rel_num, emb_dim)
        self.drop_out = drop_out

    def forward(self, heads, rels, tails):
        h_embs = self.ent_embeds(heads)
        r_embs = self.rel_embeds(rels)
        t_embs = self.ent_embeds(tails)
        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.drop_out, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores

class DistMult(torch.nn.Module):
    def __init__(self, ent_num, rel_num, emb_dim, drop_out):
        super(DistMult, self).__init__()
        self.ent_embeds = nn.Embedding(ent_num, emb_dim)
        self.rel_embeds = nn.Embedding(rel_num, emb_dim)
        self.drop_out = drop_out

    def forward(self, heads, rels, tails):
        h_embs = self.ent_embeds(heads)
        r_embs = self.rel_embeds(rels)
        t_embs = self.ent_embeds(tails)
        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.drop_out, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores

class SimplE(torch.nn.Module):
    def __init__(self, ent_num, rel_num, emb_dim, drop_out):
        super(SimplE, self).__init__()
        self.ent_embeds_head = nn.Embedding(self.ent_num, emb_dim)
        self.ent_embeds_tail = nn.Embedding(self.ent_num, emb_dim)
        self.rel_embeds_for = nn.Embedding(self.rel_num, emb_dim)
        self.rel_embeds_inv = nn.Embedding(self.rel_num, emb_dim)
        self.drop_out = drop_out

    def forward(self, heads, rels, tails):
        h_embeds_1 = self.ent_embeds_head(heads)
        t_embeds_1 = self.ent_embeds_tail(tails)
        r_embeds_1 = self.rel_embeds_for(rels)
        h_embeds_2 = self.ent_embeds_tail(heads)
        t_embeds_2 = self.ent_embeds_head(tails)
        r_embeds_2 = self.rel_embeds_inv(rels)

        scores = ((h_embeds_1 * r_embeds_1) * t_embeds_1 + (h_embeds_2 * r_embeds_2) * t_embeds_2) / 2.0
        scores = F.dropout(scores, p=self.drop_out, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores

# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DE_SimplE(torch.nn.Module):
    def __init__(self, ent_num, rel_num, emb_dim, drop_out, se_prop=0.68):
        super(DE_SimplE, self).__init__()
        self.s_emb_dim = int(emb_dim * se_prop)
        self.t_emb_dim = emb_dim - self.s_emb_dim
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.drop_out = drop_out

        
        self.ent_embs_h = nn.Embedding(self.ent_num, self.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(self.ent_num, self.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(self.rel_num, self.s_emb_dim + self.t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(self.rel_num, self.s_emb_dim + self.t_emb_dim).cuda()
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
    
    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.freq_h = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()
        self.freq_t = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()

        # phi embeddings for the entities
        self.phi_h = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()
        self.phi_t = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()

        # frequency embeddings for the entities
        self.amps_h = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()
        self.amps_t = nn.Embedding(self.ent_num, self.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.freq_h.weight)
        nn.init.xavier_uniform_(self.freq_t.weight)

        nn.init.xavier_uniform_(self.phi_h.weight)
        nn.init.xavier_uniform_(self.phi_t.weight)

        nn.init.xavier_uniform_(self.amps_h.weight)
        nn.init.xavier_uniform_(self.amps_t.weight)

    def get_time_embedd(self, entities, timestamps, h_or_t):
        if h_or_t == "head":
            emb  = self.amps_h(entities) * self.time_nl(self.freq_h(entities) * timestamps  + self.phi_h(entities))
        else:
            emb  = self.amps_t(entities) * self.time_nl(self.freq_t(entities) * timestamps  + self.phi_t(entities))
        return emb

    def getEmbeddings(self, heads, rels, tails, timestamps, intervals = None):
        timestamps = timestamps.view(-1,1)

        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, timestamps, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, timestamps, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, timestamps, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, timestamps, "tail")), 1)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
    
    def forward(self, heads, rels, tails, timestamps):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(heads, rels, tails, timestamps)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.drop_out, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
        
class UTEE(torch.nn.Module):
    def __init__(self, ent_num, rel_num, emb_dim, drop_out, dataset='wiki', se_prop=0.68):
        super(UTEE, self).__init__()
        self.s_emb_dim = int(emb_dim * se_prop)
        self.t_emb_dim = emb_dim - self.s_emb_dim
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.drop_out = drop_out
        self.dataset = dataset

        self.ent_embs_h = nn.Embedding(self.ent_num, self.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(self.ent_num, self.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(self.rel_num, self.s_emb_dim + self.t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(self.rel_num, self.s_emb_dim + self.t_emb_dim).cuda()

        self.create_utee_time_embeds()

    def create_utee_time_embeds(self, ):
        dim = self.t_emb_dim
        if self.dataset == 'gdelt':
            t_min, t_max = 0, 45000
        elif self.dataset == 'duee':
            t_min, t_max = 20180101, 20220301
        elif self.dataset == 'wiki':
            t_min, t_max = 0, 83
        self.freq = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.amps = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.phas = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        torch.nn.init.uniform_(self.freq.data, a=t_min, b=t_max)
        torch.nn.init.xavier_uniform_(self.amps.data)
        torch.nn.init.uniform_(self.phas.data, a=0, b=t_max)

    def get_tkg_static_Embeddings(self, heads, rels, tails, intervals = None):
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_utee_time_embedd(self, timestamps):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.freq
        feat = self.amps * torch.sin(timestamps * omega + self.phas)
        return feat

    def get_utee_tkg_Embeddings(self, heads, rels, tails, timestamps):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_static_Embeddings(heads, rels, tails)
        h_embs1 = torch.cat((h_embs1, self.get_utee_time_embedd(timestamps)), 1)
        t_embs1 = torch.cat((t_embs1, self.get_utee_time_embedd(timestamps)), 1)
        h_embs2 = torch.cat((h_embs2, self.get_utee_time_embedd(timestamps)), 1)
        t_embs2 = torch.cat((t_embs2, self.get_utee_time_embedd(timestamps)), 1)
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def forward(self, heads, rels, tails, timestamps):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_utee_tkg_Embeddings(heads, rels, tails, timestamps)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.drop_out, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores