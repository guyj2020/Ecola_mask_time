from symbol import test_nocond
import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
# from more_itertools import split_after


class E2EBertTKG(BertForMaskedLM):
    config_class = BertConfig

    def __init__(self, config, ent_num, rel_num, time_num, se_prop=0.68, drop_out=0.4, ent_emb=None, rel_emb=None, tkg_model=None,
                 tkg_type='DE', dataset='GDELT', loss_lambda=0.1, ablation=0):
        # the last item in ent_emb and rel_emb is [MASK]
        super().__init__(config)
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.drop_out = drop_out
        self.loss_lambda = loss_lambda
        self.ent_lm_head = EntityMLMHead(config, ent_num)
        self.rel_lm_head = RelationMLMHead(config, rel_num)
        self.time_lm_head = TimeMLMHead(config, time_num)
        self.time_num = time_num
        self.bert_hidden_size = 512
        self.ablation = ablation
        self.dataset = dataset
        self.tkg_type = tkg_type

        if ent_emb is not None:
            self.ent_embeddings = nn.Embedding.from_pretrained(ent_emb, freeze=False)
        else:
            self.ent_embeddings = nn.Embedding(ent_num, 512)
            self.ent_embeddings.weight.requires_grad = True

        if rel_emb is not None:
            self.rel_embeddings = nn.Embedding.from_pretrained(rel_emb, freeze=False)
        else:
            self.rel_embeddings = nn.Embedding(rel_num, 512)
            self.rel_embeddings.weight.requires_grad = True

        self.static_emb_dim = int(se_prop * 512)
        self.temporal_emb_dim = 512 - self.static_emb_dim
        self.linear = nn.Linear(512, 768)
        self.linear1 = nn.Linear(self.temporal_emb_dim, 768, bias=False)
        self.linear2 = nn.Linear(512, 768)
        if self.ablation == 1:
            self.static_kg_2_bert_w = nn.Linear(self.static_emb_dim, self.bert_hidden_size, bias=False)

        if not tkg_model:  # init tkg model embeddings from scratch
            ###############   add the TKG code here   ###############
            self.ent_embs_h = nn.Embedding(self.ent_num, self.static_emb_dim)
            self.ent_embs_t = nn.Embedding(self.ent_num, self.static_emb_dim)
            self.rel_embs_f = nn.Embedding(self.rel_num, 512)
            self.rel_embs_i = nn.Embedding(self.rel_num, 512)

            nn.init.xavier_uniform_(self.ent_embs_h.weight)
            nn.init.xavier_uniform_(self.ent_embs_t.weight)
            nn.init.xavier_uniform_(self.rel_embs_f.weight)
            nn.init.xavier_uniform_(self.rel_embs_i.weight)
            # temporal embeddings for entities
            if self.tkg_type == 'DE':
                self.create_time_embeds()
                self.time_nonlinear = torch.sin
            elif self.tkg_type == 'UTEE':
                self.create_utee_time_embeds()
            # initialize weights and apply finial processing
            # self.post_init()
            self.init_weights()

        else:  # load pretrained tkg model embedding
            if self.tkg_type == 'DE':
                self.load_pretrained_tkg_embs(tkg_model)
            if self.tkg_type == 'UTEE':
                self.load_pretrained_tkg_embs(tkg_model)

    def create_utee_time_embeds(self, ):
        dim = self.temporal_emb_dim
        if self.dataset.startswith('GDELT'):
            t_min, t_max = 0, 45000
        elif self.dataset == 'DuEE':
            t_min, t_max = 20180101, 20220301
        elif self.dataset == 'Wiki':
            t_min, t_max = 0, 83
        elif self.dataset == 'Cron':
            t_min, t_max = 0, 1000
        self.freq = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.amps = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.phas = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        torch.nn.init.uniform_(self.freq.data, a=t_min, b=t_max)
        torch.nn.init.xavier_uniform_(self.amps.data)
        torch.nn.init.uniform_(self.phas.data, a=0, b=t_max)

    def load_pretrained_tkg_embs(self, tkg_model):
        # static embeddings for ents and rels
        self.ent_embs_h = nn.Embedding.from_pretrained(tkg_model.ent_embeddingsh_static.weight, freeze=False)
        self.ent_embs_t = nn.Embedding.from_pretrained(tkg_model.ent_embeddingst_static.weight, freeze=False)
        self.rel_embs_f = nn.Embedding.from_pretrained(tkg_model.rel_embeddings_f.weight, freeze=False)
        self.rel_embs_i = nn.Embedding.from_pretrained(tkg_model.rel_embeddings_i.weight, freeze=False)

        # temporal embeddings for entities
        if self.dataset.startswith('GDELT') or self.dataset == 'DuEE':
            # frequency embeddings
            self.day_freq_h = nn.Embedding.from_pretrained(tkg_model.d_freq_h.weight, freeze=False)
            self.day_freq_t = nn.Embedding.from_pretrained(tkg_model.d_freq_t.weight, freeze=False)
            self.hour_freq_h = nn.Embedding.from_pretrained(tkg_model.h_freq_h.weight, freeze=False)
            self.hour_freq_t = nn.Embedding.from_pretrained(tkg_model.h_freq_t.weight, freeze=False)
            self.min_freq_h = nn.Embedding.from_pretrained(tkg_model.m_freq_h.weight, freeze=False)
            self.min_freq_t = nn.Embedding.from_pretrained(tkg_model.m_freq_t.weight, freeze=False)

            # phi embeddings
            self.day_phi_h = nn.Embedding.from_pretrained(tkg_model.d_phi_h.weight, freeze=False)
            self.day_phi_t = nn.Embedding.from_pretrained(tkg_model.d_phi_t.weight, freeze=False)
            self.hour_phi_h = nn.Embedding.from_pretrained(tkg_model.h_phi_h.weight, freeze=False)
            self.hour_phi_t = nn.Embedding.from_pretrained(tkg_model.h_phi_t.weight, freeze=False)
            self.min_phi_h = nn.Embedding.from_pretrained(tkg_model.m_phi_h.weight, freeze=False)
            self.min_phi_t = nn.Embedding.from_pretrained(tkg_model.m_phi_t.weight, freeze=False)
            # amplitude embeddings
            self.day_amp_h = nn.Embedding.from_pretrained(tkg_model.d_amp_h.weight, freeze=False)
            self.day_amp_t = nn.Embedding.from_pretrained(tkg_model.d_amp_t.weight, freeze=False)
            self.hour_amp_h = nn.Embedding.from_pretrained(tkg_model.h_amp_h.weight, freeze=False)
            self.hour_amp_t = nn.Embedding.from_pretrained(tkg_model.h_amp_t.weight, freeze=False)
            self.min_amp_h = nn.Embedding.from_pretrained(tkg_model.m_amp_h.weight, freeze=False)
            self.min_amp_t = nn.Embedding.from_pretrained(tkg_model.m_amp_t.weight, freeze=False)

        elif self.dataset == 'Wiki' or self.dataset == 'TSQA' or self.dataset == 'Cron':

            self.day_freq_h = nn.Embedding.from_pretrained(tkg_model.freq_h.weight, freeze=False)
            self.day_freq_t = nn.Embedding.from_pretrained(tkg_model.freq_t.weight, freeze=False)
            self.day_phi_h = nn.Embedding.from_pretrained(tkg_model.phi_h.weight, freeze=False)
            self.day_phi_t = nn.Embedding.from_pretrained(tkg_model.phi_t.weight, freeze=False)
            self.day_amp_h = nn.Embedding.from_pretrained(tkg_model.amp_h.weight, freeze=False)
            self.day_amp_t = nn.Embedding.from_pretrained(tkg_model.amp_t.weight, freeze=False)

        self.time_nonlinear = torch.sin

    def extend_type_embeddings(self, token_type=4):
        self.bert.embeddings.token_type_embeddings = nn.Embedding(token_type, self.config.hidden_size,
                                                                  _weight=torch.zeros(
                                                                      (token_type, self.config.hidden_size)))

    def create_time_embeds(self):
        # frequency embeddings
        if self.dataset == 'Wiki' or self.dataset == 'TSQA' or self.dataset == 'Cron':
            self.freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.freq_h.weight)
            nn.init.xavier_uniform_(self.freq_t.weight)
            nn.init.xavier_uniform_(self.phi_h.weight)
            nn.init.xavier_uniform_(self.phi_t.weight)
            nn.init.xavier_uniform_(self.amp_h.weight)
            nn.init.xavier_uniform_(self.amp_t.weight)

        else:
            self.day_freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.day_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_freq_h.weight)
            nn.init.xavier_uniform_(self.day_freq_t.weight)
            nn.init.xavier_uniform_(self.hour_freq_h.weight)
            nn.init.xavier_uniform_(self.hour_freq_t.weight)
            nn.init.xavier_uniform_(self.min_freq_h.weight)
            nn.init.xavier_uniform_(self.min_freq_t.weight)

            # phi embeddings
            self.day_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.day_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_phi_h.weight)
            nn.init.xavier_uniform_(self.day_phi_t.weight)
            nn.init.xavier_uniform_(self.hour_phi_h.weight)
            nn.init.xavier_uniform_(self.hour_phi_t.weight)
            nn.init.xavier_uniform_(self.min_phi_h.weight)
            nn.init.xavier_uniform_(self.min_phi_t.weight)

            # amplitude embeddings
            self.day_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.day_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_amp_h.weight)
            nn.init.xavier_uniform_(self.day_amp_t.weight)
            nn.init.xavier_uniform_(self.hour_amp_h.weight)
            nn.init.xavier_uniform_(self.hour_amp_t.weight)
            nn.init.xavier_uniform_(self.min_amp_h.weight)
            nn.init.xavier_uniform_(self.min_amp_t.weight)

    def get_utee_time_embedd(self, timestamps):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.freq
        feat = self.amps * torch.sin(timestamps * omega + self.phas)
        return feat

    def get_time_embedd(self, entities, days, hours, mins, head_or_tail):
        # dataset gdelt/wiki
        if self.dataset.startswith('GDELT') or self.dataset == 'DuEE':  # dataset Gdelt/DuEE
            if head_or_tail == 'head':
                d = self.day_amp_h(entities) * self.time_nonlinear(
                    self.day_freq_h(entities) * days + self.day_phi_h(entities))
                h = self.hour_amp_h(entities) * self.time_nonlinear(
                    self.hour_freq_h(entities) * hours + self.hour_phi_h(entities))
                m = self.min_amp_h(entities) * self.time_nonlinear(
                    self.min_freq_h(entities) * mins + self.min_phi_h(entities))
            else:
                d = self.day_amp_t(entities) * self.time_nonlinear(
                    self.day_freq_t(entities) * days + self.day_phi_t(entities))
                h = self.hour_amp_t(entities) * self.time_nonlinear(
                    self.hour_freq_t(entities) * hours + self.hour_phi_t(entities))
                m = self.min_amp_t(entities) * self.time_nonlinear(
                    self.min_freq_t(entities) * mins + self.min_phi_t(entities))
            return d + h + m

        elif self.dataset == 'Wiki' or self.dataset == 'TSQA' or self.dataset == 'Cron':
            if head_or_tail == 'head':
                d = self.amp_h(entities) * self.time_nonlinear(self.freq_h(entities) * days + self.phi_h(entities))
            else:
                d = self.amp_t(entities) * self.time_nonlinear(self.freq_t(entities) * days + self.phi_t(entities))
            return d

    # default e2e2 setting
    def get_tkg_added_ent_embs(self, ent_ids, h_embs1, t_embs1):
        # import pdb; pdb.set_trace()
        try:
            bert_h = self.ent_embeddings(ent_ids[:, 0])
        except:
            bert_h = self.bert.embeddings.word_embeddings(ent_ids[:, 0])
        try:
            bert_t = self.ent_embeddings(ent_ids[:, 1])
        except:
            bert_t = self.bert.embeddings.word_embeddings(ent_ids[:1])
        # import pdb;pdb.set_trace()
        if self.ablation >= 3:
            entity_embeddings = torch.cat((h_embs1.unsqueeze(1), t_embs1.unsqueeze(1)), dim=1)
        else:
            entity_embeddings = torch.cat(((bert_h + h_embs1).unsqueeze(1), (bert_t + t_embs1).unsqueeze(1)), dim=1)
        return entity_embeddings

    def get_tkg_added_rel_embs(self, rel_ids, r_embs1, r_embs2):
        # import pdb; pdb.set_trace()
        try:
            bert_r = self.rel_embeddings(rel_ids)
        except:
            bert_r = self.bert.embeddings.word_embeddings(rel_ids)
        # import pdb;pdb.set_trace()
        if self.ablation >= 3:
            return (r_embs1 + r_embs2) / 2
        else:
            return bert_r + (r_embs1 + r_embs2) / 2




    # ablation 1 setting: only add tkg static part to bert embs
    def get_tkgs_added_ent_embs(self, ent_ids, h_embs1, t_embs1):
        # import pdb; pdb.set_trace()
        try:
            bert_h = self.ent_embeddings(ent_ids[:, 0])
        except:
            bert_h = self.bert.embeddings.word_embeddings(ent_ids[:, 0])
        try:
            bert_t = self.ent_embeddings(ent_ids[:, 1])
        except:
            bert_t = self.bert.embeddings.word_embeddings(ent_ids[:1])
        entity_embeddings = torch.cat(((bert_h + h_embs1).unsqueeze(1), (bert_t + t_embs1).unsqueeze(1)), dim=1)
        return entity_embeddings

    def get_tkgs_added_rel_embs(self, rel_ids, r_embs1, r_embs2):
        # import pdb; pdb.set_trace()
        try:
            bert_r = self.rel_embeddings(rel_ids)
        except:
            bert_r = self.bert.embeddings.word_embeddings(rel_ids)
        return bert_r + (r_embs1 + r_embs2) / 2

    def get_tkg_static_Embeddings(self, heads, rels, tails, intervals=None):
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_tkg_Embeddings(self, heads, rels, tails, years, months, days, intervals=None):
        # years = years.view(-1,1)
        # months = months.view(-1,1)
        # days = days.view(-1,1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_utee_tkg_Embeddings(self, heads, rels, tails, timestamps):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_static_Embeddings(heads, rels, tails)
        h_embs1 = torch.cat((h_embs1, self.get_utee_time_embedd(timestamps)), 1)
        t_embs1 = torch.cat((t_embs1, self.get_utee_time_embedd(timestamps)), 1)
        h_embs2 = torch.cat((h_embs2, self.get_utee_time_embedd(timestamps)), 1)
        t_embs2 = torch.cat((t_embs2, self.get_utee_time_embedd(timestamps)), 1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_tuple_time(self, tuple_t):
        if self.dataset == 'DuEE':  # YYYYMMDD
            days = tuple_t // 10000  # year
            days = days.float()
            hours = (tuple_t % 10000) // 100  # month
            hours = hours.float()
            mins = (tuple_t % 10000) % 100  # day
            mins = mins.float()

        elif self.dataset.startswith('GDELT'):
            days = (tuple_t / 15) // 96 + 1
            days = days.float()
            hours = (tuple_t % 1440) // 60
            mins = ((tuple_t % 1440) % 60) // 15
            hours = hours.float()
            mins = mins.float()

        elif self.dataset == 'Wiki' or self.dataset == 'TSQA' or self.dataset == 'Cron':
            days = tuple_t.float()
            hours, mins = None, None

        days = days.view(-1, 1)
        if hours != None and mins != None:
            hours = hours.view(-1, 1)
            mins = mins.view(-1, 1)
        return days, hours, mins

    def get_kepler_embs(self, input_ids, timestamps):
        input_ids_spl = [list(split_after(sample, lambda x: x == 102)) for sample in input_ids]
        batch_size = input_ids.shape[0]
        h_ent_ids = [input_ids_spl[x][0] for x in range(batch_size)]
        t_ent_ids = [input_ids_spl[x][2] for x in range(batch_size)]
        maxlen_h = len(max(h_ent_ids, key=len))
        maxlen_t = len(max(t_ent_ids, key=len))
        exe = [head.extend([103] * (maxlen_h - len(head))) for head in h_ent_ids]
        exe = [tail.extend([103] * (maxlen_t - len(tail))) for tail in t_ent_ids]
        t_ent_ids = [[101] + tail for tail in t_ent_ids]
        h_ent_ids = torch.LongTensor(h_ent_ids).cuda()
        t_ent_ids = torch.LongTensor(t_ent_ids).cuda()
        # t_ent_ids = torch.concat([torch.full((batch_size, 1), 101, dtype=torch.LongTensor), t_ent_ids], dim=1)
        h_embs = self.bert.embeddings.word_embeddings(h_ent_ids)
        t_embs = self.bert.embeddings.word_embeddings(t_ent_ids)
        h_embs = torch.cat((h_embs, self.get_utee_time_embedd(timestamps)), 1)
        t_embs = torch.cat((t_embs, self.get_utee_time_embedd(timestamps)), 1)
        r_embs = self.rel_embeddings(input_ids[:, -1])
        return h_embs[:, 0, :], r_embs, t_embs[:, 0, :]

    def forward(self, input_ids=None, num_tokens=None, attention_mask=None, token_type_ids=None, inputs_embeds=None,
                word_masked_lm_labels=None, entity_masked_lm_labels=None, relation_masked_lm_labels=None, time_masked_lm_labels=None,
                tkg_tuple=None,
                tuple_labels=None):
        loss_fct = CrossEntropyLoss()
        batch_size = input_ids.shape[0]
        # DE-SimplE part
        # get heads, rels, tails, timestamps(convert to min, hour, day) from tkg_tuple
        tkg_tuple = tkg_tuple.view(-1, 4)
        heads = tkg_tuple[:, 0].long()
        rels = tkg_tuple[:, 1].long()
        tails = tkg_tuple[:, 2].long()
        days, hours, mins = self.get_tuple_time(tkg_tuple[:, 3])

        if self.ablation == 10:
            h_embs, r_embs, t_embs = self.get_kepler_embs(input_ids, tkg_tuple[:, 3].long())
            tkg_scores = (h_embs * r_embs) * t_embs
        else:
            # get embeddings
            if self.tkg_type == 'DE':
                h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_Embeddings(heads, rels, tails, days,
                                                                                               hours, mins)
            elif self.tkg_type == 'UTEE':
                h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_utee_tkg_Embeddings(heads, rels, tails,
                                                                                                    (tkg_tuple[:, 3]-1500).long())
            # get tkg scores
            tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        # get tkg loss
        num_examples = tuple_labels.shape[0]
        tkg_scores = tkg_scores.reshape(2 * num_examples, -1)
        tuple_labels = tuple_labels.reshape(-1).long()
        tkg_loss = loss_fct(tkg_scores, tuple_labels)

        # bert MLM part
        num_word_tokens = num_tokens[0] - 4
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids[:, :num_word_tokens])

        # ablation whether to use static part only
        if self.ablation == 0 or self.ablation == 2 or self.ablation >= 3 and self.ablation <= 9:
            entity_embeddings = self.get_tkg_added_ent_embs(input_ids[:, num_word_tokens: num_word_tokens + 2], \
                                                            h_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0],
                                                            t_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0])
            relation_embeddings = self.get_tkg_added_rel_embs(input_ids[:, -1], \
                                                              r_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0],
                                                              r_embs2.view(batch_size, -1, self.bert_hidden_size)[:,
                                                              0]).unsqueeze(1)
        elif self.ablation == 11:
            # import pdb; pdb.set_trace()
            entity_embeddings = self.get_tkg_added_ent_embs(input_ids[:, num_word_tokens: num_word_tokens + 2], \
                                                            h_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0],
                                                            t_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0])
            entity_embeddings = self.linear(entity_embeddings)

            relation_embeddings = self.get_tkg_added_rel_embs(input_ids[:, -2], \
                                                              r_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0],
                                                              r_embs2.view(batch_size, -1, self.bert_hidden_size)[:,
                                                              0]).unsqueeze(1)
            relation_embeddings = self.linear2(relation_embeddings)
            time_embeddings = self.get_utee_time_embedd(input_ids[:, -1]).unsqueeze(1)
            # time_embeddings = time_embeddings.view(batch_size, -1, self.temporal_emb_dim)[:, 0].unsqueeze(1)

            time_embeddings = self.linear1(time_embeddings)


        elif self.ablation == 1:
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_static_Embeddings(heads, rels, tails)
            h_embs1 = self.static_kg_2_bert_w(h_embs1)
            t_embs1 = self.static_kg_2_bert_w(t_embs1)
            entity_embeddings = self.get_tkgs_added_ent_embs(input_ids[:, num_word_tokens: num_word_tokens + 2], \
                                                             h_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0],
                                                             t_embs1.view(batch_size, -1, self.bert_hidden_size)[:, 0])
            relation_embeddings = self.get_tkgs_added_rel_embs(input_ids[:, -1], \
                                                               r_embs1.view(batch_size, -1, self.bert_hidden_size)[:,
                                                               0],
                                                               r_embs2.view(batch_size, -1, self.bert_hidden_size)[:,
                                                               0]).unsqueeze(1)
        elif self.ablation == 10:
            entity_embeddings = torch.concat(h_embs, t_embs)
            relation_embeddings = r_embs
        # entity_embeddings = self.ent_embeddings(input_ids[:, num_word_tokens: num_word_tokens+2])
        # relation_embeddings = self.rel_embeddings(input_ids[:, -1]).unsqueeze(1)

        # MLM loss
        # concat the 3 parts of embeddings
        inputs_embeds = torch.cat([word_embeddings, entity_embeddings, relation_embeddings, time_embeddings], dim=1)
        # import pdb; pdb.set_trace()
        outputs_mlm = self.bert(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                inputs_embeds=inputs_embeds)
        sequence_output = outputs_mlm[0]
        # except Exception as e:
        #     print('input_embeds:', inputs_embeds)
        #     print(e)
        #     print('len_word: ', len(word_embeddings[0]), 'len_ent: ',len(entity_embeddings[0]), 'len_rel: ', len(relation_embeddings[0]))

        word_prediction_scores = self.cls(sequence_output)
        word_prediction = torch.argmax(word_prediction_scores[:, :num_word_tokens, :], dim=-1)
        word_lm_loss = loss_fct(word_prediction_scores.view(-1, self.config.vocab_size), word_masked_lm_labels.view(-1))

        entity_prediction_scores = self.ent_lm_head(sequence_output)
        entity_prediction = torch.argmax(entity_prediction_scores[:, num_word_tokens:num_word_tokens + 2, :], dim=-1)
        entity_lm_loss = loss_fct(entity_prediction_scores.view(-1, self.ent_num), entity_masked_lm_labels.view(-1))

        relation_prediction_scores = self.rel_lm_head(sequence_output)
        relation_prediction = torch.argmax(relation_prediction_scores[:, -2, :], dim=-1).unsqueeze(1)
        # import pdb;
        # pdb.set_trace()
        relation_lm_loss = loss_fct(relation_prediction_scores.view(-1, self.rel_num), relation_masked_lm_labels.view(-1))

        time_prediction_scores = self.time_lm_head(sequence_output)
        time_prediction = torch.argmax(time_prediction_scores[:, -1, :], dim=-1).unsqueeze(1)
        time_lm_loss = loss_fct(time_prediction_scores.view(-1, self.time_num),
                                    time_masked_lm_labels.view(-1))

        if self.ablation <= 3:
            mlm_loss = word_lm_loss + entity_lm_loss + relation_lm_loss
        if self.ablation == 11:
            mlm_loss = word_lm_loss + entity_lm_loss + relation_lm_loss + time_lm_loss
        elif self.ablation == 4 or self.ablation == 10:
            mlm_loss = word_lm_loss
        elif self.ablation == 5:
            mlm_loss = entity_lm_loss
        elif self.ablation == 6:
            mlm_loss = relation_lm_loss
        elif self.ablation == 7:
            mlm_loss = word_lm_loss + relation_lm_loss
        elif self.ablation == 8:
            mlm_loss = word_lm_loss + entity_lm_loss
        elif self.ablation == 9:
            mlm_loss = entity_lm_loss + relation_lm_loss

        total_loss = self.loss_lambda * mlm_loss + tkg_loss
        # return {'total_loss': total_loss, 'mlm_loss': mlm_loss, 'tkg_loss': tkg_loss,
        #         'word_pred': word_prediction, 'ent_pred': entity_prediction, 'rel_pred': relation_prediction}

        return {'total_loss': total_loss, 'mlm_loss': mlm_loss, 'tkg_loss': tkg_loss, 'word_lm_loss':word_lm_loss, 'entity_lm_loss': entity_lm_loss, 'relation_lm_loss': relation_lm_loss, 'time_lm_loss':time_lm_loss,
                'word_pred': word_prediction, 'ent_pred': entity_prediction, 'rel_pred': relation_prediction,  'time_pred': time_prediction}

    def val_or_test(self, heads, rels, tails, days, hours, mins):
        # in this function, just use the tkg model to do test, and just compute the score
        # DE-SimplE part
        # get embeddings
        days = days.view(-1, 1)
        if hours != None and mins != None:
            hours = hours.view(-1, 1)
            mins = mins.view(-1, 1)
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_Embeddings(heads, rels, tails, days, hours,
                                                                                       mins)
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        return tkg_scores

    def val_or_test_utee(self, heads, rels, tails, timestamps):
        # in this function, just use the tkg model to do test, and just compute the score
        # UTEE part
        # get embeddings
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_utee_tkg_Embeddings(heads, rels, tails,
                                                                                            timestamps)
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        return tkg_scores


class EntityMLMHead(nn.Module):
    def __init__(self, config, ent_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, ent_num, bias=False)
        self.bias = nn.Parameter(torch.zeros(ent_num), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h


class RelationMLMHead(nn.Module):
    def __init__(self, config, rel_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, rel_num, bias=False)
        self.bias = nn.Parameter(torch.zeros(rel_num), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h


class TimeMLMHead(nn.Module):
    def __init__(self, config, time_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, time_num, bias=False)
        self.bias = nn.Parameter(torch.zeros(time_num), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h


class DE_SimplE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_SimplE, self).__init__()
        self.dataset = dataset
        self.params = params

        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.s_emb_dim + params.t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.s_emb_dim + params.t_emb_dim).cuda()

        self.create_time_embedds()

        self.time_nl = torch.sin

        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)

    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days + self.d_phi_h(entities))
        else:
            emb = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days + self.d_phi_t(entities))

        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals=None):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def forward(self, heads, rels, tails, years, months, days):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(heads, rels, tails, years, months,
                                                                                  days)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
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
        elif self.dataset == 'Cron':
            t_min, t_max = 0, 1000
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