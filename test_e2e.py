import os
import sys
import numpy as np
import argparse
import torch
import random
import time
import datetime
from transformers import BertTokenizer, BertConfig
from fastNLP import TorchLoaderIter
import logging
import transformers
from models import E2EBertTKG
from dataset import E2EDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

def replace_batch_data(facts, head_or_tail, num_ent, device):
    # generate replaced batch data
    # the first one is the ground truth
    batch_size = facts.shape[0]
    replaced_data = np.repeat(np.copy(facts), num_ent+1, axis=0)
    if head_or_tail == 'heads':
        for i in range(batch_size):
            # replaced_data[i*num_ent+1: (i+1)*num_ent+1, 0] = np.arange(0, num_ent)
            replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1, 0] = np.arange(0, num_ent)
    else:
        for i in range(batch_size):
            replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1, 2] = np.arange(0, num_ent)

    # convert to tensor and move to the correct device
    heads = torch.tensor(replaced_data[:, 0]).long().to(device)
    rels = torch.tensor(replaced_data[:, 1]).long().to(device)
    tails = torch.tensor(replaced_data[:, 2]).long().to(device)
    timestamps = torch.tensor(replaced_data[:, 3]).float().to(device)
    # convert to days, hour, minites
    days = (replaced_data[:, 3] / 15) // 96 + 1
    days = torch.tensor(days).float().to(device)
    hours = (replaced_data[:, 3] % 1440) // 60
    mins = ((replaced_data[:, 3] % 1440) % 60) // 15
    hours = torch.tensor(hours).float().to(device)
    mins = torch.tensor(mins).float().to(device)
    return heads, rels, tails, timestamps, days, hours, mins, replaced_data


def validate_batch_and_save(model, dataset, args, val_or_test, logger, epoch=None):
    metrics = {'hit1': 0.0, 'hit3': 0.0, 'hit10': 0.0, 'hit50':0.0, 'hit100':0.0, 'hit200':0.0, 'hit500':0.0, 'hit1000':0.0,'mrr': 0.0, 'mr': 0}
    start_batch = 0
    end_batch = 0
    num_ent = dataset.num_ent
    l = len(dataset.tkg_data[val_or_test])

    # logger_hit1 = logging.getLogger()
    # logger_hit1.setLevel(logging.INFO)
    # if not args.kg_model_chkpnt:
    #     log_hit1_file = f'test_{args.model_save_name}_{args.model_index}_hit1.log'
    # else:
    #     log_hit1_file = f'test_{args.model_save_name}_{args.model_index}_desimple_hit1.log'
        
    # if (logger_hit1.hasHandlers()):
    #     logger_hit1.handlers.clear()
    # console = logging.StreamHandler()
    # logger_hit1.addHandler(console)
    # fileHandler = logging.FileHandler(os.path.join(args.model_save_dir, log_hit1_file), mode='w')
    # logger_hit1.addHandler(fileHandler)

    step = 0
    while end_batch < l:
        if start_batch + args.val_batch_size > l:
            end_batch = l
        else:
            end_batch += args.val_batch_size
        batch_facts = dataset.tkg_data[val_or_test][start_batch: end_batch]
        for head_or_tail in ['heads', 'tails']:
            heads, rels, tails, timestamps, days, hours, mins, replaced_data = \
                replace_batch_data(batch_facts, head_or_tail, num_ent, args.device)
            scores = model.val_or_test(heads, rels, tails, days, hours, mins)

            replaced_data = replaced_data.tolist()
            for i in range(len(replaced_data)//(1+num_ent)):
                checked_data = replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1]
                for idx, cd in enumerate(checked_data):
                    if tuple(cd) in dataset.all_data_as_tuples:
                        scores[i*(1+num_ent)+1+idx] = float('-inf')
            # import pdb; pdb.set_trace()
            # reshape the scores
            ranks = torch.ones(end_batch - start_batch)
            scores = scores.reshape(-1, 1+num_ent)
            targets = scores[:, 0].unsqueeze(1)
            targets = targets.repeat(1, 1+num_ent)
            ranks += torch.sum((scores > targets).float(), dim=1).cpu()
            # ranks[start_batch: end_batch] += torch.sum((scores[:, :] >= scores[:, 0]).float(), dim=1).cpu()
            """
            for j in range(args.val_batch_size):
                ranks[start_batch+j] += torch.sum((scores[j] >= scores[j, 0]).float()).cpu()
            """
            # log hit@1 quadruples
            # if not args.kg_model_chkpnt:
            #     if len(np.where(ranks == 1.0)[0]) != 0:
            #         for id in np.nditer(np.where(ranks == 1.0)):
            #             logger_hit1.info(f'{heads.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {rels.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {tails.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {int(timestamps.detach().clone().reshape(-1, num_ent+1)[id,0].item())}')
            # else:
            #     if len(np.where(ranks > 1.0)[0]) != 0:
            #         for id in np.nditer(np.where(ranks > 1.0)):
            #             logger_hit1.info(f'{heads.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {rels.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {tails.detach().clone().reshape(-1, num_ent+1)[id,0].item()} {int(timestamps.detach().clone().reshape(-1, num_ent+1)[id,0].item())} {ranks[id]}')

            metrics['mr'] += torch.sum(ranks)
            metrics['mrr'] += torch.sum(1.0 / ranks)
            metrics['hit1'] += torch.sum((ranks == 1.0).float())
            metrics['hit3'] += torch.sum((ranks <= 3.0).float())
            metrics['hit10'] += torch.sum((ranks <= 10.0).float())
            metrics['hit50'] += torch.sum((ranks <= 50.0).float())
            metrics['hit100'] += torch.sum((ranks <= 100.0).float())
            metrics['hit200'] += torch.sum((ranks <= 200.0).float())
            metrics['hit500'] += torch.sum((ranks <= 500.0).float())
            metrics['hit1000'] += torch.sum((ranks <= 500.0).float())

        start_batch = end_batch
        step += 1
        # if step % args.print_out_loss_steps == 0:
        #     logger.info(f'current {step} step, already validated {end_batch} samples')

    # normalize
    for k in metrics.keys():
        metrics[k] /= (2 * l)
    logger.info(f'{val_or_test} result:\n')
    logger.info(f'\tHit@1 = {metrics["hit1"]}')
    logger.info(f'\tHit@3 = {metrics["hit3"]}')
    logger.info(f'\tHit@10 = {metrics["hit10"]}')
    logger.info(f'\tHit@50 = {metrics["hit50"]}')
    logger.info(f'\tHit@100 = {metrics["hit100"]}')
    logger.info(f'\tHit@200 = {metrics["hit200"]}')
    logger.info(f'\tHit@500 = {metrics["hit500"]}')
    logger.info(f'\tHit@1000 = {metrics["hit1000"]}')
    logger.info(f'\tMR = {metrics["mr"]}')
    logger.info(f'\tMRR = {metrics["mrr"]}')

    return metrics['mrr']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../data/pretrain_data/e2e_data/')
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--model_save_dir', type=str, default='../model/ckpts_e2e_2/')
    parser.add_argument('--model_save_name', type=str)
    parser.add_argument('--print_out_loss_steps', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='../logs/e2e_2')
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--entity_dic_file', type=str, default='../data/pretrain_data/entities.txt',
                        help='each line is of form: entity in text, entity index')
    parser.add_argument('--relation_dic_file', type=str, default='../data/pretrain_data/relations.txt',
                        help='each line is of form: relation id in gdelt, relation index, relation in plain text')
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization and dataset shuffling")
    parser.add_argument('--neg_ratio', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--kg_model_chkpnt', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_save_dir):
        print('The model path '+args.model_save_dir + ' is not found...')
        return
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    # set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = f'test_{args.model_save_name}_{args.model_index}.log'
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)
    fileHandler = logging.FileHandler(os.path.join(args.model_save_dir, log_file), mode='w')
    logger.addHandler(fileHandler)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # print all the args configurations
    logger.info("****** CONFIGURATION ******")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info("\n")

    # prepare the dataset
    logger.info('****** Preparing the dataset ******')
    # load the entity and relation vocab length
    with open(args.entity_dic_file, 'r') as freader:
        lines = freader.readlines()
    ent_num = len(lines)

    with open(args.relation_dic_file, 'r') as freader:
        lines = freader.readlines()
    rel_num = len(lines)

    tokenizer = BertTokenizer.from_pretrained('../model/bert_origin/bert-base-uncased')
    config = BertConfig.from_pretrained('../model/bert_origin/bert-base-uncased')
    word_mask_index = tokenizer.mask_token_id
    dataset = E2EDataset(args.data_dir, word_mask_index, config.vocab_size, ent_num, rel_num,
                            args.seed, tokenizer, args.neg_ratio)
    logger.info(f'#test data = {len(dataset.tkg_data["test"])}')
    logger.info('\n')

    # prepare the model
    logger.info('****** Preparing the model ******')
    model_path = os.path.join(args.model_save_dir, f'model_{args.model_index}.bin')
    plm_state_dict = torch.load(model_path)
    pretrain_config = BertConfig.from_pretrained('../model/bert_origin/bert-base-uncased', type_vocab_size=3)
    plm_model = E2EBertTKG(pretrain_config, ent_num=5850, rel_num=238)
    plm_model.load_state_dict(plm_state_dict, strict=True)

    # load the pre-trained tkg checkpoint if exists
    if args.kg_model_chkpnt is not None:
        pretrained_model = torch.load(args.kg_model_chkpnt, map_location=torch.device('cpu')).module
        pretrained_model.eval()
        pretrained_dict = pretrained_model.state_dict()
        target_keys = ['ent_embeddingsh_static.weight', 'ent_embeddingst_static.weight', 'rel_embeddings_f.weight', 'rel_embeddings_i.weight']
        new_keys = ['ent_embs_h.weight','ent_embs_t.weight','rel_embs_f.weight', 'rel_embs_i.weight']
        for key,n_key in zip(target_keys, new_keys):
            pretrained_dict[n_key] = pretrained_dict.pop(key)
        # print(pretrained_dict.keys())
        model_dict = plm_model.state_dict()
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # print(model_dict.keys())
        plm_model.load_state_dict(model_dict)

    # import pdb; pdb.set_trace()
    plm_model = plm_model.to(args.device)
    logger.info('****** Start to test the model ******')
    plm_model.eval()
    # test_mrr = validate_and_save(model, dataset, args, 'test', logger)
    test_mrr = validate_batch_and_save(plm_model, dataset, args, 'test', logger)
    logger.info(f'All training and val/test has been finished, the best test mrr is {test_mrr}')

if __name__ == "__main__":
    main()