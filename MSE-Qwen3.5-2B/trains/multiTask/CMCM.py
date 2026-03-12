import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
# [MOD] 删除旧 AMP 导入
# from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain

logger = logging.getLogger('MSA')


class CMCM():
    def __init__(self, args):

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

        # [MOD] 保存标签映射，供分类评测时把生成结果转成数字 id
        self.label2id = getattr(args, "label_index_mapping", None)

    # [MOD] 新增：把 generate 的文本输出解析成分类 id
    # 目标是尽量兼容这些形式：
    # "4"
    # " 4"
    # "label is 4"
    # "The label is 4"
    # 若解析失败，默认回退为 0
    def _parse_cls_output(self, s):
        s = str(s).strip()

        # 优先找单个数字字符
        for ch in s:
            if ch.isdigit():
                return int(ch)

        # 兜底：如果输出是标签名，也尝试映射
        if self.label2id is not None:
            low = s.lower()
            for k, v in self.label2id.items():
                if str(k).lower() in low:
                    return int(v)

        return 0

    def do_train(self, model, dataloader):

        # [MOD] 新 AMP 写法，消除 FutureWarning
        amp_enabled = torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        optimizer = optim.AdamW(model.Model.parameters(), lr=self.args.learning_rate, eps=1e-4)
        total_steps = len(dataloader['train']) * self.args.warm_up_epochs   # 大致的一个训练step数

        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
        #                               threshold=0.0001, eps=1e-08)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),   # [MOD] 显式转 int，更稳
            num_training_steps=total_steps
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        losses = []

        CPC_Losses = []
        # valid_F1 = []
        lr = []
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            CPC_Loss_sum = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)

                    if self.args.train_mode == 'regression':
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        prefix_label = batch_data['labels_prefix']
                        cur_id = batch_data['id']
                        ids.extend(cur_id)
                    else:
                        labels_m = batch_data['labels']['M']

                    indexes = batch_data['index'].view(-1)

                    if not self.args.need_data_aligned:
                        text_lengths = batch_data['text_lengths'].to(self.args.device)
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        # [MOD] 对齐模式下兜底，避免变量未定义
                        text_lengths = None
                        audio_lengths = None
                        vision_lengths = None

                    # forward
                    # [MOD] 新 AMP 写法
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        output = model(labels_m, (text, text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                        loss = output['Loss']

                    # backward
                    scaler.scale(loss).backward()
                    train_loss += loss.item()
                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])

                    # update parameters
                    if not left_epochs:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    scaler.step(optimizer)
                    scaler.update()

            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f" % (
                self.args.modelName,
                epochs - best_epoch,
                epochs,
                self.args.cur_time,
                train_loss
            ))
            losses.append(train_loss)

            # validation
            if epochs >= 1:
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]

                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' \
                    else cur_valid >= (best_valid + 1e-6)

                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    self.save_model(model, epochs, self.args.model_save_path)
                    model.to(self.args.device)

                if epochs - best_epoch >= self.args.early_stop:
                    if self.args.save_labels:
                        with open(
                            os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'),
                            'wb'
                        ) as df:
                            plk.dump(saved_labels, df, protocol=4)
                    return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        amp_enabled = torch.cuda.is_available()   # [MOD]

        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)

                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        else:
                            # [MOD] 对齐模式下兜底
                            text_lengths = None
                            audio_lengths = None
                            vision_lengths = None

                        # [MOD] 新 AMP 写法
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        predict_label = torch.Tensor(outputs).to(self.args.device)

                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())

            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            eval_results = self.metrics(pred, true)
            logger.info('M: >> ' + dict_to_str(eval_results))

        else:
            # train_mode == 'classification'
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)

                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        else:
                            # [MOD] 对齐模式下兜底
                            text_lengths = None
                            audio_lengths = None
                            vision_lengths = None

                        # [MOD] 新 AMP 写法
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths),
                                                     (vision, vision_lengths))

                        # [MOD] 核心修改：
                        # 原来直接把字符串输出喂给 metrics，导致 string vs number 报错
                        # 现在统一转成数字类别 id
                        predict_label = [self._parse_cls_output(x) for x in outputs]

                        labels_m = batch_data['labels']['M']
                        if isinstance(labels_m, torch.Tensor):
                            labels_m = labels_m.view(-1).cpu().tolist()
                        else:
                            labels_m = list(labels_m)

                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)

            pred, true = list(chain(*y_pred['M'])), list(chain(*y_true['M']))
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        return eval_results

    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss

    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def save_model(self, model, epoch, save_path):
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.cpu().state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_path))
        torch.save(state_dict, save_path)