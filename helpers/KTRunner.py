# -*- coding: UTF-8 -*-

import torch
import logging
from time import time
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
import copy
import os
import matplotlib
import matplotlib.pyplot as plt

from utils import utils


class KTRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=0.,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--metric', type=str, default='AUC, Accuracy, MAE,RMSE',
                            help='metrics: AUC, F1, Accuracy, Recall, Presicion;'
                                 'The first one will be used to determine whether to early stop')
        return parser

    def __init__(self, args):
        self.optimizer_name = args.optimizer
        self.learning_rate = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.metrics = args.metric.strip().lower().split(',')
        self.early_stop = args.early_stop
        self.time = None

        self.valid_results, self.test_results = {}, {}
        for i in range(len(self.metrics)):
            self.metrics[i] = self.metrics[i].strip()
            self.valid_results[self.metrics[i]] = list()
            self.test_results[self.metrics[i]] = list()

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adadelta':
            logging.info("Optimizer: Adadelta")
            optimizer = torch.optim.Adadelta(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def predict(self, model, corpus, set_name):
        model.eval()
        predictions, labels = [], []
        batches = model.prepare_batches(corpus, corpus.data_df[set_name], self.eval_batch_size, phase=set_name)
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = model.batch_to_gpu(batch)
            outdict = model(batch)
            prediction, label = outdict['prediction'], outdict['label']
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())
        return np.array(predictions), np.array(labels)

    def fit(self, model, corpus, epoch_train_data, epoch=-1):  # fit the results for an input set
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        model.train()
        loss_lst = list()
        batches = model.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
            batch = model.batch_to_gpu(batch)
            model.optimizer.zero_grad()
            output_dict = model(batch)
            loss = model.loss(batch, output_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        model.eval()
        return np.mean(loss_lst)

    def eva_termination(self, model):   #早停机制
        valid = self.valid_results[self.metrics[0]]
        if len(valid) > 15 and utils.non_increasing(valid[-8:]):
            return True
        elif len(valid) - valid.index(max(valid)) > 15:
            return True
        return False

    def train(self, model, corpus):
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)

        try:
            for epoch in range(self.epoch):
                gc.collect()
                self._check_time()
                epoch_train_data = copy.deepcopy(corpus.data_df['train'])
                epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True)
                loss = self.fit(model, corpus, epoch_train_data, epoch=epoch + 1)
                del epoch_train_data
                training_time = self._check_time()

                # output validation
                valid_result = self.evaluate(model, corpus, 'dev')
                test_result = self.evaluate(model, corpus, 'test')
                testing_time = self._check_time()

                for metric in self.metrics:
                    self.valid_results[metric].append(valid_result[metric])
                    self.test_results[metric].append(test_result[metric])

                logging.info("Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t valid=({}) test=({}) [{:<.1f} s] ".format(
                             epoch + 1, loss, training_time, utils.format_metric(valid_result),
                             utils.format_metric(test_result), testing_time))

                if max(self.valid_results[self.metrics[0]]) == self.valid_results[self.metrics[0]][-1]:
                    model.save_model()
                if self.eva_termination(model) and self.early_stop:
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best validation result across iterations
        best_valid_score = max(self.valid_results[self.metrics[0]])
        best_epoch = self.valid_results[self.metrics[0]].index(best_valid_score)
        valid_res_dict, test_res_dict = dict(), dict()
        for metric in self.metrics:
            valid_res_dict[metric] = self.valid_results[metric][best_epoch]
            test_res_dict[metric] = self.test_results[metric][best_epoch]
        logging.info("\nBest Iter(dev)=  %5d\t valid=(%s) test=(%s) [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(valid_res_dict),
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))

        best_test_score = max(self.test_results[self.metrics[0]])
        best_epoch = self.test_results[self.metrics[0]].index(best_test_score)
        for metric in self.metrics:
            valid_res_dict[metric] = self.valid_results[metric][best_epoch]
            test_res_dict[metric] = self.test_results[metric][best_epoch]
        logging.info("Best Iter(test)= %5d\t valid=(%s) test=(%s) [%.1f s] \n"
                     % (best_epoch + 1,
                        utils.format_metric(valid_res_dict),
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))
        model.load_model()

    def evaluate(self, model, corpus, set_name):  # evaluate the results for an input set
        predictions, labels = self.predict(model, corpus, set_name)
        # knowledge tracing on a specific knowledge set
        lengths = np.array(list(map(lambda lst: len(lst) - 1, corpus.data_df[set_name]['skill_seq'])))
        concat_pred, concat_label = list(), list()
        for pred, label, length in zip(predictions, labels, lengths):
            concat_pred.append(pred[:length])
            concat_label.append(label[:length])
        concat_pred = np.concatenate(concat_pred)
        concat_label = np.concatenate(concat_label)
        return model.pred_evaluate_method(concat_pred, concat_label, self.metrics)

    def print_res(self, model, corpus):
        set_name = 'test'
        result = self.evaluate(model, corpus, set_name)
        res_str = utils.format_metric(result)
        return res_str
    
    def plt_res(self, model, corpus):
        set_name = 'train'
        predictions, labels = self.predict(model, corpus, set_name)
        predictions = torch.from_numpy(predictions)
        labels = torch.from_numpy(labels)
        # 假设 q 是 `skill_seq` 的数据
        q = corpus.data_df[set_name]['problem_seq'] 
        
        s = corpus.data_df[set_name]['correct_seq'] 
        
        # 找到最长的序列长度
        max_len = max(len(seq) for seq in q)

        # 使用 np.pad 在每个序列的末尾填充 -1，使其长度相同
        q_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=-1) for seq in q])
        s_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=-1) for seq in s])

        # 将 numpy 数组转换为 PyTorch 张量
        q_tensors = torch.tensor(q_padded, dtype=torch.long)
        s_tensors = torch.tensor(s_padded, dtype=torch.long)
# #         print(q_tensors.shape)
# #         print(q_tensors.dtype)
#         # 保存张量到文件
# #         torch.save(predictions, 'plt/predictions.pt')
        torch.save(q_tensors, 'plt/p_train_tensors.pt')
        torch.save(s_tensors, 'plt/s_train_tensors.pt')
#         fig = trace_map(predictions, q_tensors[:,1:], labels, text_label=True)
#         # 保存生成的图像
#         fig.savefig('plt/trace_map_output.png', bbox_inches='tight')
    
        return 

def trace_map(y, q, s, sample_indices=[4,5,6], span=range(30), k_color=None, text_label=False, figsize=(22, 6)):
    # 仅选择指定的样本进行可视化
    selected_y = y[sample_indices, :]
    selected_q = q[sample_indices, :]
    selected_s = s[sample_indices, :]

    n_samples = len(sample_indices)  # 选中的样本数
    n_timesteps = len(span)  # 时间步长

    # 创建一个单一的图像
    fig, ax = plt.subplots(figsize=figsize)

    # 合并所有样本的图像数据
    combined_image = np.zeros((n_samples, n_timesteps))
    for idx, sample_idx in enumerate(sample_indices):
        combined_image[idx, :] = selected_y[idx, span].reshape(1,-1).numpy()

    # 绘制合成的图像
    im = ax.imshow(combined_image, aspect='auto', cmap="RdYlGn", vmin=0, vmax=1)

    # 添加 colorbar
    plt.colorbar(im, ax=ax, location="right")

    # circle label
    if k_color is None:
        knows = list(set(selected_q.flatten().tolist()))
        cmap = matplotlib.cm.get_cmap("tab20")
        k_color = {k: cmap(i) for i, k in enumerate(knows)}

    x_offset = 0.5
    y_offset = 0.5  # 修改偏移量，使圆圈居中显示在单元格上方
    for x, i in enumerate(span):
        if i == 0:
            continue
        for idx, sample_idx in enumerate(sample_indices):
            q_ = selected_q[idx, i - 1].item()
            s_ = selected_s[idx, i - 1].item()
            ax.add_patch(
                plt.Circle((x + x_offset, idx + y_offset), 0.3, color=k_color[q_], clip_on=False)
            )
            if s_ == 1:
                ax.add_patch(
                    plt.Circle(
                        (x + x_offset, idx + y_offset), 0.15, color="w", zorder=100, clip_on=False
                    )
                )

    # text label
    if text_label:
        label = []
        for i in span:
            if i == 0:
                label.append("-")
            else:
                label.append(f"{selected_q[:, i-1].tolist()}-{selected_s[:, i-1].tolist()}")
        ax.set_xticks(np.arange(0.5, len(span)), label)

    ax.set_yticks(np.arange(n_samples))
    ax.set_yticklabels(sample_indices)
    
    return fig



