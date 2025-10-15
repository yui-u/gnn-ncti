import copy
import math
from pathlib import Path

import torch
from torcheval.metrics import BinaryAccuracy, BinaryRecall, BinaryPrecision
from torch.nn.utils import clip_grad_value_
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange

from core.common.constants import *
from core.eval.evaluator import Evaluator
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.preprocess.instances import batch_to_device


class Trainer(object):
    @staticmethod
    def train_nct(train_instances, dev_instances, model, config, num_epochs, logger, backup_filename=None, eps=1e-13):
        if config.metric in [TOTAL_LOSS, NLL, PERPLEXITY]:
            assert config.metric in [TOTAL_LOSS]
            lower_is_better = True
        else:
            lower_is_better = False

        train_data = CustomDataset(data=train_instances)

        sampler = RandomSampler(train_data)

        batch_size = config.batch_size
        iterator = trange(num_epochs, desc='Epoch', disable=False)
        data_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=False)

        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        logger.info('***** Start Training *****')
        best_eval_metric = None
        best_epoch = -1
        best_model = None
        warmup_nct = min(num_epochs - 1, config.warmup_nct)
        for epoch in iterator:
            logger.info('***** Epoch: {} *****'.format(epoch))
            total_loss = 0.0
            total_items = 0
            total_gold = []
            total_pred = []
            for _, batch in enumerate(data_loader):
                batch = batch_to_device(batch, config.device)
                model.to(config.device)

                model.train()
                model.zero_grad()

                output = model(batch)
                loss = output[LOCAL_LOSS] / output['num_items']
                loss.backward()
                total_loss += output[LOCAL_LOSS].detach().item()
                if 0 < config.gradient_clip_value:
                    clip_grad_value_(model.parameters(), config.gradient_clip_value)
                optimizer.step()

                total_items += output['num_items']
                total_gold.extend(sum(output['gold'], []))
                total_pred.extend(sum(output['pred'], []))

            assert len(total_gold) == len(total_pred)
            total_loss = total_loss / total_items
            accuracy = BinaryAccuracy().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
            precision = BinaryPrecision().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
            recall = BinaryRecall().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
            fscore = (1.0 + config.fscore_beta**2) * precision * recall / (config.fscore_beta**2 * precision + recall + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            logger.info(
                'Train-Loss:{}, accuracy:{}, precision:{}, recall:{}, f1: {}, fscore(beta={}):{}'.format(
                    total_loss, accuracy, precision, recall, f1, config.fscore_beta, fscore))

            # eval
            eval_result = Evaluator.evaluate_nct(dev_instances, DEV, model, config, logger)
            eval_metric = eval_result[config.metric]
            if epoch < warmup_nct:
                model_to_be_updated = False
            else:
                if best_eval_metric is None:
                    model_to_be_updated = True
                elif lower_is_better and (eval_metric < best_eval_metric):
                    model_to_be_updated = True
                elif (not lower_is_better) and (best_eval_metric < eval_metric):
                    model_to_be_updated = True
                else:
                    model_to_be_updated = False

            if model_to_be_updated:
                logger.info('Update best epoch')
                best_eval_metric = eval_metric
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                if backup_filename is not None:
                    torch.save(best_model.state_dict(), backup_filename)
            else:
                if (0 < config.patience) and (config.patience < epoch - best_epoch):
                    logger.info('Early stopping, Best Epoch: {}'.format(best_epoch))
                    break
        logger.info('End Training, Best Epoch: {}'.format(best_epoch))
        return best_model, best_epoch