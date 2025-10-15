import math

import torch
from torcheval.metrics import BinaryAccuracy, BinaryRecall, BinaryPrecision
from torch.utils.data import DataLoader, RandomSampler

from core.common.constants import *
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.preprocess.instances import batch_to_device


class Evaluator(object):
    @staticmethod
    def evaluate_nct(
            instances,
            instance_label,
            model,
            config,
            logger,
            nct_pos_th=None,
            metadata_output=False,
            eps=1e-13
    ):
        eval_data = CustomDataset(data=instances)

        sampler = RandomSampler(eval_data)

        batch_size = config.batch_size
        data_loader = DataLoader(dataset=eval_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=False)

        logger.info('***** Evaluating *****')
        model.zero_grad()
        total_loss = 0.0
        total_items = 0
        total_output = []
        total_gold = []
        total_pred = []
        for _, batch in enumerate(data_loader):
            batch = batch_to_device(batch, config.device)
            model.eval()
            with torch.no_grad():
                output = model(batch, nct_pos_th=nct_pos_th)
                total_loss += output[LOCAL_LOSS].detach().item()
                total_items += output['num_items']
                total_gold.extend(sum(output['gold'], []))
                total_pred.extend(sum(output['pred'], []))

                if metadata_output:
                    output[META_DATA] = batch[META_DATA]
                total_output.append(output)

        assert len(total_gold) == len(total_pred)
        total_loss = total_loss / total_items
        accuracy = BinaryAccuracy().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
        precision = BinaryPrecision().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
        recall = BinaryRecall().update(input=torch.tensor(total_pred), target=torch.tensor(total_gold)).compute()
        fscore = (1.0 + config.fscore_beta ** 2) * precision * recall / (config.fscore_beta ** 2 * precision + recall + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        logger.info(
            'Eval({}-th{})-Loss:{}, accuracy:{}, precision:{}, recall:{}, f1:{}, fscore(beta={}):{}'.format(
                instance_label, nct_pos_th, total_loss, accuracy, precision, recall, f1, config.fscore_beta, fscore))
        eval_output = {
            TOTAL_LOSS: total_loss,
            OUTPUT: total_output,
            'nct_pos_th': nct_pos_th,
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'fscore': fscore.item(),
            'fscore_beta': config.fscore_beta
        }
        return eval_output

