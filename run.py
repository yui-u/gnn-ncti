import argparse
import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import music21 as m21
import torch
from torchinfo import summary

from core.common.constants import *
from core.eval.evaluator import Evaluator
from core.model.model import NCTModel
from core.preprocess.reader import (NCT_BOTH_INDEX,
                                    NCT_CHORD_INDEX,
                                    NCT_NONCHORD_INDEX,
                                    NCT_IGNORE_INDEX,
                                    MxlReader)
from core.trainer.trainer import Trainer
from core.util.config import NCTModelConfig, TrainConfig
from core.util.logging import create_logger
from core.util.util import set_seed

DATA_PARSER = argparse.ArgumentParser(add_help=False)
DATA_PARSER.add_argument('--dataset',
                         type=str,
                         default='little-organ-book-rntxt',
                         help='Dataset name')
DATA_PARSER.add_argument('--cv_num_set',
                         type=int,
                         default=-1,
                         help='Number of cross-validation sets. Set -1 when using fixed train, dev, and test splits.')
DATA_PARSER.add_argument('--resolution_str',
                         type=str,
                         default='halfbeat',
                         choices=['16th', '8th', 'halfbeat', 'beat'],
                         help='Time resolution for predicting harmonic analysis')
DATA_PARSER.add_argument('--ann_resolution',
                         type=float,
                         default=0.125,
                         help='Time resolution of the annotation')
DATA_PARSER.add_argument('--chord_type',
                         type=str,
                         default='full',
                         choices=['triad', 'triad+dominant', 'full'],
                         help='Chordtone type in generating labels for NCT recognition.')
DATA_PARSER.add_argument('--alert_nct_ratio_th',
                         type=float,
                         default=0.5,
                         help='If the NCT ratio exceeds this threshold, '
                              'there is a high possibility of inconsistencies '
                              '(such as differences in key signatures) in the annotation and score, '
                              'so exclude it from the dataset.')

TRAIN_PARSER = argparse.ArgumentParser(add_help=False)
TRAIN_PARSER.add_argument('--dir_preprocessed_dataset',
                          type=str,
                          help='Path of the preprocessed data set used for training')
TRAIN_PARSER.add_argument('--debug',
                          action='store_true',
                          help='Debug mode: only the first phrase of each piece is used.')
TRAIN_PARSER.add_argument('--device',
                          type=str,
                          default='cpu',
                          help='`cuda:n` where `n` is an integer, or `cpu`')
TRAIN_PARSER.add_argument('--seed',
                          type=int,
                          default=123,
                          help='seed')
TRAIN_PARSER.add_argument('--cv_set_no',
                          type=int,
                          default=0,
                          help='Cross-validation set number')
TRAIN_PARSER.add_argument('--batch_size',
                          type=int,
                          default=2,
                          help='batch size')
TRAIN_PARSER.add_argument('--learning_rate',
                          type=float,
                          default=1e-3,
                          help='learning rate')
TRAIN_PARSER.add_argument('--metric',
                          type=str,
                          default='fscore',
                          choices=['accuracy', 'precision', 'recall', 'fscore'],
                          help='Metric to update the best model')
TRAIN_PARSER.add_argument('--fscore_beta',
                              type=float,
                              default=1.0,
                              help='Beta for the F-score.')
TRAIN_PARSER.add_argument('--warmup_nct',
                          type=int,
                          default=16,
                          help='Number of warm-up epochs for NCT model.')
TRAIN_PARSER.add_argument('--gradient_clip_value',
                          type=float,
                          default=-1,
                          help='gradient clip value. Set -1 to disable')
TRAIN_PARSER.add_argument('--patience',
                          type=int,
                          default=48,
                          help='early stop patience. set -1 to disable')

NCT_MODEL_PARSER = argparse.ArgumentParser(add_help=False)
NCT_MODEL_PARSER.add_argument('--gnn_model_type',
                              type=str,
                              default='gatv2',
                              choices=['gatv2', 'sage'],
                              help='GNN model type')
NCT_MODEL_PARSER.add_argument('--gnn_hidden_size',
                              type=int,
                              default=32,
                              help='Number of GNN hidden states')
NCT_MODEL_PARSER.add_argument('--gnn_activation_fn',
                              type=str,
                              choices=['tanh', 'relu', 'leaky_relu', 'mish'],
                              default='relu',
                              help='Type of activation function for GAT')
NCT_MODEL_PARSER.add_argument('--num_gat_heads',
                              type=int,
                              default=2,
                              help='Number of attention heads for GATv2')
NCT_MODEL_PARSER.add_argument('--num_message_passing',
                              type=int,
                              default=4,
                              help='Number of Message Passing in GNN encoder')
NCT_MODEL_PARSER.add_argument('--mlp_activation_fn',
                              type=str,
                              choices=['tanh', 'relu', 'leaky_relu', 'mish'],
                              default='tanh',
                              help='Type of activation function for MLP')
NCT_MODEL_PARSER.add_argument('--mlp_hidden_size',
                              type=int,
                              default=16,
                              help='Number of MLP hidden states.')
NCT_MODEL_PARSER.add_argument('--ablation_midi',
                              action='store_true',
                              help='Ablation study (midi)')
NCT_MODEL_PARSER.add_argument('--ablation_pc',
                              action='store_true',
                              help='Ablation study (pc.)')
NCT_MODEL_PARSER.add_argument('--ablation_beat',
                              action='store_true',
                              help='Ablation study (beat)')
NCT_MODEL_PARSER.add_argument('--ablation_edge_feat',
                              action='store_true',
                              help='Ablation study (edge features. GAT(v2) only)')
NCT_MODEL_PARSER.add_argument('--dropout_p',
                              type=float,
                              default=0.25,
                              help='dropout proportion')


def run_preprocess_dataset(args):
    assert 3 <= args.cv_num_set
    dir_output = Path(args.dir_output)
    dataset_label = '{}-cvn{}-{}-{}-nctrth{}'.format(
        Path(args.dataset).name, args.cv_num_set, args.resolution_str, args.chord_type, args.alert_nct_ratio_th)
    dir_output = dir_output / Path(dataset_label)
    if not dir_output.is_dir():
        dir_output.mkdir(parents=True)
    instance_path = dir_output / Path('dataset.pkl')
    if instance_path.is_file():
        print('The preprocessed dataset already exists: {}'.format(instance_path))
    else:
        dest_log = dir_output / Path('preprocess_dataset.log')
        logger = create_logger(dest_log=dest_log)
        logger.info('\n'.join(['{}: {}'.format(k, v) for k, v in vars(args).items()]))

        dir_debug_nct_ann = dir_output / Path('nct-score')
        if not dir_debug_nct_ann.is_dir():
            dir_debug_nct_ann.mkdir()

        reader = MxlReader(
            dataset=args.dataset,
            cv_num_set=args.cv_num_set,
            resolution_str=args.resolution_str,
            chord_type=args.chord_type,
            alert_nct_ratio_th=args.alert_nct_ratio_th,
            dir_output_dataset=dir_output
        )
        instances = reader.create_instance(logger)
        pickle.dump(instances, instance_path.open('wb'))


def run_train_nct(args):
    args.dataset_normalized = True  # always normalize
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    if not Path(args.dir_output).is_dir():
        Path(args.dir_output).mkdir(parents=True)
    dir_output = Path(args.dir_output) / Path('nct-checkpoint-{}-cv{}-seed{}'.format(now, args.cv_set_no, args.seed))
    Path(dir_output).mkdir()

    # copy pretrained model
    if bool(args.pretrained_model):
        pretrained_model_path = Path(args.pretrained_model)
        (dir_output / Path('pretrained_model')).mkdir()
        shutil.copy2(
            args.pretrained_model,
            dir_output / Path('pretrained_model') / Path('{}.pt'.format(pretrained_model_path.stem)))
        with (dir_output / Path('pretrained_model') / Path('pretrained_model_info.txt')).open('w') as f:
            f.write(args.pretrained_model)
        dir_pretrained_model = pretrained_model_path.parent
        org_train_config_path = '{}/{}'.format(dir_pretrained_model, 'nct_train_config.json')
        train_config_path = '{}/{}'.format(dir_output, 'nct_train_config.json')
        shutil.copy2(org_train_config_path, train_config_path)
        with open(train_config_path) as f:
            train_config = TrainConfig(**json.load(f))
        if not torch.cuda.is_available():
            train_config.device = 'cpu'
        train_config.batch_size = args.batch_size
        with open(train_config_path, mode='wt', encoding='utf-8') as f:
            json.dump(asdict(train_config), f, ensure_ascii=True, indent=4)

        org_model_config_path = '{}/{}'.format(dir_pretrained_model, 'nct_model_config.json')
        model_config_path = '{}/{}'.format(dir_output, 'nct_model_config.json')
        shutil.copy2(org_model_config_path, model_config_path)
        with open(model_config_path) as f:
            model_config = NCTModelConfig(**json.load(f))
    else:
        train_options = [v.option_strings[0].replace('-', '') for v in vars(TRAIN_PARSER)['_actions']]
        train_args = dict([(k, v) for k, v in vars(args).items() if k in train_options])
        train_config = TrainConfig(**train_args)
        with open(dir_output / Path('nct_train_config.json'), mode='wt', encoding='utf-8') as f:
            json.dump(asdict(train_config), f, ensure_ascii=True, indent=4)

        model_options = [v.option_strings[0].replace('-', '') for v in vars(NCT_MODEL_PARSER)['_actions']]
        model_args = dict([(k, v) for k, v in vars(args).items() if k in model_options])
        model_config = NCTModelConfig(**model_args)
        with open(dir_output / Path('nct_model_config.json'), mode='wt', encoding='utf-8') as f:
            json.dump(asdict(model_config), f, ensure_ascii=True, indent=4)

    set_seed(train_config.seed, train_config.device)

    dir_preprocessed_dataset = Path(args.dir_preprocessed_dataset)
    if not dir_preprocessed_dataset.is_dir():
        print('The specified preprocessed data does not exist. Please generate the preprocessed data first.')
        return
    preprocessed_dataset_path = dir_preprocessed_dataset / Path('dataset.pkl')

    instances = pickle.load(preprocessed_dataset_path.open('rb'))
    filename2instance = dict([(v[META_DATA]['filename'], i) for (i, v) in enumerate(instances[KEY_PREPROCESS_NONE])])

    reader = MxlReader(
        dataset=instances['dataset'],
        cv_num_set=instances['cv_num_set'],
        resolution_str=instances['resolution_str'],
        chord_type=instances['chord_type'],
        alert_nct_ratio_th=instances['alert_nct_ratio_th'],
        dir_output_dataset=instances['dir_output_dataset']
    )

    dest_log = dir_output / Path('nct_train.log')
    logger = create_logger(dest_log=dest_log)
    logger.info('\n'.join(['{}: {}'.format(k, v) for k, v in vars(args).items()]))

    if args.dataset_normalized:
        train_instances, dev_instances, test_instances = reader.get_train_dev_test_instances(
            logger, instances[KEY_PREPROCESS_NORMALIZE], instances['splits'], train_config.cv_set_no)
    else:
        raise ValueError
        # train_instances, dev_instances, test_instances = reader.get_train_dev_test_instances(
        #     logger, instances[KEY_PREPROCESS_NONE], instances['splits'], train_config.cv_set_no)
    for instance_type, _instances in [(TRAIN, train_instances), (DEV, dev_instances), (TEST, test_instances)]:
        n_chord = 0
        n_nonchord = 0
        n_both = 0
        for _instance in _instances:
            n_chord += (_instance['graph'].y.squeeze(-1) == NCT_CHORD_INDEX).long().sum()
            n_nonchord += (_instance['graph'].y.squeeze(-1) == NCT_NONCHORD_INDEX).long().sum()
            n_both += (_instance['graph'].y.squeeze(-1) == NCT_BOTH_INDEX).long().sum()
        n_all = float(n_chord + n_nonchord + n_both)
        logger.info('{}:, chord({})-ratio:{}, nonchord({})-ratio:{}, both({})-ratio:{}'.format(
            instance_type,
            n_chord, n_chord / n_all,
            n_nonchord, n_nonchord / n_all,
            n_both, n_both / n_all)
        )

    model = NCTModel(model_config, train_config.device)
    if bool(args.pretrained_model):
        model.load_state_dict(
            torch.load(args.pretrained_model,
                       map_location=torch.device(train_config.device),
                       weights_only=True))
    logger.info('\n{}'.format(summary(model)))
    logger.info('\n{}'.format(summary(model.graph_encoder)))

    logger.info('Start training (NCT), num_epochs={}'.format(args.num_epochs))
    model.to(train_config.device)
    best_model, best_epoch = Trainer.train_nct(
        train_instances, dev_instances, model, train_config, args.num_epochs, logger,
        backup_filename=dir_output / Path('model_nct_bk.pt')
    )
    logger.info('Test (NCT)')
    test_result = Evaluator.evaluate_nct(
        test_instances, TEST, best_model, train_config, logger, metadata_output=True)
    scores = {
        'accuracy': test_result['accuracy'],
        'precision': test_result['precision'],
        'recall': test_result['recall'],
        'f1': test_result['f1'],
        'fscore': test_result['fscore'],
        'fscore_beta': test_result['fscore_beta'],
        'random_scores': {}
    }
    model_filename = dir_output / Path('model_nct.pt')
    torch.save(best_model.state_dict(), model_filename)

    with (dir_output / Path('nct_test_scores.json')).open('w') as f:
        json.dump(scores, f, ensure_ascii=True, indent=2)

    dir_nct_score_pred = dir_output / Path('nct_score_pred')
    dir_nct_score_pred.mkdir()

    for o in test_result[OUTPUT]:
        batch_gold = o['gold']
        batch_pred = o['pred']
        batch_node_ids = o['node_ids']
        batch_metadata = o[META_DATA]
        for out_gold, out_pred, out_node_ids, metadata in zip(
                batch_gold, batch_pred, batch_node_ids, batch_metadata):
            assert len(out_gold) == len(out_pred) == len(out_node_ids)
            instance_none = instances[KEY_PREPROCESS_NONE][filename2instance[metadata['filename']]]
            assert instance_none[META_DATA]['filename'] == metadata['filename']
            graph = instance_none['graph']
            nct_gold = graph.y.squeeze(-1)
            nct_gold = torch.where(
                nct_gold == NCT_BOTH_INDEX,
                torch.ones_like(nct_gold) * NCT_NONCHORD_INDEX,
                nct_gold
            ).unsqueeze(-1)
            nct_pred = torch.ones((graph.x.size(0), 1)) * NCT_IGNORE_INDEX
            for og, op, nid in zip(out_gold, out_pred, out_node_ids):
                assert og == nct_gold[nid, 0]
                nct_pred[nid, 0] = op
            if 'Chorales' in instances['dataset']:
                score = m21.corpus.parse(metadata['full_filename'])
            else:
                score = m21.converter.parse(metadata['full_filename'])
            nct_score = reader.generate_nct_score(score, graph.x, nct_gold=nct_gold, nct_pred=nct_pred)
            nct_score.write('musicxml', Path(dir_nct_score_pred) / Path('{}_nct_pred.musicxml'.format(metadata['filename'])))


def run_inference_nct(args):
    args.dataset_normalized = True  # always normalize
    dir_output = Path(args.dir_output)
    if not dir_output.is_dir():
        Path(args.dir_output).mkdir(parents=True)

    dir_preprocessed_dataset = Path(args.dir_preprocessed_dataset)
    if not dir_preprocessed_dataset.is_dir():
        print('The specified preprocessed data does not exist. Please generate the preprocessed data first.')
        return
    preprocessed_dataset_path = dir_preprocessed_dataset / Path('dataset.pkl')

    instances = pickle.load(preprocessed_dataset_path.open('rb'))
    filename2instance = dict([(v[META_DATA]['filename'], i) for (i, v) in enumerate(instances[KEY_PREPROCESS_NONE])])

    reader = MxlReader(
        dataset=instances['dataset'],
        cv_num_set=instances['cv_num_set'],
        resolution_str=instances['resolution_str'],
        chord_type=instances['chord_type'],
        alert_nct_ratio_th=instances['alert_nct_ratio_th'],
        dir_output_dataset=instances['dir_output_dataset']
    )

    dest_log = dir_output / Path('nct_inference.log')
    logger = create_logger(dest_log=dest_log)
    logger.info('\n'.join(['{}: {}'.format(k, v) for k, v in vars(args).items()]))

    if args.dataset_normalized:
        train_instances, dev_instances, test_instances = reader.get_train_dev_test_instances(
            logger, instances[KEY_PREPROCESS_NORMALIZE], instances['splits'], args.cv_set_no)
    else:
        raise ValueError
        # train_instances, dev_instances, test_instances = reader.get_train_dev_test_instances(
        #     logger, instances[KEY_PREPROCESS_NONE], instances['splits'], args.cv_set_no)
    for instance_type, _instances in [(TRAIN, train_instances), (DEV, dev_instances), (TEST, test_instances)]:
        n_chord = 0
        n_nonchord = 0
        n_both = 0
        for _instance in _instances:
            n_chord += (_instance['graph'].y.squeeze(-1) == NCT_CHORD_INDEX).long().sum()
            n_nonchord += (_instance['graph'].y.squeeze(-1) == NCT_NONCHORD_INDEX).long().sum()
            n_both += (_instance['graph'].y.squeeze(-1) == NCT_BOTH_INDEX).long().sum()
        n_all = float(n_chord + n_nonchord + n_both)
        logger.info('{}:, chord({})-ratio:{}, nonchord({})-ratio:{}, both({})-ratio:{}'.format(
            instance_type,
            n_chord, n_chord / n_all,
            n_nonchord, n_nonchord / n_all,
            n_both, n_both / n_all)
        )

    train_config_path = '{}/{}'.format(args.dir_model, 'nct_train_config.json')
    with open(train_config_path) as f:
        train_config = TrainConfig(**json.load(f))
    train_config.device = 'cpu'
    model_config_path = '{}/{}'.format(args.dir_model, 'nct_model_config.json')
    with open(model_config_path) as f:
        model_config = NCTModelConfig(**json.load(f))

    model_path = '{}/{}'.format(args.dir_model, 'model_nct.pt')
    model = NCTModel(model_config, train_config.device)
    model.load_state_dict(
        torch.load(model_path,
                   map_location=torch.device('cpu'),
                   weights_only=True))
    model.to('cpu')
    logger.info('\n{}'.format(summary(model)))
    logger.info('\n{}'.format(summary(model.graph_encoder)))

    logger.info('Test (NCT)')
    test_result = Evaluator.evaluate_nct(
        test_instances, TEST, model, train_config, logger, metadata_output=True)
    scores = {
        'accuracy': test_result['accuracy'],
        'precision': test_result['precision'],
        'recall': test_result['recall'],
        'f1': test_result['f1'],
        'fscore': test_result['fscore'],
        'fscore_beta': test_result['fscore_beta'],
        'random_scores': {}
    }

    with (dir_output / Path('nct_test_scores.json')).open('w') as f:
        json.dump(scores, f, ensure_ascii=True, indent=2)

    dir_nct_score_pred = dir_output / Path('nct_score_pred')
    dir_nct_score_pred.mkdir()

    for o in test_result[OUTPUT]:
        batch_gold = o['gold']
        batch_pred = o['pred']
        batch_node_ids = o['node_ids']
        batch_metadata = o[META_DATA]
        for out_gold, out_pred, out_node_ids, metadata in zip(
                batch_gold, batch_pred, batch_node_ids, batch_metadata):
            assert len(out_gold) == len(out_pred) == len(out_node_ids)
            instance_none = instances[KEY_PREPROCESS_NONE][filename2instance[metadata['filename']]]
            assert instance_none[META_DATA]['filename'] == metadata['filename']
            graph = instance_none['graph']
            nct_gold = graph.y.squeeze(-1)
            nct_gold = torch.where(
                nct_gold == NCT_BOTH_INDEX,
                torch.ones_like(nct_gold) * NCT_NONCHORD_INDEX,
                nct_gold
            ).unsqueeze(-1)
            nct_pred = torch.ones((graph.x.size(0), 1)) * NCT_IGNORE_INDEX
            for og, op, nid in zip(out_gold, out_pred, out_node_ids):
                assert og == nct_gold[nid, 0]
                nct_pred[nid, 0] = op
            if 'Chorales' in instances['dataset']:
                score = m21.corpus.parse(metadata['full_filename'])
            else:
                score = m21.converter.parse(metadata['full_filename'])
            nct_score = reader.generate_nct_score(score, graph.x, nct_gold=nct_gold, nct_pred=nct_pred)
            nct_score.write('musicxml', Path(dir_nct_score_pred) / Path('{}_nct_pred.musicxml'.format(metadata['filename'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # generate preprocessed dataset
    parser_preprocess_dataset = subparsers.add_parser('preprocess_dataset', parents=[DATA_PARSER])
    parser_preprocess_dataset.add_argument('--dir_output',
                                           default='out',
                                           help='Path to save preprocessed data')
    parser_preprocess_dataset.set_defaults(func=run_preprocess_dataset)

    # nct train parameters
    parser_train_nct = subparsers.add_parser('train_nct', parents=[TRAIN_PARSER, NCT_MODEL_PARSER])
    parser_train_nct.add_argument('--pretrained_model',
                                  type=str,
                                  default='',
                                  help='Pretrained model path (optional)')
    parser_train_nct.add_argument('--dir_output',
                                  type=str,
                                  default='out',
                                  help='Output directory path')
    parser_train_nct.add_argument('--num_epochs',
                                  type=int,
                                  default=1024,
                                  help='Maximum epochs')
    parser_train_nct.set_defaults(func=run_train_nct)

    # inference
    parser_inference_nct = subparsers.add_parser('inference_nct')
    parser_inference_nct.add_argument('--dir_preprocessed_dataset',
                                      type=str,
                                      help='Path of the preprocessed data set')
    parser_inference_nct.add_argument('--cv_set_no',
                                      type=int,
                                      default=0,
                                      help='Cross-validation set number')
    parser_inference_nct.add_argument('--dir_model',
                                      type=str,
                                      help='Path to the trained model checkpoint.')
    parser_inference_nct.add_argument('--dir_output',
                                      type=str,
                                      default='inference',
                                      help='Output directory path')
    parser_inference_nct.set_defaults(func=run_inference_nct)

    main_args = parser.parse_args()
    main_args.func(main_args)
