import os
import sys
import copy
import torch
import logging
from utils import factory
from utils.toolkit import *
from utils.data_manager import DataManager


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):
    assert args['fixed_memory']
    experiment = f"{args['convnet_type']}/{args['dataset']}/{args['model_name']}/{args['init_cls']}_{args['increment']}_{'growing' if args['fixed_memory'] else 'fixed'}/aug_x{args['augmentations_per_image']}_seed{args['model_seed']}/aug{args['augmentation_prob']}x{args['augmentations_per_image']}__exp{args['real_per_class']}+{args['syn_per_class']}"
    log_dir = f"{output_folder()}/Outputs/{experiment}__acc"
    os.makedirs(log_dir, exist_ok=True)
    args['log_dir'] = log_dir
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=f'{log_dir}/outputs.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _set_random(args['model_seed'])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for _ in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}'.format(nme_curve['top5']))

            logging.info('Average Accuracy (CNN): {}'.format(sum(cnn_curve['top1'])/len(cnn_curve['top1'])))
            logging.info('Average Accuracy (NME): {}\n'.format(sum(nme_curve['top1'])/len(nme_curve['top1'])))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))

            logging.info('Average Accuracy (CNN): {}\n'.format(sum(cnn_curve['top1'])/len(cnn_curve['top1'])))
    
def _set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device_type == -1: device = torch.device('cpu')
        else: device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus

def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
