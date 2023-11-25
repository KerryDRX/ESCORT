import os
import copy
import torch
import logging
import numpy as np
from torch import nn
from utils.toolkit import *
from tqdm import tqdm, trange
from torch.utils.data import DataLoader


EPSILON = 1e-8
batch_size = 64
num_workers = 4

class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args['memory_size']
        self._memory_per_class = args.get('memory_per_class', None)
        self._fixed_memory = args.get('fixed_memory', False)
        self._device = args['device'][0]
        self._multiple_gpus = args['device']

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        return self._network.module.feature_dim if isinstance(self._network, nn.DataParallel) else self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'task': self._cur_task,
            'model_state_dict': self._network.state_dict(),
            'data_memory': self._data_memory,
            'targets_memory': self._targets_memory,
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Checkpoint saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._network.load_state_dict(checkpoint['model_state_dict'])
        self._network.to(self._device)
        self._data_memory = checkpoint['data_memory']
        self._targets_memory = checkpoint['targets_memory']
        logging.info(f'Checkpoint loaded from {checkpoint_path}')

    def after_task(self):
        self._known_classes = self._total_classes
        real_paths = [path for path in self._data_memory if 'exemplars' not in path]
        syn_paths = [path for path in self._data_memory if 'exemplars' in path]
        logging.info(f'Exemplar size: {self.exemplar_size}=real{len(real_paths)}+syn{len(syn_paths)}')

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=1)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy, nme_accy = self._evaluate(y_pred, y_true), None
        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        return (self._data_memory, self._targets_memory) if len(self._data_memory) > 0 else None
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _extract_vectors(self, loader):
        self._network.eval()
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(network.extract_vector(_inputs.to(self._device)))
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def _get_samples_per_class(self):
        num_total = self.args['real_per_class'] + self.args['syn_per_class']
        num_real = self.args['real_per_class']
        num_syn = self.args['syn_per_class']
        return num_total, num_real, num_syn

    def _reduce_exemplar(self, data_manager, m):
        return
    
    def _construct_exemplar(self, data_manager, m):
        return
    
    def _construct_exemplar_unified(self, data_manager, m):
        num_total, num_real, _ = self._get_samples_per_class()
        for class_idx in trange(self._known_classes, self._total_classes, desc='Constructing exemplars'):
            logging.info(f'Class: {class_idx}')
            data, _, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train', mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, num_total + 1):
                S = np.sum(exemplar_vectors, axis=0)
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                image_path = data[i]
                selected_exemplars.append(np.array(to_exemplar(image_path, seed=0) if k > num_real else image_path))
                exemplar_vectors.append(np.array(vectors[i]))
                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(len(selected_exemplars), class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if len(self._targets_memory) != 0 else exemplar_targets
