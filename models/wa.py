import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import *


EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [60, 100, 140]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
T = 2


class WA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        if self._cur_task > 0: self._network.weight_align(self._total_classes - self._known_classes)
        self._old_network = self._network.copy().freeze()
        super().after_task()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', appendent=self._get_memory(), augmentation_prob=self.args['augmentation_prob'], augmentations_per_image=self.args['augmentations_per_image'])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.args['num_workers'])
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.args['num_workers'])
        
        checkpoint_path = f"{self.args['log_dir']}/{self._cur_task}.pkl"
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            if len(self._multiple_gpus) > 1: self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.train_loader, self.test_loader)
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            if len(self._multiple_gpus) > 1: self._network = self._network.module
            self.save_checkpoint(checkpoint_path)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None: self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=init_lr, momentum=0.9, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1: self._network.module.weight_align(self._total_classes - self._known_classes)
            else: self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        checkpoint_path = f"{output_folder()}/Models/{self.args['convnet_type']}/{self.args['dataset']}_cls{self.args['init_cls']}/{self.args['model_name']}/aug{self.args['augmentation_prob']}x{self.args['augmentations_per_image']}_seed{self.args['model_seed']}.pt"
        if os.path.exists(checkpoint_path):
            (self._network.module if len(self._multiple_gpus) > 1 else self._network).load_state_dict(torch.load(checkpoint_path))
            return
        prog_bar = tqdm(range(init_epoch))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save((self._network.module if len(self._multiple_gpus) > 1 else self._network).state_dict(), checkpoint_path)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        kd_lambda = self._known_classes / self._total_classes
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(logits[:, : self._known_classes], self._old_network(inputs)['logits'], T)
                loss = (1-kd_lambda) * loss_clf + kd_lambda * loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
