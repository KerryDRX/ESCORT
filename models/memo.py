import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import *


EPSILON = 1e-8
batch_size = 64


class MEMO(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        args.update({
            'convnet_type':         'memo_' + args['convnet_type'],
            'train_base':           True,
            'train_adaptive':       False,

            'init_lr' :             0.1,
            'init_weight_decay' :   5e-4,
            'init_lr_decay' :       0.1,

            'lrate' :               0.1,
            'weight_decay' :        2e-4,
            'lrate_decay' :         0.1,

            'init_epoch':           200,
            'epochs':               170,
            'init_milestones' :     [60, 120, 170],
            'milestones' :          [80, 120, 150],
            
            'alpha_aux' :           1.0,
        })
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args, False)
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        super().after_task()
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info('Train Generalized Blocks...')
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info('Fix Generalized Blocks...')
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', appendent=self._get_memory(), augmentation_prob=self.args['augmentation_prob'], augmentations_per_image=self.args['augmentations_per_image'])
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers'])
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers'])

        checkpoint_path = f'{self.args["log_dir"]}/{self._cur_task}.pkl'
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            if len(self._multiple_gpus) > 1: self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.train_loader, self.test_loader)
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            if len(self._multiple_gpus) > 1: self._network = self._network.module
            self.save_checkpoint(checkpoint_path)

    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()
        if self.args['train_base']: self._network.TaskAgnosticExtractor.train()
        else: self._network.TaskAgnosticExtractor.eval()
        
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1: self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self.args['init_lr'], momentum=0.9, weight_decay=self.args['init_weight_decay'])
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args['init_milestones'], gamma=self.args['init_lr_decay'])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self.args['lrate'], momentum=0.9, weight_decay=self.args['weight_decay'])
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args['milestones'], gamma=self.args['lrate_decay'])
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1: self._network.module.weight_align(self._total_classes - self._known_classes)
            else: self._network.weight_align(self._total_classes - self._known_classes)
            
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        checkpoint_path = f"{output_folder()}/Models/{self.args['convnet_type']}/{self.args['dataset']}_cls{self.args['init_cls']}/{self.args['model_name']}/aug{self.args['augmentation_prob']}x{self.args['augmentations_per_image']}_seed{self.args['model_seed']}.pt"
        if os.path.exists(checkpoint_path):
            (self._network.module if len(self._multiple_gpus) > 1 else self._network).load_state_dict(torch.load(checkpoint_path))
            return
        prog_bar = tqdm(range(self.args['init_epoch']))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save((self._network.module if len(self._multiple_gpus) > 1 else self._network).state_dict(), checkpoint_path)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        for epoch in prog_bar:
            self.set_network()
            losses = 0.
            losses_clf = 0.
            losses_aux = 0.
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs['logits'], outputs['aux_logits']
                loss_clf = F.cross_entropy(logits,targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets-self._known_classes+1>0, aux_targets-self._known_classes+1,0)
                loss_aux = F.cross_entropy(aux_logits,aux_targets)
                loss = loss_clf + self.args['alpha_aux'] * loss_aux
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch+1, self.args['epochs'], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch+1, self.args['epochs'], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
            logging.info(info)
    
    def _construct_exemplar(self, data_manager, m):
        raise NotImplementedError('Not Implemented')
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source='train',
                mode='test',
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                
                if len(vectors) == 0:
                    break
            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            # exemplar_targets = np.full(m, class_idx)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source='train',
                mode='test',
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean