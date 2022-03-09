import json
import os
import time
from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpyencoder import NumpyEncoder

from faceformer.post_processing import (filter_faces_by_encloseness, map_coedge_into_edges)
from faceformer.utils import flatten_list


class Trainer(pl.LightningModule):
    def __init__(self, hparams, model_class, dataset_class):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model_class(**self.hparams.model)
        self.dataset_class = dataset_class
        self.validation_num = 0
        self.time_count = 0
        self.total_time = 0

    def forward(self, batch):
        return self.model(batch)

    def train_dataloader(self):
        dataset = self.dataset_class(self.hparams.root_dir, self.hparams.datasets_train, self.hparams.model)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.hparams.batch_size_train, num_workers=4,
            shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        name_dir = os.path.join(self.logger.log_dir, self.hparams.trainer.version)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)
        dataset = self.dataset_class(self.hparams.root_dir, self.hparams.datasets_valid, self.hparams.model)
        self.dataset = dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.hparams.batch_size_valid, num_workers=4,
            shuffle=False, drop_last=False)
        return dataloader
    
    def test_dataloader(self):
        dataset = self.dataset_class(self.hparams.root_dir, self.hparams.datasets_test, self.hparams.model)
        self.dataset = dataset
        self.hparams.batch_size_valid = 1
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.hparams.batch_size_valid, num_workers=4,
            shuffle=False, drop_last=False)
        json_dir = os.path.join(self.logger.log_dir, 'json')    
        if not os.path.exists(json_dir):
            os.mkdir(json_dir) 
        return dataloader

    def compute_loss(self, outputs):
        embedding, pointer, labels = outputs['embedding'], outputs['pointer'], outputs['label']

        # embedding N x L x E, pointer N x T x E
        # logits: N x L x T
        logits = torch.bmm(embedding, pointer.transpose(1, 2))

        #label: N x T
        labels = labels.detach().clone()
        loss = F.cross_entropy(
            logits, labels, ignore_index=self.hparams.model.token.PAD, reduction='sum')

        valid = labels != self.hparams.model.token.PAD
        valid_sum = valid.sum()
        pred = torch.argmax(logits, dim=1)
        outputs['predict'] = pred
        acc_sum = (valid * (pred == labels)).sum()
        cls_acc = float(acc_sum) / (valid_sum + 1e-10)

        loss = loss / valid_sum
        return loss, cls_acc

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss, acc = self.compute_loss(outputs)
        self.log('train_loss', loss, logger=True)
        self.log('train_cls_acc', acc, prog_bar=True, logger=True)
        if torch.isnan(loss):
            return None
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        acc, outputs = self.face_accuracy(outputs)
        
        self.log('valid_accuracy', np.mean(outputs['accuracy']), logger=True)
        self.log('valid_type_acc_coedge_seq', np.mean(outputs['type_acc_coedge_seq']), logger=True)
        self.log('valid_precision', np.mean(outputs['precisions']), logger=True)
        self.log('valid_recall', np.mean(outputs['recalls']), logger=True)
        self.log('type_acc', np.mean(outputs['type_acc']), logger=True)
        for pred, label, prec in zip(outputs['predictions'], outputs['labels'], outputs['precisions']):
            self.logger.experiment.add_text('result', f'pred: {pred} \n\n label: {label} \n\n precision: {prec}', self.validation_num)
        # return outputs

    # saves all necessary info for reconstruction
    def test_step(self, batch, batch_idx):
        torch.cuda.synchronize()
        a = time.time()
        outputs = self.forward(batch)
        torch.cuda.synchronize()
        self.total_time += time.time() - a
        self.time_count += 1
        print("Avg Time", self.total_time / self.time_count, "seconds.")
        acc, outputs = self.face_accuracy(outputs)
        
        self.log('test_precision', np.mean(outputs['precisions']), logger=True)
        self.log('test_recall', np.mean(outputs['recalls']), logger=True)
        self.log('test_type_acc', np.mean(outputs['type_acc']), logger=True)
        for ind in range(len(outputs['predictions'])):
            predict_faces_w_types = outputs['predictions'][ind]
            label_faces_w_types = outputs['labels'][ind]
            json_name = batch['name'][ind]
            name = json_name[5:13]

            with open(os.path.join(self.hparams.root_dir, json_name), "r") as f:
                raw_data = json.loads(f.read())
            edges = raw_data['edges']

            predicted_data = {}
            predicted_data['edges'] = edges
            predicted_data['dominant_directions'] = raw_data['dominant_directions']
            predicted_data['pred_faces'] = predict_faces_w_types
            predicted_data['label_faces'] = label_faces_w_types


            with open(os.path.join(self.logger.log_dir, 'json', f'{name}.json'), 'w') as f:
                json.dump(predicted_data, f, cls=NumpyEncoder)

    def validation_epoch_end(self, outputs):
        self.validation_num += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)
        if self.hparams.trainer.lr_step == 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.trainer.lr_step)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    # Parse single-sequence faces
    # Return (0, faces' indices) to be consistent with face_typing task
    def parse_faces(self, predicts, labels, num_edges):
        # cut off tokens after the [EOS]
        label = np.split(labels, np.where(labels == self.hparams.model.token.EOS)[0]+1)[0]
        predict = np.split(predicts, np.where(predicts == self.hparams.model.token.EOS)[0] + 1)[0]

        label_faces = np.split(label, np.where(label == self.hparams.model.token.SEP)[0]+1) # split by SEP
        constructed_label_faces = []
        for face in label_faces:
            label = face[:-1] - self.hparams.model.token.len # remove SEP and remove token offset
            label = label[label >= 0]
            label = label[label < num_edges]
            if len(label) > 0:
                constructed_label_faces.append((0, tuple(label.tolist())))

        predict_faces = np.split(predict, np.where(predict == self.hparams.model.token.SEP)[0]+1) # split by SEP
        constructed_predict_faces = []
        for face in predict_faces:
            if len(face) > 1:
                predict = face[:-1] - self.hparams.model.token.len # remove SEP and remove token offset
                predict = predict[predict >= 0]
                predict = predict[predict < num_edges]
                if len(predict) > 0:
                    constructed_predict_faces.append((0, tuple(predict.tolist())))

        return constructed_predict_faces, constructed_label_faces

    # Parse multi-sequence faces
    # Return faces' indices
    def parse_parallel_faces(self, predicts, labels, num_edges):
        predict_faces, label_faces = [], []
        # cut off tokens after the [EOS] (Type of face in this case)
        for label in labels:
            label = np.split(label, np.where((label >= self.hparams.model.token.face_type_offset) & (label < self.hparams.model.token.len))[0]+1)[0]
            # extract face type
            face_type = label[-1] - self.hparams.model.token.face_type_offset
            # remove token offset
            label -= self.hparams.model.token.len
            # only take the valid indices
            label = label[label >= 0]
            if len(label) > 0:
                # only count the face if face is not empty and not full of paddings
                label_faces.append((face_type, tuple(label.tolist())))
            
        for predict in predicts:
            predict = np.split(predict, np.where((predict >= self.hparams.model.token.face_type_offset) & (predict < self.hparams.model.token.len))[0]+1)[0]
            # extract face type
            face_type = predict[-1] - self.hparams.model.token.face_type_offset
            # remove token offset
            predict -= self.hparams.model.token.len
            # only take the valid indices
            predict = predict[predict >= 0]
            predict = predict[predict < num_edges]
            if len(predict) > 0:
                predict_faces.append((face_type, tuple(predict.tolist())))

        return predict_faces, label_faces

    def face_accuracy(self, outputs):
        labels = outputs['label'].cpu().numpy()             # N (x F) x T
        predicts = outputs['predict'].cpu().numpy()         # N (x L) x T

        outputs.update({'precisions': [], 'labels': [], 'type_acc_coedge_seq': [], \
                        'recalls': [], 'predictions': [], 'accuracy': [], 'type_acc':[]})

        for ind in range(len(labels)):
            edges = self.dataset.raw_datas[outputs['id'][ind]]['edges']
            if len(labels.shape) == 3:
                # multi-seq, parallel
                predict_faces, label_faces = self.parse_parallel_faces(predicts[ind], labels[ind], len(edges))
            else:
                # single-seq
                predict_faces, label_faces = self.parse_faces(predicts[ind], labels[ind], len(edges))

            if self.hparams.post_process.is_coedge:
                pairings = self.dataset.raw_datas[outputs['id'][ind]]['pairings']

                predict_faces = filter_faces_by_encloseness(edges, predict_faces, self.hparams.post_process.enclosedness_tol)
                label_faces = filter_faces_by_encloseness(edges, label_faces, self.hparams.post_process.enclosedness_tol)
                
                # calculate accuracy for faces with coedge
                # consider accuracy as the percent of predictions made correct
                face_tp = 0
                type_tp = 0
                for pred_type, pred_face in predict_faces:
                    for label_type, label_face in set(label_faces):
                        if pred_face == label_face:
                            face_tp += 1
                            if pred_type == label_type:
                                type_tp += 1
                            break
                
                if len(predict_faces) == 0:
                    outputs['accuracy'].append(0)
                    outputs['type_acc_coedge_seq'].append(0)
                else:
                    outputs['accuracy'].append(face_tp / len(predict_faces))
                    if face_tp == 0:
                        outputs['type_acc_coedge_seq'].append(0)
                    else:
                        outputs['type_acc_coedge_seq'].append(type_tp / face_tp)
                # map coedge into edges
                label_faces = [(ftype, map_coedge_into_edges(pairings, flatten_list(loops))) for ftype, loops in label_faces]
                predict_faces = [(ftype, map_coedge_into_edges(pairings, flatten_list(loops))) for ftype, loops in predict_faces]

            # filter duplicate label faces
            label_faces_set = list(set([(ftype, tuple(sorted(set(indices)))) for ftype, indices in label_faces]))
        
            # determine face type by majority vote
            predict_unique_faces = {}
            for ftype, indices in predict_faces:
                face = tuple(sorted(set(indices)))
                if face in predict_unique_faces:
                    predict_unique_faces[face].append(ftype)
                else:
                    predict_unique_faces[face] = [ftype]
            
            predict_faces_set = [(Counter(ftypes).most_common(1)[0][0], face) for face, ftypes in predict_unique_faces.items()]

            # count TP
            face_tp = 0
            type_tp = 0
            for pred_type, pred_face in predict_faces_set:
                for label_type, label_face in label_faces_set:
                    if pred_face == label_face:
                        face_tp += 1
                        if pred_type == label_type:
                            type_tp += 1
                        break

            if len(predict_faces_set) == 0 or len(label_faces_set) == 0:
                outputs['precisions'].append(0)
                outputs['recalls'].append(0)
                outputs['type_acc'].append(0)
            else:
                outputs['precisions'].append(face_tp / len(predict_faces_set))
                outputs['recalls'].append(face_tp / len(label_faces_set))
                if face_tp == 0:
                    outputs['type_acc'].append(0)
                else:
                    outputs['type_acc'].append(type_tp / face_tp)
            outputs['predictions'].append(predict_faces_set)
            outputs['labels'].append(label_faces_set)
        
        # with first token removed, we only look at non-padded elements
        valid = labels > self.hparams.model.token.PAD
        acc_sum = (valid * (predicts == labels)).sum()
        valid_sum = valid.sum()
        return acc_sum/valid_sum, outputs


