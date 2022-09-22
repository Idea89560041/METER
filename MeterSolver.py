import torch
from scipy import stats
import numpy as np
import models
import data_loader
from torch.nn import functional as F

class MeterIQASolver(object):
    """Solver for training and testing MeterIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_aqp = models.AQPNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_aqp.train(True)
        self.model_dti = models.DTINet(6, 224, 448, 224).cuda()
        self.model_dti.train(True)
        self.l1_loss = torch.nn.SmoothL1Loss().cuda()
        self.cross_loss = torch.nn.CrossEntropyLoss().cuda()
        backbone_params = list(map(id, self.model_aqp.res.parameters()))
        self.aqp_params = filter(lambda p: id(p) not in backbone_params, self.model_aqp.parameters())
        self.dti_params = self.model_dti.parameters()
        self.lr = config.lr
        self.dti_lr = config.dti_lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.aqp_params, 'lr': self.lr * self.lrratio},
                 {'params': self.dti_params, 'lr': self.dti_lr},
                 {'params': self.model_aqp.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTrain_ACC(%)\tTest_SRCC\tTest_PLCC\tTest_ACC(%)')

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            pred_type = []
            pred_quality = []
            gt_quality = []
            num_correct = 0.0
            num_total = 0.0

            for img, label_quality, label_type in self.train_data:
                img = torch.as_tensor(img.cuda())
                label_quality = torch.as_tensor(label_quality.cuda())
                label_type = torch.as_tensor(label_type.cuda())
                self.solver.zero_grad()

                # Generate weights for aqp network
                paras = self.model_aqp(img)  # 'paras' contains the network weights conveyed to aqp network

                # Building network
                model_AFC = models.AFCNet(paras).cuda()
                model_DTI = self.model_dti.cuda()
                for param in model_AFC.parameters():
                    param.requires_grad = False
                for param in model_DTI.parameters():
                    param.requires_grad = True

                # Type prediction
                pred_type = model_DTI(paras['target_in_vec'])

                # Quality prediction
                pred_scores = model_AFC(paras['target_in_vec'])
                pred_q = torch.mul(pred_type, pred_scores)
                pred_q = torch.sum(pred_q, dim=1, keepdim=False)
                pred_quality = pred_quality + pred_q.cpu().tolist()
                gt_quality = gt_quality + label_quality.cpu().tolist()


                loss_1 = self.l1_loss(pred_q.squeeze(), label_quality.float().detach())
                loss_2 = self.cross_loss(pred_type, label_type.detach())
                loss = 50 * loss_1 + loss_2
                epoch_loss.append(loss.item())

                _, prediction = torch.max(pred_type.data, 1)

                num_total += label_type.size(0)
                num_correct += torch.sum(prediction == label_type)
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_quality, gt_quality)
            train_plcc, _ = stats.pearsonr(pred_quality, gt_quality)
            train_acc = 100 * num_correct / num_total
            test_srcc, test_plcc, test_acc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model_aqp.state_dict(), "AQP_XXX.pth")
                torch.save(self.model_dti.state_dict(), "DTI_XXX.pth")

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f'%
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, train_acc, test_srcc, test_plcc, test_acc))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            dti_lr = self.dti_lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.aqp_params, 'lr': lr * self.lrratio},
                          {'params': self.dti_params, 'lr': dti_lr * self.lrratio},
                          {'params': self.model_aqp.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_aqp.train(False)
        self.model_dti.train(False)
        pred_scores = []
        pred_quality = []
        gt_quality = []
        num_correct = 0.0
        num_total = 0.0

        for img, label_quality, label_types in data:
            # Data.
            img = torch.as_tensor(img.cuda())
            label_quality = torch.as_tensor(label_quality.cuda())
            label_types = torch.as_tensor(label_types.cuda())

            paras = self.model_aqp(img)
            model_AFC = models.AFCNet(paras).cuda()
            model_AFC.train(False)
            model_DTI = self.model_dti.cuda()
            model_DTI.train(True)
            pred_scores = model_AFC(paras['target_in_vec'])
            pred_type = model_DTI(paras['target_in_vec'])
            pred_q = torch.mul(pred_type, pred_scores)
            pred_q = torch.sum(pred_q, dim=1, keepdim=False)

            _, prediction = torch.max(pred_type.data, 1)
            num_total += label_types.size(0)
            num_correct += torch.sum(prediction == label_types)
            pred_quality.append(float(pred_q.item()))
            gt_quality = gt_quality + label_quality.cpu().tolist()

        pred_quality = np.mean(np.reshape(np.array(pred_quality), (-1, self.test_patch_num)), axis=1)
        gt_quality = np.mean(np.reshape(np.array(gt_quality), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_quality, gt_quality)
        test_plcc, _ = stats.pearsonr(pred_quality, gt_quality)
        test_acc = 100 * num_correct / num_total
        self.model_aqp.train(True)
        self.model_dti.train(True)
        return test_srcc, test_plcc, test_acc



