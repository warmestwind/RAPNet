import glob
import os
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import utils.common_utils as common_utils
import losses
# from nn_additional_losses import losses
from utils.data_utils import split_batch
from utils.log_utils import LogWriter
from cus_loss import *
import nrrd

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'

def gradient_loss(ss, penalty='l2'):
    '''s = flow'''
    d = 0
    for s in ss:
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        # dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        d += (torch.mean(dx) + torch.mean(dy))/2
    return d / s.shape[0]

class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=losses.DiceLoss(),
                 model_name='OneShotSegmentor',
                 labels=None,
                 num_epochs=10,
                 log_nth=5,
                 lr_scheduler_step_size=5,
                 lr_scheduler_gamma=0.5,
                 use_last_checkpoint=True,
                 exp_dir='experiments',
                 log_dir='logs'):

        self.device = device
        self.model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs
        if torch.cuda.is_available():
            self.loss_func = loss_func.cuda(device)
        else:
            self.loss_func = loss_func

        self.optim = optim([{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': 5e-5,  'weight_decay': 0.00001}])

        exp_dir_path = os.path.join(exp_dir, exp_name)
        common_utils.create_if_not(exp_dir_path)
        common_utils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(
            num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint
        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0
        self.bloss = losses.SurfaceLoss()
        self.onedice = onecls_DiceLoss()
        self.multidice = multicls_DiceLoss()

        # if use_last_checkpoint:
        # self.load_checkpoint('180_lv5hot') # 100_1shot_la
        # self.load_stn_checkpoint(1000)


    def train(self, train_loader, test_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim = self.model, self.optim

        data_loader = {
            'train': train_loader,
            'val': test_loader
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        self.logWriter.log('START TRAINING. : model name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.logWriter.log(
                'train', "\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            phase = 'train'
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # print('len data:', len(data_loader[phase]))
            print('epoch: ', epoch)
            for i_batch, sampled_batch in enumerate(data_loader[phase]):

                s_x = sampled_batch[2] # [B, Support, slice_num, 1, 256, 256]
                s_y = sampled_batch[3] # [B, Support, slice_num, 1, 256, 256]
                q_x = sampled_batch[0] # [B, slice_num, 1, 256, 256]
                q_y = sampled_batch[1].type(torch.LongTensor)  # [B, slice_num, 1, 256, 256]
                # sp_prior = sampled_batch[5][:,1].cuda(self.device, non_blocking=True)
                # sp_prior = sampled_batch[5].unsqueeze(1).float().cuda(self.device, non_blocking=True)
                # cls = sampled_batch[6]

                #s_x = sampled_batch['s_x'] # [B, Support, slice_num=1, 1, 256, 256]
                X = s_x.squeeze(2)  # [B, Support, 1, 256, 256]
                #s_y = sampled_batch['s_y'] # [B, Support, slice_num, 1, 256, 256]
                Y = s_y.squeeze(2)  # [B, Support, 1, 256, 256]
                Y = Y.squeeze(2)  # [B, Support, 256, 256]
                #q_x = sampled_batch['q_x'] # [B, slice_num, 1, 256, 256]
                q_x = q_x.squeeze(1)  # [B, 1, 256, 256]
                #q_y = sampled_batch['q_y'] # [B, slice_num, 1, 256, 256]
                q_y = q_y.squeeze(1)  # [B, 1, 256, 256]
                q_y = q_y.squeeze(1)  # [B, 256, 256]

                query_input = q_x
                # condition_input = torch.cat((input1, y1.unsqueeze(1)), dim=1)  slice =1
                # condition_input = torch.cat((input1, y1), dim=2)      # slice=3
                y2 = q_y
                y2 = y2.type(torch.LongTensor)

                temp_out = []
                temp_cond = []

                # # multi-shot method 1 fuse all pred
                # loss_cond = 0
                # for sp_index in range(X.shape[1]):
                #     input1 = X[:, sp_index, ...]  # use 1 shot at first
                #     y1 = Y[:, sp_index, ...]
                #     y1 = y1.type(torch.LongTensor)
                #     condition_input = torch.cat((input1, y1), dim=2)  # slice=3
                #
                #     if model.is_cuda:
                #         condition_input, query_input, y2, y1 = condition_input.cuda(self.device, non_blocking=True), \
                #                                                query_input.cuda(self.device, non_blocking=True), \
                #                                                y2.cuda(self.device, non_blocking=True), \
                #                                                y1.cuda(self.device, non_blocking=True)
                #
                #     # sp-se
                #     # weights = model.conditioner(condition_input)
                #     # output = model.segmentor(query_input, weights, sp_prior)
                #
                #     # lba-net
                #     condition_input = condition_input.view(-1, 6, 256, 256)
                #     attention, cond_output = model.conditioner(condition_input)
                #     # output = model.segmentor(query_input, attention, condition_input, y1.unsqueeze(1).float()) #  slice=1
                #     query_input = query_input.squeeze(2)
                #     # output = model.segmentor(query_input, attention, condition_input, y1.unsqueeze(1).float()) # slice=1
                #     output = model.segmentor(query_input, attention, condition_input, y1.float())
                #     temp_out.append(output)
                #     temp_cond.append(cond_output)
                #     # cond loss
                #     loss_cond += self.loss_func(F.softmax(cond_output, dim=1), y1[:, 1].squeeze(1))

                # method 2 prototype fused


                # import matplotlib
                # matplotlib.use('TkAgg')
                # import matplotlib.pyplot as plt
                # img = X[0, 0, 1:2, ...].squeeze().detach().cpu().numpy()
                # # img2 = tmp_sp_img_porior.squeeze().detach().cpu().numpy()
                # # img2 = inpt[:, 1:2, ...].squeeze().detach().cpu().numpy()
                # fig, axes = plt.subplots(1, 2)
                # axes[0].imshow(img)
                # # axes[1].imshow(img2)
                # # plt.imshow(img)
                # plt.show()

                temp_atten = []
                cond_inputs = []
                cond_y1s = []
                # for sp_index in range(X.shape[1]):
                #     input1 = X[:, sp_index, ...]  # use 1 shot at first
                #     y1 = Y[:, sp_index, ...]
                #     y1 = y1.type(torch.LongTensor)
                #     condition_input = torch.cat((input1, y1), dim=1)  # slice=3
                #     query_input = query_input.squeeze(2)
                #
                #     if model.is_cuda:
                #         condition_input, query_input, y2, y1 = condition_input.cuda(self.device, non_blocking=True), \
                #                                                query_input.cuda(self.device, non_blocking=True), \
                #                                                y2.cuda(self.device, non_blocking=True), \
                #                                                y1.cuda(self.device, non_blocking=True)

                    # import nrrd
                    # nrrd.write(f'prediction_la_dice_1000/y1.nrrd',
                    #            y1[0, 1].cpu().numpy().transpose(2, 1, 0).astype(np.uint8))

                    # lba-net
                   #  condition_input = condition_input.view(-1, 6, 256, 256)

                    # import matplotlib
                    # matplotlib.use('TkAgg')
                    # import matplotlib.pyplot as plt
                    # img = condition_input[0, 2:3,...].squeeze().detach().cpu().numpy()
                    # img2 = condition_input[0, 1:2, ...].squeeze().detach().cpu().numpy()
                    # # img2 = tmp_sp_img_porior.squeeze().detach().cpu().numpy()
                    # # img2 = inpt[:, 1:2, ...].squeeze().detach().cpu().numpy()
                    # fig, axes = plt.subplots(1, 2)
                    # axes[0].imshow(img)
                    # axes[1].imshow(img2)
                    # # plt.imshow(img)
                    # plt.show()

                    # attention, cond_output = model.conditioner(condition_input)
                    # temp_atten.append(attention)
                    #
                    # cond_inputs.append(condition_input)
                    # cond_y1s.append(y1)

                # # pseg
                # query_input = query_input.squeeze(2)
                # attention_p, cond_output_p = model.psegmentor(query_input)
                #
                # base_map_list = []
                # c_id_array = torch.arange(3, device='cuda')
                # for b_id in range(X.shape[0]):
                #     c_id = cls[b_id]  # cat_idx = fore cls index
                #     # if c_id == 4:
                #     #     c_id = 2
                #     c_mask = (c_id_array != c_id)
                #     pseg_fg = torch.prod(1 - cond_output_p[b_id, c_mask, :, :], dim=0) # one channel only fg, bg
                #     pseg_fg = pseg_fg * cond_output_p[b_id, c_id, :, :].unsqueeze(0).unsqueeze(0)
                #     base_map_list.append(pseg_fg)
                # base_map = torch.cat(base_map_list, 0)
                #
                # choosen_cond = []
                # cond_softmax = cond_output_p
                # for bs, c in enumerate(cls):
                #     choosen_cond.append(cond_softmax[bs, c].unsqueeze(0).unsqueeze(0))
                # cond_output_choosen = torch.cat(choosen_cond, 0)
                # loss_cond = self.onedice(cond_output_choosen, y2[:, 1])
                # # get fusd atten and prototype for cond
                # max_attens = []
                # for att_id in range(9):
                #     atten_i = []
                #     for bs_id in range(X.shape[0]):
                #         temp = []
                #         for sp_id in range(X.shape[1]):
                #             temp.append(temp_atten[sp_id][0][att_id][bs_id])
                #         max_atten = torch.mean(torch.stack(temp), 0, keepdim=True)[0]
                #         atten_i.append(max_atten)
                #     max_attens.append(torch.stack(atten_i))
                #
                # max_attens.append(None)
                # attention = list(attention)
                # attention[0] = list(attention[0])
                # attention[0] = max_attens

                if model.is_cuda:
                    y2 = y2.cuda(self.device, non_blocking=True)
                    query_input = query_input.squeeze(2).cuda(self.device, non_blocking=True)
                    cond_inputs = torch.cat([s_x.permute(1,0,2,3,4,5), s_y.permute(1,0,2,3,4,5)], 2).squeeze(3).cuda(self.device, non_blocking=True)
                    cond_y1s = s_y.permute(1,0,2,3,4,5).squeeze(3).cuda(self.device, non_blocking=True)
                else:
                    return -1
                # cond_inputs = torch.stack(cond_inputs)
                # cond_y1s = torch.stack(cond_y1s)
                sp_prior = None
                # output, sp_prior_pred, max_corr_e2, sp_img_prior = model.segmentor(query_input, cond_inputs, cond_y1s.float())
                output, sp_prior_pred, max_corr_e2 = model.segmentor(query_input, cond_inputs, cond_y1s.float())
                # sp_prior_pred, sp_img_prior, flow = model.segmentor(query_input, cond_inputs,
                #                                                      cond_y1s.float())

                # sp_prior_pred,  max_corr_e2, sp_img_prior, flow = model.segmentor(query_input, cond_inputs,
                #                                                      cond_y1s.float())
                # output, sp_prior_pred,  max_corr_e2, sp_img_prior, flow = model.segmentor(query_input, cond_inputs,
                #                                                      cond_y1s.float())
                #
                # loss = self.loss_func(F.softmax(output, dim=1), y2) # slice=3
               #  loss = self.loss_func(output, y2[:,1].squeeze(1)) # slice=3
                # print(output.shape)

                # loss = self.onedice(output, y2[:,1])
                # loss = self.onedice(max_corr_e2, y2[:, 1])
                loss = self.onedice(output, y2[:,1])
                # loss_sp_mse = torch.mean((sp_img_prior - query_input[:, 1:2, ...]) ** 2)
                # loss_g = gradient_loss(flow)
                # loss += loss_sp_mse
                # loss += loss_g
                # print(loss_g, loss_sp_mse)
                # loss = self.onedice(max_corr_e2, y2[:, 1])
                # loss_cor_e3 = self.onedice(max_corr_e3, y2[:, 1])
                print('seg loss:', loss.item())
                # print('sp loss: ', loss_sp.item())
                # print('sp loss mse: ', loss_sp_mse.item())
                # print('corr loss 2: ', loss_cor_e2.item())
                # print('corr loss 3: ', loss_cor_e3.item())
                print('-'*20)
                # loss += 0.3*loss_sp_mse
                # loss+= 0.3*loss_sp
                # loss+= 0.3*loss_cor_e2# +0.3*loss_cor_e3
                #dis_map = torch.tensor(losses.one_hot2dist(losses.class2one_hot(y2[:,1].squeeze(1), 2).cpu().numpy())).cuda(self.device, non_blocking=True)
                # loss_b = self.bloss(F.softmax(output, dim=1), dis_map)
                # loss +=loss_b
                # del condition_input, y2, y1#, dis_map
                torch.cuda.empty_cache()
                # align loss
                ''' 
                _, q_pred = torch.max(F.softmax(output, dim=1), dim=1, keepdim=True)
                align_condition_input = torch.cat((query_input, q_pred.repeat(1,3,1,1)), dim=1)
                attention, cond_output = model.conditioner(align_condition_input)
                s1 = X[:, 0, ...].squeeze(2).cuda(self.device, non_blocking=True)  # use 1 shot at first
                y1 = Y[:, 0, ...].squeeze(2).cuda(self.device, non_blocking=True)
                align_output = model.segmentor(s1, attention, align_condition_input, q_pred.repeat(1,3,1,1).unsqueeze(2).float())
                align_loss = self.loss_func(F.softmax(align_output, dim=1), y1[:,1].squeeze(1))
                loss += align_loss
                '''
                optim.zero_grad()
                loss.backward()
                if phase == 'train':
                    optim.step()

                    if i_batch % self.log_nth == 0:
                        self.logWriter.loss_per_iter(
                            loss.item(), i_batch, current_iteration)
                        # print('bloss loss: ', loss_b)
#                        print('cond loss: ', loss_cond/5*0.3)
                    current_iteration += 1

                if phase == 'val':
                    if i_batch != len(data_loader[phase]) - 1:
                        # print("#", end='', flush=True)
                        pass
                    else:
                        print("100%", flush=True)
            if phase == 'train' and epoch % 10 ==0:
                self.logWriter.log('saving checkpoint ....')
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'start_iteration': current_iteration + 1,
                    'arch': self.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    #'scheduler_c': scheduler_c.state_dict(),
                    #'optimizer_s': optim_s.state_dict(),
                    'best_ds_mean_epoch': self.best_ds_mean_epoch,
                    #'scheduler_s': scheduler_s.state_dict()
                }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

                self.logWriter.log(
                    "==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            self.logWriter.log('FINISH.')
        self.logWriter.close()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def save_best_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        print("Best Epoch... " + str(self.best_ds_mean_epoch))
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

    def load_checkpoint(self, epoch=None):
        if epoch is not None:
            checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                           'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION)
            self._load_checkpoint_file(checkpoint_path)
        else:
            all_files_path = os.path.join(
                self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
            list_of_files = glob.glob(all_files_path)
            if len(list_of_files) > 0:
                checkpoint_path = max(list_of_files, key=os.path.getctime)
                self._load_checkpoint_file(checkpoint_path)
            else:
                self.logWriter.log(
                    "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))

    def load_stn_checkpoint(self, epoch=None):
        checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                       'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        # for k in state_dict.keys():
        #     print(k)
        part_sd = {k: v for k, v in state_dict.items() if 'segmentor.unet' in k}
        # print(part_sd.keys())
        self.model.state_dict().update(part_sd)
        for p in self.model.segmentor.unet.parameters():
            #  print(p)
            p.requires_grad = False
        return 0



    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        self.best_ds_mean_epoch = checkpoint['best_ds_mean_epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])

        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)


        self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(
            file_path, checkpoint['epoch']))
