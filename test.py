import numpy as np
import torch.nn as nn
import torch
import nrrd
import cv2
import torch.nn.functional as F
import random
import os
import pandas as pd
import numpy
import copy
import RAP as fs
from settings import Settings
import shutil



support_path = r'your dir'
global_path = r'your dir'


def MR_normalize(x_in):
    # return (x_in - x_in.mean()) / x_in.std()
    # return (x_in - np.min(x_in)) / (np.max(x_in) - np.min(x_in))
    return x_in/255

def ts_main(ckpt_path):

    settings = Settings() # parse .ini
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    model = fs.RAP(net_params)

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.cuda()
    model.eval()

    # some params
    query_root = global_path
    shot = 5
    size = 256
    all_img_path = glob.glob(query_root+'/*_im.nrrd')
    all_support_path = glob.glob(support_path+'/*_im.nrrd')

    save_path = './prediction_la_dice_1000'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    # data flow and pred
    with torch.no_grad():
        for pid in all_img_path:
            print('qid:', pid)
            query_name = pid.split('\\')[-1].split('.')[0]
            # if query_name != '08-63 WANGQIAN_im':
            #     continue
            img_query = nrrd.read(pid)[0].transpose(2, 1, 0)
            mask_query = nrrd.read(pid.replace('im', 'm'))[0].transpose(2, 1, 0)

            tmp_support_path = copy.deepcopy(all_support_path)
            try:
                tmp_support_path.remove(pid)
            except:
                pass

            pred_mask = []
            tmp_sprior = []
            sp_mask = []
            sp_slices = 3
            for query_slice in range(img_query.shape[0]):
                if sp_slices == 1:
                    input = cv2.resize(img_query[query_slice], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                    input = MR_normalize(input)
                    # 3 or 1 channel input
                    # input = torch.from_numpy(np.repeat(input[np.newaxis, np.newaxis, ...], 3, 1)).float().cuda()
                    query = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float().cuda()

                else:
                    # sp_slices == 3
                    input = cv2.resize(img_query[query_slice], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                    input = MR_normalize(input)
                    query = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float()
                    if query_slice == 0:
                        query_pre = query
                    else:
                        input = cv2.resize(img_query[query_slice-1], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                        input = MR_normalize(input)
                        query_pre = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float()
                    if query_slice == img_query.shape[0]-1:
                        query_next = query
                    else:
                        input = cv2.resize(img_query[query_slice+1], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                        input = MR_normalize(input)
                        query_next = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float()
                    # finish read query img(1 or 3 slices) and mask (1 slice)
                    query = torch.cat([query_pre, query, query_next], dim=1).cuda()
                    mask_query = cv2.resize(mask_query[query_slice], dsize=(size, size), interpolation=cv2.INTER_NEAREST)


                # every slice a support
                support_paths = random.sample(tmp_support_path, shot)
                print('sids:', support_paths)
                sp_imgs = []
                sp_msks = []
                for i in range(shot):
                    img_support = nrrd.read(support_paths[i])[0].transpose(2, 1, 0)
                    mask_support = nrrd.read(support_paths[i].replace('_im', '_m'))[0].transpose(2, 1, 0).astype(
                        np.uint8)
                    sp_imgs.append(img_support)
                    sp_msks.append(mask_support)

                # get cur_slice support
                s_inputs = []
                s_masks = []
                cond_inputs = []
                for i in range(shot):
                    img_support = sp_imgs[i]
                    mask_support = sp_msks[i]
                    sp_shp0 = img_support.shape[0]
                    if sp_slices == 1:
                        sp_index = int(query_slice/img_query.shape[0]*img_support.shape[0])
                        img_support = cv2.resize(img_support[sp_index], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                        img_support = MR_normalize(img_support)
                        s_input = torch.from_numpy(img_support[np.newaxis, np.newaxis, np.newaxis,...]).float().cuda()
                        msk_support = cv2.resize(mask_support[sp_index], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                        s_mask = torch.from_numpy(msk_support[np.newaxis, np.newaxis, np.newaxis, ...]).float().cuda()
                    else:
                        # S1
                        # sp_index = sp_shp0//2

                        # S2
                        # bias = sp_shp0 / 3 / 2
                        # ratio = query_slice / img_query.shape[0]
                        # if ratio < 1 / 3:
                        #     sp_index = int(bias)
                        # elif ratio >= 1 / 3 and ratio < 2 / 3:
                        #     sp_index = int(1 / 3 * sp_shp0 + bias)
                        # else:
                        #     sp_index = int(2 / 3 * sp_shp0 + bias)

                        # S3
                        sp_index = int(query_slice / img_query.shape[0] * sp_shp0)


                        sp_indexes= [max(sp_index-1, 0), sp_index, min(sp_index+1, sp_shp0-1)]
                        sp_imgs_tmp = []
                        sp_masks_tmp =[]
                        for sp_index in sp_indexes:
                            img_support_r = cv2.resize(img_support[sp_index], dsize=(size, size),
                                                     interpolation=cv2.INTER_LINEAR)
                            img_support_r = MR_normalize(img_support_r)
                            s_input = torch.from_numpy(img_support_r[np.newaxis, np.newaxis, np.newaxis, ...]).float().cuda()
                            sp_imgs_tmp.append(s_input)
                            msk_support = cv2.resize(mask_support[sp_index], dsize=(size, size),
                                                     interpolation=cv2.INTER_NEAREST)==1
                            s_mask = torch.from_numpy(msk_support[np.newaxis, np.newaxis, np.newaxis, ...]).float().cuda()
                            sp_masks_tmp.append(s_mask)

                        s_input = torch.cat(sp_imgs_tmp, 2)  # [1,1,slice,H,W]
                        s_mask = torch.cat(sp_masks_tmp, 2)  # [1,1,slice,H,W]

                    s_inputs.append(s_input)
                    s_masks.append(s_mask)

                # finish read support img and mask
                s_input = torch.cat(s_inputs, 1)  # 1, Kshot, slice, h, w
                s_mask = torch.cat(s_masks, 1)

                # # run model
                support = torch.cat([s_input, s_mask], 2)  # b, Kshot, slice*2, h, w
                cond_inputs_ = support.permute(1,0,2,3,4)  # Kshot, b, slice*2, h, w

                # forward
                out, sp_pred, max_corr2 = model.segmentor(query, cond_inputs_, s_mask.permute(1, 0, 2, 3, 4))

                tmp_sprior.append(out.detach().cpu().numpy())
                out = F.interpolate(out, size=img_query.shape[1:], mode='bilinear', align_corners=True)
                sp_pred = F.interpolate(sp_pred, size=img_query.shape[1:], mode='bilinear', align_corners=True)

                output = (out >.5).squeeze(1)
                sp_pred = (sp_pred > .5).squeeze(1)

                pred_mask.append(output.cpu().numpy())
                sp_mask.append(sp_pred.squeeze(1).cpu().numpy())

            pred = np.concatenate(pred_mask, 0)
            sp = np.concatenate(sp_mask, 0)

            nrrd.write(f'{save_path}/{query_name}_pred.nrrd', pred.transpose(2,1,0).astype(np.uint8))
            nrrd.write(f'{save_path}/{query_name}_sp.nrrd', sp.transpose(2,1,0).astype(np.uint8))

