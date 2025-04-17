import os
import os.path
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import nibabel as nib
import nrrd
import cv2



def make_dataset(sub_list):
    '''
    :param sub_list:  specify cls index as train or val set , here can be [LV(2,3,4) RV(4,F) LA(2,3,4) LAA]
    :return:
    '''
    # same view cannot in train and val set

    data_dict = {'LV2': r'D:\Dataset\PH_30_UNET\PH_LV_A2C_TRAIN',
                 'LV3': r'D:\Dataset\PH_30_UNET\PH_LV_A3C_TRAIN',
                 'LV4': r'D:\Dataset\PH_30_UNET\PH_LV_A4C_TRAIN',
                 'LA2': r'D:\Dataset\PH_30_UNET\PH_LA_A2C_TRAIN',
                 'LA3': r'D:\Dataset\PH_30_UNET\PH_LA_A3C_TRAIN',
                 'LA4': r'D:\Dataset\PH_30_UNET\PH_LA_A4C_TRAIN',
                 }



    # same view cannot in train and val set
    # data_dict = {'LV2': '/home/fengyong/Dataset/PH_30/PH_LV_A2C',
    #              'LV3': '/home/fengyong/Dataset/PH_30/PH_LV_A3C',
    #              'LV4': '/home/fengyong/Dataset/PH_30/PH_LV_A4C',
    #              'RV4': '/home/fengyong/Dataset/PH_30/RV',
    #              'LA2': '/home/fengyong/Dataset/PH_30/PH_LA_A2C',
    #              'LA3': '/home/fengyong/Dataset/PH_30/PH_LA_A3C',
    #              'LA4': '/home/fengyong/Dataset/PH_30/PH_LA_A4C',
    #              'LAA': '/home/fengyong/Dataset/PH_30/LAA',
    #              }

    print(f"Processing data...{sub_list}")

    image_label_list = []
    sub_class_file_list = {}
    # for sub_c in sub_list:
    #     sub_class_file_list[sub_c] = []
    global_cls_id = 1
    total_cls = []
    for c in sub_list:
        root_c = data_dict[c]
        pids_pair = sorted(os.listdir(root_c))
        temp_cls = total_cls
        for idx in range(0, len(pids_pair), 2):
            pid_img = pids_pair[idx]
            print(pid_img)
            pid_lab = pids_pair[idx+1]
            image_name = os.path.join(root_c, pid_img)
            # image_name = [name for name in img_paths if os.path.isfile(name)][0]

            label_name = os.path.join(root_c, pid_lab)
            # label_name = [name for name in msk_paths if os.path.isfile(name)][0]
            item = (image_name, label_name, c)

            if temp_cls == total_cls:
                # load label check cls
                if os.path.splitext(label_name)[-1] == '.gz':
                    mask_nib = nib.load(label_name)
                    label = mask_nib.get_data().transpose(2, 1, 0)
                else:
                    label = nrrd.read(label_name)[0].transpose((2,1,0))

                label_class = np.unique(label).tolist()
                if 0 in label_class:
                    label_class.remove(0)
                temp_cls = [c+'_'+str(lc) for lc in label_class]

            if len(label_class) > 0:
                image_label_list.append(item)

            for t_c in temp_cls:
                    if t_c not in sub_class_file_list.keys():
                        sub_class_file_list[t_c] = []
                    sub_class_file_list[t_c].append(item)

        total_cls += temp_cls
        np.savez('train_la_list.npz', img_lab=image_label_list, cls_file=sub_class_file_list)
    print('DOne make data...')
    return image_label_list, sub_class_file_list


class SemData_US(Dataset):
    def __init__(self, shot=1, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.shot = shot
        self.sub_list = ['LV2', 'LV3', 'LV4'] # ['LA2', 'LA3', 'LA4', 'LAA']
        # self.sub_list = ['LA2', 'LA3', 'LA4']
        # self.sub_list = ['LV2', 'LV3', 'LV4', 'RV4', 'LAA']
        self.sub_val_list = ['LV2']
        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(self.sub_list)
            print('Load Train Set')
            # file_list = np.load('./coco_train_list.npz', allow_pickle=True)
            # self.data_list = list(file_list['img_lab'])
            # self.sub_class_file_list = file_list['cls_file'].item()

            # assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            print('Load Val Set')
            self.data_list, self.sub_class_file_list = make_dataset(self.sub_val_list)
            # file_list = np.load('./coco_val_list.npz', allow_pickle=True)
            # self.data_list = list(file_list['img_lab'])
            # self.sub_class_file_list = file_list['cls_file'].item()
            # assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        self.transform = transform
        self.slice_num = 3
        self.w_ = (self.slice_num-1)//2

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path, mode_name = self.data_list[index]
        query_path = image_path

        # if 'LA_A2C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('la2.npy'))
        # if 'LA_A3C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('la3.npy'))
        # if 'LA_A4C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('la4.npy'))
        # if 'LV_A2C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('lv2.npy'))
        # if 'LV_A3C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('lv3.npy'))
        # if 'LV_A4C' in image_path:
        #     choosen_cls = 0
        #     sp_prior = torch.from_numpy(np.load('lv4.npy'))
        # if 'RV' in image_path:
        #     choosen_cls = 1
        #     sp_prior = torch.from_numpy(np.load('rv4.npy'))
        # if 'LAA' in image_path:
        #     choosen_cls = 2
        #     sp_prior = torch.from_numpy(np.load('laa.npy'))

        # load label check cls

        if os.path.splitext(label_path)[-1] == '.gz':
            mask_nib = nib.load(label_path)
            label = mask_nib.get_data().transpose(2, 1, 0)
        else:
            label = nrrd.read(label_path)[0].transpose((2, 1, 0))

        image = nrrd.read(image_path)[0].transpose((2,1,0))
        # slice_idx = random.randint(0, image.shape[0]-1)  #[a,b]
        slice_idx = random.randint(self.w_, image.shape[0] - 1 - self.w_)  # [a,b]
        # image = image[slice_idx].astype(np.float32)
        image = image[slice_idx - self.w_:slice_idx + self.w_ + 1].astype(np.float32)
        # image = np.repeat(image[slice_idx][..., np.newaxis], 3, -1)
        label = label[slice_idx - self.w_:slice_idx + self.w_ + 1].astype(np.uint8)
        raw_label = label.copy()

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        # print('cls unique', label_class)

        if 0 in label_class:
            label_class.remove(0)

        if len(label_class) == 0:
            print('lb = 0,', image_path, slice_idx)
            label_class = [1]
            # assert len(label_class) > 0

        chosen_idx = random.randint(1, len(label_class)) - 1
        class_chosen = label_class[chosen_idx]
        # class_chosen = class_chosen
        # if self.mode == 'val':
        #     print('chosen class:', class_chosen, 'from: ', label_class, query_path)
        target_pix = np.where(label == class_chosen)

        if target_pix[0].shape[0] > 0 and len(target_pix)==3:
            label[:, :, :] = 0
            label[target_pix[0], target_pix[1], target_pix[2]] = 1
        elif len(target_pix)==2:
            label[:, :] = 0
            label[target_pix[0], target_pix[1]] = 1

        chosen_mode_name = mode_name + '_' + str(class_chosen)
        file_class_chosen = self.sub_class_file_list[chosen_mode_name]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while ((
                           support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path, mode = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)
        # print('sup list:', support_image_path_list)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):
            if self.mode == 'train':
                # subcls_list.append(self.sub_list.index(class_chosen))
                subcls_list.append(chosen_mode_name)
            else:
                # subcls_list.append(self.sub_val_list.index(class_chosen))
                subcls_list.append(chosen_mode_name)
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            # print(support_image_path, support_label_path)

            support_image = nrrd.read(support_image_path)[0].transpose((2, 1, 0))
            # slice_idx = random.randint(0, support_image.shape[0] - 1)  # [a,b]
            slice_idx = random.randint(0+self.w_, support_image.shape[0] - 1 -self.w_)  # [a,b]
            # support_image = np.repeat(support_image[slice_idx][..., np.newaxis], 3, -1).astype(np.float32)
            support_image = support_image[slice_idx-self.w_:slice_idx+self.w_+1].astype(np.float32)

            # load label check cls
            if os.path.splitext(support_label_path)[-1] == '.gz':
                mask_nib = nib.load(support_label_path)
                support_label = mask_nib.get_data().transpose(2, 1, 0)
            else:
                support_label = nrrd.read(support_label_path)[0].transpose((2, 1, 0))
            # support_label = support_label[slice_idx].astype(np.uint8)
            support_label = support_label[slice_idx - self.w_: slice_idx + self.w_ + 1].astype(np.uint8)

            target_pix = np.where(support_label == class_chosen)
            # ignore_pix = np.where(support_label == 255)
            support_label[:, :, :] = 0
            support_label[target_pix[0], target_pix[1], target_pix[2]] = 1
            # support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        flip = (random.random() > 0.5)
        if self.transform is not None:
            trans_image = []
            trans_label = []
            if flip:
                for i in range(self.slice_num):
                    image[i] = cv2.flip(image[i], 1)
                    label[i] = cv2.flip(label[i], 1)
            for i in range(self.slice_num):
                trans_image_i, trans_label_i = self.transform(image[i], label[i])
                trans_image.append(trans_image_i)
                trans_label.append(trans_label_i)
                # image, label = self.transform(image, label)

            image = torch.stack(trans_image)
            label = torch.stack(trans_label).unsqueeze(1)


            for k in range(self.shot):
                trans_image_si = []
                trans_label_si = []
                if flip:
                    for i in range(self.slice_num):
                        support_image_list[k][i] = cv2.flip(support_image_list[k][i], 1)
                        support_label_list[k][i] = cv2.flip(support_label_list[k][i], 1)
                for i in range(self.slice_num):
                    support_image_ki, support_label_ki = self.transform(support_image_list[k][i],
                                                                        support_label_list[k][i])

                    trans_image_si.append(support_image_ki)
                    trans_label_si.append(support_label_ki.unsqueeze(0))


                support_image_list[k] = torch.stack(trans_image_si)
                support_label_list[k] = torch.stack(trans_label_si)

        s_xs = torch.stack(support_image_list, 0)
        s_ys = torch.stack(support_label_list, 0)

        # sp_prior = torch.mean(s_ys.float(), 0)  # slice 1 256, 256

        if self.mode == 'train':
            return image, label, s_xs, s_ys, subcls_list#, sp_prior, choosen_cls
        else:
            return image, label, s_xs, s_ys, subcls_list, raw_label, query_path, support_image_path_list

        # model: s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda()
        # raw_label: 0-1 2D
        # image: 3D cv2 read query image
        # label: 2D cv2 read label image
        # s_x : [1, n, 3, h, w]
        # s_x : [1, n, h, w]
        # subcls_liit : [chosen cls]


if __name__ == '__main__':

    # l1, d1 = make_dataset(['LV4', 'LAA'])
    # print(len(l1))
    # print(d1.keys())


    # import shutil
    # root = r'D:\Dataset\NRRD\Train'
    # tar = r'D:\Dataset\NRRD\BI_V_ALL'
    # pids = os.listdir(root)
    #
    # for pid in pids:
    #     ori_img = os.path.join(root, pid, 'im.nrrd')
    #     tar_img = os.path.join(tar, f'{pid}_im.nrrd')
    #
    #     ori_msk = os.path.join(root, pid, 'm.nrrd')
    #     tar_msk = os.path.join(tar, f'{pid}_m.nrrd')
    #     shutil.copyfile(ori_img, tar_img)
    #     shutil.copyfile(ori_msk, tar_msk)


    import transform

    # value_scale = 255
    # mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    # std = [0.229, 0.224, 0.225]
    # std = [item * value_scale for item in std]
    # train_transform = [
    #     # transform.RandScale([0.8, 1.25]),
    #     # transform.RandRotate([-10, 10], padding=[0,0,0], ignore_label=0),
    #     # transform.RandomGaussianBlur(),
    #     # transform.RandomHorizontalFlip(),
    #     transform.Crop([641, 641], crop_type='rand', padding=mean, ignore_label=255),
    #     # transform.Resize([256, 256]),
    #     transform.ToTensor()]
    #     # transform.Normalize(mean=mean, std=std)]
    # train_transform = transform.Compose(train_transform)
    #
    # data = SemData_US(shot=5, transform=train_transform, mode='val')
    #
    train_transform = [
        # transform.RandScale([0.8, 1.25]),
        # transform.RandRotate([-10, 10], padding=mean, ignore_label=0),
        transform.RandomGaussianBlur(),
        # transform.RandomHorizontalFlip(),
        # transform.Crop([256, 256], crop_type='rand', padding=mean, ignore_label=0),
        transform.Resize_fy([256, 256]),
        transform.ToTensor(),
        transform.Scaling()]
    train_transform = transform.Compose(train_transform)
    train_data = SemData_US(shot=1,  transform=train_transform, mode='train')
    train_sampler = None
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=(train_sampler is None),
                                               num_workers=1, pin_memory=True, sampler=train_sampler,
                                               drop_last=False)

    for i in trainloader:
        image, label, s_xs, s_ys, subcls_list, sp_prior = i
        break
    # image, label, s_xs, s_ys, subcls_list, raw_label, query_path, support_image_path_list = data[0]

    print(image.shape, label.shape, s_xs.shape, s_ys.shape)
    print(subcls_list)
    #
    nrrd.write('t.nrrd', s_xs.squeeze().cpu().numpy().transpose(2,1,0))
    nrrd.write('tl.nrrd', s_ys.squeeze().cpu().numpy().transpose(2, 1, 0).astype(np.uint8))

