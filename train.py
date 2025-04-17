import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# from dataloaders_medical.prostate import *
# import few_shot_segmentor as fs
import RAP as fs
# import v9_only_pro as fs
# import v9_only_stn as fs
# import v9_pro_stn as fs
# import v9_decoder as fs
# import lba_cbam_v9_noskip as fs
from settings import Settings
from solver_us_pro import Solver
# from solver import Solver
# from solver_se import Solver
from dataset import transform, dataset_us
from utils import config

torch.set_default_tensor_type('torch.FloatTensor')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_configs():
    parser = argparse.ArgumentParser(description='PyTorch Few Shot Semantic Segmentation')
    parser.add_argument('--config', type=str, default='./settings.yaml', help='config file')
    parser.add_argument('--mode', '-m',  default='train',
                        help='run mode, valid values are train and eval')
    parser.add_argument('--device', '-d', default=0,
                        help='device to run on')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg, args  # cfg.xxx


def train(train_params, common_params, data_params, net_params):

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
    train_data = dataset_us.SemData_US(shot=1,  transform=train_transform, mode='train')
    train_sampler = None
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=(train_sampler is None),
                                               num_workers=1, pin_memory=True, sampler=train_sampler,
                                               drop_last=False)

    validationloader = trainloader



    final_model_path = os.path.join(
        common_params['save_model_dir'],  'last.pth.tar')

    few_shot_model = fs.RAP(net_params)


    solver = Solver(few_shot_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "weight_decay": train_params['optim_weight_decay'],
                                "momentum": train_params['momentum']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'])

    solver.train(trainloader, validationloader)
   # solver.save_best_model(final_model_path)
    print("final model saved @ " + str(final_model_path))





if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', '-m',  default='train',
    #                     help='run mode, valid values are train and eval')
    # parser.add_argument('--device', '-d', default=0,
    #                     help='device to run on')
    # args = parser.parse_args()
    #
    # settings = Settings() # parse .ini
    # common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
    #     'NETWORK'], settings['TRAINING'], settings['EVAL']
    #
    # if args.device is not None:
    #     common_params['device'] = args.device
    #
    # if args.mode == 'train':
    #     train(train_params, common_params, data_params, net_params)
    # elif args.mode == 'eval':
    #     pass
    # else:
    #     raise ValueError(
    #         'Invalid value for mode. only support values are train and eval')

    cfgs, args = get_configs()
    print(cfgs.DATA)
    common_params, data_params, net_params, train_params, eval_params = cfgs.COMMON, cfgs.DATA, cfgs.NETWORK, \
                                                                        cfgs.TRAINING, cfgs.EVAL

    if args.device is not None:
        common_params['device'] = args.device

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
