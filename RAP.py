import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import modules.conv_blocks as cm
import torch.nn.functional as F
from modules.voxelmorph_us import U_Network, SpatialTransformer
from modules.cre import Correlation


class SDnetSegmentor(nn.Module):
    """
    Segmentor Code
    """
    def __init__(self, params):
        super(SDnetSegmentor, self).__init__()
        params['num_channels'] = 3
        params['num_filters'] = 16
        self.encode1 = cm.SDnetEncoderBlock(params)
        params['num_channels'] = 16
        params['num_filters'] = 32
        self.encode2 = cm.SDnetEncoderBlock(params)
        params['num_channels'] = 32
        params['num_filters'] = 64
        self.encode3 = cm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        params['num_filters'] = 64
        self.encode4 = cm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        params['num_filters'] = 64
        self.bottleneck = cm.GenericBlock(params)

        params['num_channels'] = 128
        params['num_filters'] = 32
        self.decode3 = cm.SDnetDecoderBlock(params)
        params['num_channels'] = 128
        params['num_filters'] = 64
        self.decode4 = cm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        params['num_class'] = 1
        self.classifier = cm.ClassifierBlock(params)
        params['num_channels'] = 1

        cood_conv = cm.CoordConv(16*3, 1, kernel_size=3, padding=1)  #24
        self.soft_max = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.mask_conv = nn.Sequential(cood_conv,  nn.Sigmoid())

        self.unet = U_Network(2, [16, 32, 32, 32], [32, 32, 32, 32, 8, 8])
        self.stn = SpatialTransformer((256, 256))

        self.conv00 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        self.conv22 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 3),
            nn.ReLU(inplace=True)
        )


        self.q4 = nn.Sequential(
            nn.Conv2d(64 + (3 * 2 + 1) ** 2, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.q3 = nn.Sequential(
            nn.Conv2d(64 + (3 * 2 + 1) ** 2, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.q2 = nn.Sequential(
            nn.Conv2d(32 + (3 * 2 + 1) ** 2, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.af_4 = cm.AF(128, 16, 3, 1, 1)
        self.af_3 = cm.AF(128, 16, 5, 1, 2)
        self.af_2 = cm.AF(64, 16, 7, 1, 3)

        self._init_weights()

    def seg_branch_encoder(self, input):
        e1, _, ind1 = self.encode1(input)
        e2, _, ind2 = self.encode2(e1)
        e3, _, ind3 = self.encode3(e2)
        e4, out4, ind4 = self.encode4(e3)
        bn = self.bottleneck(e4)

        return bn, ind4, ind3, ind2, ind1, e4, e3, e2, e1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # def forward(self, inpt, weights=None, inpt_sp=None, inpt_mask=None, weights_pure=None, base_map=None, sp_prior_test=None, query_mask=None):
    def forward(self, inpt, inpt_sp=None, inpt_mask=None):


        bn, ind4, ind3, ind2, ind1, e4, e3, e2, e1 = self.seg_branch_encoder(inpt)


        if inpt_sp is not None and len(inpt_sp.shape) == 5:
            tmp_sp_poriors = []
            tmp_sp_img_poriors = []
            for i in range(inpt_sp.shape[0]):

                flow = self.unet(inpt_sp[i][:, :3], inpt)

                tmp_sp_porior = self.stn(inpt_mask[i][:, 1:2, ...], flow)
                tmp_sp_img_porior = self.stn(inpt_sp[i][:, 1:2, ...], flow)

                tmp_sp_poriors.append(tmp_sp_porior)
                tmp_sp_img_poriors.append(tmp_sp_img_porior)


        sp_prior = torch.cat(tmp_sp_poriors, 1).mean(1, keepdim=True)
        sp_img_prior = torch.cat(tmp_sp_img_poriors, 1).mean(1, keepdim=True)

        # sp_prior = inpt_mask[:, :, 1:2, :, :].permute(1, 0, 2, 3, 4).mean(1)

        if inpt_sp is not None and len(inpt_sp.shape) == 5:
            # k shot atten
            sp_feats_fg_e4 = []
            sp_feats_bg_e4 = []
            sp_feats_fg_e3 = []
            sp_feats_bg_e3 = []
            sp_feats_fg_e2 = []
            sp_feats_bg_e2 = []

            for i in range(inpt_sp.shape[0]):
                bn, ind4, ind3, ind2, ind1, e4_sp, e3_sp, e2_sp, e1_sp = self.seg_branch_encoder(inpt_sp[i][:, :3])

                sp_level_features = e4_sp
                sp_mask = F.interpolate(inpt_mask[i][:, 1:2, ...], size=(sp_level_features.shape[-2:]), mode='nearest')

                corr_fg = Correlation(sp_level_features * sp_mask, sp_level_features * (1 - sp_mask))
                corr_bg = Correlation(sp_level_features * (1 - sp_mask), sp_level_features * sp_mask)
                corr_fg = self.q4(torch.cat([corr_fg, sp_level_features * sp_mask], dim=1))
                corr_bg = self.q4(torch.cat([corr_bg, sp_level_features * (1 - sp_mask)], dim=1))


                fore_avg_feat = torch.sum(torch.sum(corr_fg * sp_mask, dim=3), dim=2) / (torch.sum(sp_mask) + torch.tensor(1e-10).to(sp_mask.device))
                bg_avg_feat = torch.sum(torch.sum(corr_bg * (1 - sp_mask), dim=3), dim=2) / torch.sum(1 - sp_mask)

                sp_feats_fg_e4.append(fore_avg_feat)
                sp_feats_bg_e4.append(bg_avg_feat)

                sp_level_features = e3_sp
                sp_mask = F.interpolate(inpt_mask[i][:, 1:2, ...], size=(sp_level_features.shape[-2:]), mode='nearest')

                # aug each level features
                corr_fg = Correlation(sp_level_features * sp_mask, sp_level_features * (1 - sp_mask))
                corr_bg = Correlation(sp_level_features * (1 - sp_mask), sp_level_features * sp_mask)
                corr_fg = self.q3(torch.cat([corr_fg, sp_level_features * sp_mask], dim=1))
                corr_bg = self.q3(torch.cat([corr_bg, sp_level_features * (1 - sp_mask)], dim=1))

                fore_avg_feat = torch.sum(torch.sum(corr_fg * sp_mask, dim=3), dim=2) /  (torch.sum(sp_mask) + torch.tensor(1e-10).to(sp_mask.device))
                bg_avg_feat = torch.sum(torch.sum(corr_bg * (1 - sp_mask), dim=3), dim=2) / torch.sum(1 - sp_mask)

                sp_feats_fg_e3.append(fore_avg_feat)
                sp_feats_bg_e3.append(bg_avg_feat)

                #  repeat
                sp_level_features = e2_sp
                sp_mask = F.interpolate(inpt_mask[i][:,1:2,...], size=(sp_level_features.shape[-2:]), mode='nearest')

                # aug each level features
                corr_fg = Correlation(sp_level_features * sp_mask, sp_level_features * (1 - sp_mask))
                corr_bg = Correlation(sp_level_features * (1 - sp_mask), sp_level_features * sp_mask)
                corr_fg = self.q2(torch.cat([corr_fg, sp_level_features * sp_mask], dim=1))
                corr_bg = self.q2(torch.cat([corr_bg, sp_level_features * (1 - sp_mask)], dim=1))

                fore_avg_feat = torch.sum(torch.sum(corr_fg * sp_mask, dim=3), dim=2) /  (torch.sum(sp_mask) + torch.tensor(1e-10).to(sp_mask.device))
                bg_avg_feat = torch.sum(torch.sum(corr_bg * (1 - sp_mask), dim=3), dim=2) / torch.sum(1 - sp_mask)

                sp_feats_fg_e2.append(fore_avg_feat)
                sp_feats_bg_e2.append(bg_avg_feat)


            sp_feat_fg_e4 = torch.mean(torch.stack(sp_feats_fg_e4), 0).unsqueeze(-1)
            sp_feat_bg_e4 = torch.mean(torch.stack(sp_feats_bg_e4), 0).unsqueeze(-1)

            sp_feat_fg_e3 = torch.mean(torch.stack(sp_feats_fg_e3), 0).unsqueeze(-1)
            sp_feat_bg_e3 = torch.mean(torch.stack(sp_feats_bg_e3), 0).unsqueeze(-1)

            sp_feat_fg_e2 = torch.mean(torch.stack(sp_feats_fg_e2), 0).unsqueeze(-1)
            sp_feat_bg_e2 = torch.mean(torch.stack(sp_feats_bg_e2), 0).unsqueeze(-1)

            q_level_features = e4
            # aug query each  level features
            sp_prior_r = F.interpolate(sp_prior, size=(q_level_features.shape[-2:]), mode='nearest')

            corr_fg = Correlation(q_level_features *sp_prior_r, q_level_features * (1 - sp_prior_r))
            corr_bg = Correlation(q_level_features * (1 - sp_prior_r), q_level_features * sp_prior_r)

            q_level_features_fg = self.q4(torch.cat([corr_fg, q_level_features * sp_prior_r], dim=1))
            q_level_features_bg = self.q4(torch.cat([corr_bg, q_level_features * (1-sp_prior_r)], dim=1))

            q_b, q_n, q_h, q_w = q_level_features_fg.shape
            q_level_features_fg = q_level_features_fg.view(q_b, q_n, -1)
            q_level_features_bg = q_level_features_bg.view(q_b, q_n, -1)

            correlative_map_fg = F.cosine_similarity(q_level_features_fg, sp_feat_fg_e4)
            correlative_map_fg_e4 = correlative_map_fg.view(q_b, 1, q_h, q_w)

            correlative_map_bg = F.cosine_similarity(q_level_features_bg, sp_feat_bg_e4)
            correlative_map_bg_e4 = correlative_map_bg.view(q_b, 1, q_h, q_w)

            q_level_features = e3
            # aug q
            sp_prior_r = F.interpolate(sp_prior, size=(q_level_features.shape[-2:]), mode='nearest')

            corr_fg = Correlation(q_level_features *sp_prior_r, q_level_features * (1 - sp_prior_r))
            corr_bg = Correlation(q_level_features * (1 - sp_prior_r), q_level_features * sp_prior_r)

            q_level_features_fg = self.q3(torch.cat([corr_fg, q_level_features * sp_prior_r], dim=1))
            q_level_features_bg = self.q3(torch.cat([corr_bg, q_level_features * (1-sp_prior_r)], dim=1))

            q_b, q_n, q_h, q_w = q_level_features_fg.shape
            q_level_features_fg = q_level_features_fg.view(q_b, q_n, -1)
            q_level_features_bg = q_level_features_bg.view(q_b, q_n, -1)


            correlative_map_fg = F.cosine_similarity(q_level_features_fg, sp_feat_fg_e3)
            correlative_map_fg_e3 = correlative_map_fg.view(q_b, 1, q_h, q_w)

            correlative_map_bg = F.cosine_similarity(q_level_features_bg, sp_feat_bg_e3)
            correlative_map_bg_e3 = correlative_map_bg.view(q_b, 1, q_h, q_w)

            _, max_corr_e3 = torch.max(self.soft_max(torch.cat([correlative_map_bg_e3, correlative_map_fg_e3], 1)), dim=1, keepdim=True)

            # repeat
            q_level_features = e2
            # aug q level
            sp_prior_r = F.interpolate(sp_prior, size=(q_level_features.shape[-2:]), mode='nearest')

            corr_fg = Correlation(q_level_features *sp_prior_r, q_level_features * (1 - sp_prior_r))
            corr_bg = Correlation(q_level_features * (1 - sp_prior_r), q_level_features * sp_prior_r)

            q_level_features_fg = self.q2(torch.cat([corr_fg, q_level_features * sp_prior_r], dim=1))
            q_level_features_bg = self.q2(torch.cat([corr_bg, q_level_features * (1-sp_prior_r)], dim=1))

            q_b, q_n, q_h, q_w = q_level_features_fg.shape
            q_level_features_fg = q_level_features_fg.view(q_b, q_n, -1)
            q_level_features_bg = q_level_features_bg.view(q_b, q_n, -1)


            correlative_map_fg = F.cosine_similarity(q_level_features_fg, sp_feat_fg_e2)
            correlative_map_fg_e2 = correlative_map_fg.view(q_b, 1, q_h, q_w)

            correlative_map_bg = F.cosine_similarity(q_level_features_bg, sp_feat_bg_e2)
            correlative_map_bg_e2 = correlative_map_bg.view(q_b, 1, q_h, q_w)


        sp_prior_e4 = F.interpolate(sp_prior, size=(e4.shape[-2:]), mode='nearest')
        attention_e4 = self.conv00(torch.cat([correlative_map_bg_e4, correlative_map_fg_e4, sp_prior_e4], 1))

        sp_prior_e3 = F.interpolate(sp_prior, size=(e3.shape[-2:]), mode='nearest')
        attention_e3 = self.conv11(torch.cat([correlative_map_bg_e3, correlative_map_fg_e3, sp_prior_e3], 1))

        sp_prior_e2 = F.interpolate(sp_prior, size=(e2.shape[-2:]), mode='nearest')
        attention_e2 = self.conv22(torch.cat([correlative_map_bg_e2, correlative_map_fg_e2, sp_prior_e2], 1))

        bn = torch.cat([bn, e4], 1)
        bn_sa = self.af_4(bn, attention_e4)
        d4 = self.decode4(bn, None, ind4)

        d4 = torch.cat([d4, e3], 1)
        d4_sa = self.af_3(d4, attention_e3)
        d3 = self.decode3(torch.cat([d4], 1), None, ind3)

        d3 = torch.cat([d3, e2], 1)
        d3_sa = self.af_2(d3, attention_e2)

        d4_sa = F.interpolate(d4_sa, size=(d3_sa.shape[-2:]), mode='nearest')
        bn_sa = F.interpolate(bn_sa, size=(d3_sa.shape[-2:]), mode='nearest')

        logit = self.mask_conv(torch.cat([bn_sa, d4_sa, d3_sa], 1))


        logit = F.interpolate(logit, size=(inpt.shape[-2:]), mode='nearest')


        max_corr = F.interpolate(max_corr_e3.float(), size=(inpt.shape[-2:]), mode='nearest').requires_grad_()


        return logit, sp_prior, max_corr


class RAP(nn.Module):

    def __init__(self, params):
        super(RAP, self).__init__()
        self.segmentor = SDnetSegmentor(params)

    def forward(self, input1, input_sp, input_mask):
        '''
        :param input1:
        :param input2:
        :return:
        '''
        segment = self.segmentor(input1, input_sp, input_mask)
        return segment

    @property
    def is_cuda(self):
        """
        Check if modules parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save modules with its parameters to the given path. Conventionally the
        path should end with "*.modules".

        Inputs:
        - path: path string
        """
        print('Saving modules... %s' % path)
        torch.save(self, path)


if __name__ == '__main__':

    import argparse
    from utils import config

    def get_configs():
        parser = argparse.ArgumentParser(description='PyTorch Few Shot Semantic Segmentation')
        parser.add_argument('--config', type=str, default='./settings.yaml', help='config file')
        parser.add_argument('--mode', '-m', default='train',
                            help='run mode, valid values are train and eval')
        parser.add_argument('--device', '-d', default=0,
                            help='device to run on')
        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)
        return cfg, args

    cfgs, args = get_configs()
    print(cfgs.DATA)
    common_params, data_params, net_params, train_params, eval_params = cfgs.COMMON, cfgs.DATA, cfgs.NETWORK, \
                                                                        cfgs.TRAINING, cfgs.EVAL

    few_shot_model = SDnetSegmentor(net_params)

    import torchinfo

    batch_size = 2
    shot = 1
    torchinfo.summary(few_shot_model, input_size=[(batch_size, 3, 256, 256), (shot, batch_size, 6, 256, 256),  (shot, batch_size, 3, 256, 256)])
