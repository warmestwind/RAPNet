import torch
import torch.nn.functional as F

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float() # shape = 2,h,w
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def Correlation(fmap1, fmap2, r=3):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())
    corr = corr.view(-1, 1, ht, wd)
    # corr = F.adaptive_avg_pool2d(corr, (64, 64))
    # corr = corr.view(batch, ht, wd, -1)
    # corr = corr.permute(0, 3, 1, 2).contiguous()

    coords = coords_grid(batch, ht, wd).to(fmap1.device)
    coords = coords.permute(0, 2, 3, 1)
    batch, h1, w1, _ = coords.shape
    dx = torch.linspace(-r, r, 2 * r + 1)
    dy = torch.linspace(-r, r, 2 * r + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

    centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl

    corr = bilinear_sampler(corr, coords_lvl)
    corr = corr.view(batch, h1, w1, -1)
    out = corr.permute(0, 3, 1, 2).contiguous().float()

    return out

if __name__ == '__main__':
    fmap1 = torch.ones(1,1,3,3)
    fmap2 = torch.ones(1, 1, 3, 3)
    out = Correlation(fmap1, fmap2)
    print(out.shape)
