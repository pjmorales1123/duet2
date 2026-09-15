"""Small custom RGB-only keypoint network for the bottle feedback experiment."""
import torch
from torch import nn
import torch.nn.functional as F


def block(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                         nn.GroupNorm(4, out_channels), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                         nn.GroupNorm(4, out_channels), nn.SiLU())


class BottleKeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = block(3, 16, 2)
        self.enc2 = block(16, 32, 2)
        self.enc3 = block(32, 64, 2)
        self.enc4 = block(64, 96, 2)
        self.dec3 = block(160, 64)
        self.dec2 = block(96, 32)
        self.dec1 = block(48, 24)
        self.heatmaps = nn.Conv2d(24, 2, 1)
        self.mask = nn.Conv2d(24, 1, 1)
        self.visibility = nn.Linear(192, 1)

    def forward(self, image):
        a = self.enc1(image*2-1)
        b = self.enc2(a)
        c = self.enc3(b)
        d = self.enc4(c)
        up = lambda value, skip: F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([up(d, c), c], 1))
        x = self.dec2(torch.cat([up(x, b), b], 1))
        x = self.dec1(torch.cat([up(x, a), a], 1))
        pooled = torch.cat([d.mean((2, 3)), d.amax((2, 3))], 1)
        return self.heatmaps(x), self.mask(x), self.visibility(pooled).squeeze(1)


def pixel_grid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    # Centers of stride-two cells in input pixel coordinates.
    return torch.stack([x*2+.5, y*2+.5], -1).float()


def decode_heatmaps(logits):
    batch, points, height, width = logits.shape
    probabilities = logits.flatten(2).softmax(-1).reshape_as(logits)
    grid = pixel_grid(height, width, logits.device)
    coordinates = (probabilities[..., None]*grid).sum((2, 3))
    peak = probabilities.flatten(2).amax(-1)
    variance = (probabilities[..., None]*(grid[None, None]-coordinates[:, :, None, None])**2).sum((2, 3, 4))
    return coordinates, peak, variance


def observation_loss(outputs, keypoints, masks, visible):
    heatmaps, foreground, visibility = outputs
    grid = pixel_grid(*heatmaps.shape[-2:], heatmaps.device)
    distance = ((grid[None, None]-keypoints[:, :, None, None])**2).sum(-1)
    target = torch.exp(-distance/(2*2.4**2))
    target = target/(target.sum((2, 3), keepdim=True)+1e-9)
    key_loss = -(target*heatmaps.flatten(2).log_softmax(-1).reshape_as(heatmaps)).sum((2, 3)).mean(1)
    key_loss = (key_loss*visible).sum()/visible.sum().clamp_min(1)
    small_mask = F.interpolate(masks[:, None].float(), size=foreground.shape[-2:], mode='area')
    mask_loss = F.binary_cross_entropy_with_logits(foreground, small_mask, pos_weight=torch.tensor(12., device=heatmaps.device))
    vis_loss = F.binary_cross_entropy_with_logits(visibility, visible)
    coordinates, _, _ = decode_heatmaps(heatmaps)
    coordinate_loss = F.smooth_l1_loss(coordinates, keypoints, reduction='none').mean((1, 2))
    coordinate_loss = (coordinate_loss*visible).sum()/visible.sum().clamp_min(1)
    loss = key_loss+.5*mask_loss+.2*vis_loss+.02*coordinate_loss
    return loss, {'keypoint_loss': float(key_loss.detach()), 'mask_loss': float(mask_loss.detach()),
                  'visibility_loss': float(vis_loss.detach()), 'coordinate_loss_px': float(coordinate_loss.detach())}
