import math
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd


"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2, 7, 16, 25, 34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        """
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        """
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer) - 1):
                self.features.add_module(
                    "child" + str(i),
                    nn.Sequential(
                        *list(model.features.children())[(feature_layer[i] + 1) : (feature_layer[i + 1] + 1)]
                    ),
                )
        else:
            self.features = nn.Sequential(*list(model.features.children())[: (feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss"""

    def __init__(
        self,
        feature_layer=[2, 7, 16, 25, 34],
        weights=[0.1, 0.1, 1.0, 1.0, 1.0],
        lossfn_type="l1",
        use_input_norm=True,
        use_range_norm=False,
    ):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(
            feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm
        )
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == "l1":
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f"feature_layer: {feature_layer}  with weights: {weights}")

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss


# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == "softplusgan":

            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError("GAN type [{:s}] is not found".format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ["wgan", "softplusgan"]:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


class FourierHighFrequencyLoss(nn.Module):
    """Masked Fourier-domain loss that ignores low frequencies around DC."""

    def __init__(self, loss_type="l1", mask_radius=0, mask_ratio=0.1):
        super(FourierHighFrequencyLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.mask_radius = int(mask_radius)
        self.mask_ratio = float(mask_ratio)

        if self.loss_type not in ["l1", "l2"]:
            raise NotImplementedError("Fourier loss type [{:s}] is not found".format(loss_type))

    def _build_mask(self, height, width, device, dtype):
        mask = torch.ones((height, width), device=device, dtype=dtype)

        if self.mask_radius > 0:
            radius = float(min(self.mask_radius, min(height, width) // 2))
        elif self.mask_ratio > 0:
            radius = float(max(1, int(round(min(height, width) * self.mask_ratio))))
            radius = min(radius, min(height, width) // 2)
        else:
            return mask

        yy = torch.arange(height, device=device, dtype=dtype) - (height // 2)
        xx = torch.arange(width, device=device, dtype=dtype) - (width // 2)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radial_distance = torch.sqrt(grid_y.pow(2) + grid_x.pow(2))
        mask = (radial_distance > radius).to(dtype=dtype)
        return mask

    def forward(self, x, y):
        if x.shape != y.shape:
            raise ValueError("FourierHighFrequencyLoss expects tensors with the same shape.")
        if x.ndim < 4:
            raise ValueError("FourierHighFrequencyLoss expects tensors with at least 4 dimensions.")

        x_fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        y_fft = torch.fft.fftshift(torch.fft.fft2(y, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        diff = x_fft - y_fft
        if self.loss_type == "l1":
            loss_map = torch.abs(diff)
        else:
            loss_map = torch.abs(diff).pow(2)

        mask = self._build_mask(loss_map.shape[-2], loss_map.shape[-1], loss_map.device, loss_map.real.dtype)
        mask = mask.view(*([1] * (loss_map.ndim - 2)), loss_map.shape[-2], loss_map.shape[-1])

        masked_loss = loss_map * mask
        num_spectra = loss_map[..., 0, 0].numel()
        denom = mask.sum().clamp_min(1.0) * num_spectra
        return masked_loss.sum() / denom


class CharbonnierFourierLoss(nn.Module):
    """Charbonnier reconstruction loss with an extra masked Fourier loss."""

    def __init__(self, eps=1e-9, fft_weight=0.0, fft_loss_type="l1", fft_mask_radius=0, fft_mask_ratio=0.1):
        super(CharbonnierFourierLoss, self).__init__()
        self.pixel_loss = CharbonnierLoss(eps)
        self.fft_weight = float(fft_weight)
        self.fft_loss = FourierHighFrequencyLoss(
            loss_type=fft_loss_type,
            mask_radius=fft_mask_radius,
            mask_ratio=fft_mask_ratio,
        )

    def forward(self, x, y):
        loss = self.pixel_loss(x, y)
        if self.fft_weight > 0:
            loss = loss + self.fft_weight * self.fft_loss(x, y)
        return loss


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
    penalize the gradient on real data alone: when the
    generator distribution produces the true data distribution
    and the discriminator is equal to 0 on the data manifold, the
    gradient penalty ensures that the discriminator cannot create
    a non-zero gradient orthogonal to the data manifold without
    suffering a loss in the GAN game.
    Ref:
    Eq. 9 in Which training methods for GANs do actually converge.
    """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
