"""
V-JEPA Perceptual Loss
======================
Temporal-aware perceptual loss using a frozen V-JEPA encoder (ViT).
Unlike VGG perceptual loss which operates per-frame, V-JEPA processes
spatio-temporal tubes and produces features that capture motion semantics.

Usage in training config::

    "train": {
        "lambda_vjepa": 0.1,
        "vjepa_model": "vit_large",
        "vjepa_checkpoint": "/path/to/vjepa_encoder.pth",
        "vjepa_feature_layers": [6, 12, 18, 23],
        "vjepa_weights": [0.25, 0.25, 0.25, 0.25],
        "vjepa_lossfn_type": "l1",
        "vjepa_patch_size": [2, 16, 16],
        "vjepa_crop_size": 224,
        ...
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPAFeatureExtractor(nn.Module):
    """Frozen V-JEPA ViT encoder for multi-layer feature extraction.

    V-JEPA models use a ViT backbone that takes video clips as input
    (B, C, T, H, W) and produces patch-level latent features.

    We hook into intermediate transformer blocks to extract features at
    multiple depths, analogous to multi-layer VGG feature extraction.
    """

    def __init__(
        self,
        model_name: str = "vit_large",
        checkpoint_path: str = None,
        feature_layers: list = None,
        patch_size: tuple = (2, 16, 16),
        crop_size: int = 224,
        num_frames: int = 16,
    ):
        super().__init__()
        self.feature_layers = feature_layers or [6, 12, 18, 23]
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.num_frames = num_frames

        self.encoder = self._build_encoder(model_name, checkpoint_path)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Register hooks for intermediate feature extraction
        self._features = {}
        self._register_hooks()

        # ImageNet normalisation (V-JEPA uses same convention)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    # ------------------------------------------------------------------
    def _build_encoder(self, model_name: str, checkpoint_path: str):
        """Build ViT encoder matching V-JEPA architecture."""
        # Map model name to architecture params
        configs = {
            "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
            "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
            "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
            "vit_huge": dict(embed_dim=1280, depth=32, num_heads=16),
        }
        if model_name not in configs:
            raise ValueError(f"Unknown V-JEPA model: {model_name}. Choose from {list(configs.keys())}")

        cfg = configs[model_name]

        encoder = VisionTransformer3D(
            img_size=self.crop_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
        )

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # V-JEPA checkpoints may nest under 'encoder', 'model', or 'target_encoder'
            for key in ("encoder", "target_encoder", "model"):
                if key in state:
                    state = state[key]
                    break
            # Strip 'module.' prefix from DDP checkpoints
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing:
                print(f"[VJEPAFeatureExtractor] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(
                    f"[VJEPAFeatureExtractor] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )
            print(f"[VJEPAFeatureExtractor] Loaded checkpoint from {checkpoint_path}")
        else:
            print("[VJEPAFeatureExtractor] WARNING: No checkpoint provided, using random init")

        return encoder

    def _register_hooks(self):
        """Register forward hooks on the transformer blocks we want features from."""
        for idx in self.feature_layers:
            block = self.encoder.blocks[idx]
            block.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    # ------------------------------------------------------------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare video tensor for V-JEPA.

        Args:
            x: (B, T, C, H, W) in [0, 1]  — output convention of VSR models.

        Returns:
            (B, C, T, H, W) normalised and resized to (crop_size, crop_size).
        """
        B, T, C, H, W = x.shape
        # Rearrange to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Spatial resize to crop_size (bilinear per-frame via 4D resize)
        if H != self.crop_size or W != self.crop_size:
            x = x.reshape(B * C, T, H, W)  # fake batch for spatial resize is wrong
            # Correct: resize H, W dimensions
            x = x.view(B, C, T, H, W)
            x = F.interpolate(
                x.flatten(0, 1).unsqueeze(0).reshape(B * T, C, H, W),
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )  # (B*T, C, cs, cs)
            x = x.view(B, T, C, self.crop_size, self.crop_size).permute(0, 2, 1, 3, 4)

        # Temporal: sample/pad to num_frames
        T_cur = x.shape[2]
        if T_cur != self.num_frames:
            x = F.interpolate(
                x.flatten(0, 1).unsqueeze(0).reshape(B * C, 1, T_cur, self.crop_size, self.crop_size).squeeze(1),
                size=(self.num_frames, self.crop_size, self.crop_size),
                mode="trilinear",
                align_corners=False,
            )
            x = x.view(B, C, self.num_frames, self.crop_size, self.crop_size)

        # Normalise
        x = (x - self.mean) / self.std
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> list:
        """Extract multi-layer features.

        Args:
            x: (B, T, C, H, W) video in [0, 1].

        Returns:
            List of feature tensors, one per requested layer.
            Each is (B, N_patches, embed_dim).
        """
        self._features.clear()
        x = self._preprocess(x)
        _ = self.encoder(x)
        return [self._features[idx] for idx in self.feature_layers]


class VJEPAPerceptualLoss(nn.Module):
    """Perceptual loss using frozen V-JEPA encoder features.

    Computes weighted sum of L1/L2 distances between multi-layer V-JEPA
    features of predicted and target video sequences.
    """

    def __init__(
        self,
        model_name: str = "vit_large",
        checkpoint_path: str = None,
        feature_layers: list = None,
        weights: list = None,
        lossfn_type: str = "l1",
        patch_size: tuple = (2, 16, 16),
        crop_size: int = 224,
        num_frames: int = 16,
    ):
        super().__init__()

        self.vjepa = VJEPAFeatureExtractor(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            feature_layers=feature_layers,
            patch_size=patch_size,
            crop_size=crop_size,
            num_frames=num_frames,
        )

        n_layers = len(self.vjepa.feature_layers)
        if weights is None:
            weights = [1.0 / n_layers] * n_layers
        assert len(weights) == n_layers
        self.weights = weights

        if lossfn_type == "l1":
            self.lossfn = nn.L1Loss()
        elif lossfn_type == "l2":
            self.lossfn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported lossfn_type: {lossfn_type}")

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> list:
        return self.vjepa(x)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute V-JEPA perceptual loss.

        Args:
            pred: (B, T, C, H, W) predicted video in [0, 1].
            gt:   (B, T, C, H, W) ground-truth video in [0, 1].

        Returns:
            Scalar loss.
        """
        # GT features are always detached (no gradient through target)
        gt_feats = self._extract(gt)
        # Pred features need gradients to flow back
        pred_feats = self.vjepa(pred)

        loss = pred.new_tensor(0.0)
        for w, pf, gf in zip(self.weights, pred_feats, gt_feats):
            loss = loss + w * self.lossfn(pf, gf.detach())
        return loss


# =====================================================================
# Minimal 3D Vision Transformer matching V-JEPA's encoder architecture
# =====================================================================
class VisionTransformer3D(nn.Module):
    """Video ViT (tubelet embedding) compatible with V-JEPA checkpoints."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: tuple = (2, 16, 16),
        num_frames: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 3D patch embedding (tubelet)
        self.patch_embed = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Number of patches
        n_t = num_frames // patch_size[0]
        n_h = img_size // patch_size[1]
        n_w = img_size // patch_size[2]
        num_patches = n_t * n_h * n_w

        # Positional embedding (no CLS token, matching V-JEPA)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)

        Returns:
            (B, N, embed_dim) patch tokens.
        """
        # Patch embed: (B, embed_dim, nt, nh, nw) -> (B, N, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding (interpolate if sizes differ)
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
