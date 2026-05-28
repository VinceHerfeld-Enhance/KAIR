import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.model_plain import ModelPlain


class ModelELVSR(ModelPlain):
    """Train video restoration  with pixel loss"""

    def __init__(self, opt):
        super(ModelELVSR, self).__init__(opt)
        self.fix_iter = self.opt_train.get("fix_iter", 0)
        self.fix_keys = self.opt_train.get("fix_keys", [])
        self.fix_unflagged = True
        self.tsf = int(opt["netG"].get("tsf", 1)) if opt.get("netG") else 1
        self.gt_indices = None  # (B, K) LongTensor of sparse GT positions, or None for full GT

        # ---- Frozen RAFT for intermediate flow supervision ----
        self.lambda_flow = float(self.opt_train.get("lambda_flow_supervision", 0.0))
        self.raft_model = None
        if self.tsf > 1 and self.lambda_flow > 0:
            self._init_raft()

        # ---- Frozen V-JEPA for video perceptual loss ----
        self.lambda_vjepa = float(self.opt_train.get("lambda_vjepa", 0.0))
        self.vjepa_loss = None
        if self.lambda_vjepa > 0:
            self._init_vjepa()

    # ----------------------------------------
    # feed L/H data, capturing sparse GT indices when present
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        super(ModelELVSR, self).feed_data(data, need_H)
        if "gt_indices" in data:
            self.gt_indices = data["gt_indices"].to(self.device)  # (B, K)
        else:
            self.gt_indices = None
        # Store full GT sequence for flow supervision (before any sparse indexing)
        if "H_full" in data:
            self.H_full = data["H_full"].to(self.device)
        else:
            self.H_full = self.H  # when no sparse GT, H is the full sequence

        # interp_steps to be set externally before optimize_parameters
        self.interp_steps = None

    # ----------------------------------------
    # feed L to netG, with optional interp_steps
    # ----------------------------------------
    def netG_forward(self):
        if self.interp_steps is not None:
            self.E = self.netG(self.L, interp_steps=self.interp_steps)
        else:
            self.E = self.netG(self.L)

    # ----------------------------------------
    # compute loss against sparse or full GT
    # ----------------------------------------
    def compute_G_loss(self):
        if self.gt_indices is not None:
            # self.E:  (B, T_out, C, H, W)
            # self.gt_indices: (B, K) — per-sample indices into T_out
            # self.H:  (B, K, C, H, W) — only the selected GT frames
            B, K = self.gt_indices.shape
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)
            E_sel = self.E[b_idx, self.gt_indices]  # (B, K, C, H, W)
            G_loss = self.G_lossfn_weight * self.G_lossfn(E_sel, self.H)
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

        # ---- Auxiliary flow supervision for intermediate flow corrector ----
        flow_loss = self._compute_flow_supervision_loss()
        if flow_loss is not None:
            self.log_dict["flow_loss"] = flow_loss.item()
            G_loss = G_loss + flow_loss

        # ---- V-JEPA video perceptual loss ----
        vjepa_loss = self._compute_vjepa_loss()
        if vjepa_loss is not None:
            self.log_dict["vjepa_loss"] = vjepa_loss.item()
            G_loss = G_loss + vjepa_loss

        return G_loss

    # ----------------------------------------
    # Frozen V-JEPA for video perceptual loss
    # ----------------------------------------
    def _init_vjepa(self):
        """Load frozen V-JEPA encoder for video perceptual loss."""
        from models.loss_vjepa import VJEPAPerceptualLoss

        model_name = self.opt_train.get("vjepa_model", "vit_large")
        checkpoint = self.opt_train.get("vjepa_checkpoint", None)
        feature_layers = self.opt_train.get("vjepa_feature_layers", None)
        weights = self.opt_train.get("vjepa_weights", None)
        lossfn_type = self.opt_train.get("vjepa_lossfn_type", "l1")
        patch_size = tuple(self.opt_train.get("vjepa_patch_size", [2, 16, 16]))
        crop_size = int(self.opt_train.get("vjepa_crop_size", 224))
        num_frames = int(self.opt_train.get("vjepa_num_frames", 16))

        self.vjepa_loss = VJEPAPerceptualLoss(
            model_name=model_name,
            checkpoint_path=checkpoint,
            feature_layers=feature_layers,
            weights=weights,
            lossfn_type=lossfn_type,
            patch_size=patch_size,
            crop_size=crop_size,
            num_frames=num_frames,
        ).to(self.device)
        print(f"[ModelELVSR] V-JEPA perceptual loss loaded (model={model_name}, lambda={self.lambda_vjepa})")

    def _compute_vjepa_loss(self):
        """Compute V-JEPA perceptual loss between predicted and GT video."""
        if self.vjepa_loss is None or self.lambda_vjepa <= 0:
            return None

        # Use full GT sequence for perceptual comparison
        pred = self.E  # (B, T, C, H, W)
        gt = self.H_full if hasattr(self, "H_full") else self.H

        # When sparse GT: compare only at GT frame positions
        if self.gt_indices is not None:
            B, K = self.gt_indices.shape
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)
            pred = pred[b_idx, self.gt_indices]  # (B, K, C, H, W)
            gt = self.H  # already (B, K, C, H, W)

        return self.lambda_vjepa * self.vjepa_loss(pred, gt)

    # ----------------------------------------
    # Frozen RAFT for flow pseudo-GT
    # ----------------------------------------
    def _init_raft(self):
        """Load a frozen RAFT-Large for on-the-fly flow pseudo-GT."""
        raft_checkpoint = self.opt_train.get("raft_checkpoint", None)
        if raft_checkpoint:
            from torchvision.models.optical_flow import raft_large

            self.raft_model = raft_large(weights=None, progress=False)
            state_dict = torch.load(raft_checkpoint, map_location="cpu", weights_only=True)
            self.raft_model.load_state_dict(state_dict)
            print(f"[ModelELVSR] Frozen RAFT-Large loaded from {raft_checkpoint}")
        else:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

            self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
            print(f"[ModelELVSR] Frozen RAFT-Large loaded from torchvision defaults")
        self.raft_model.eval()
        self.raft_model.to(self.device)
        for p in self.raft_model.parameters():
            p.requires_grad_(False)
        print(f"[ModelELVSR] Flow supervision lambda={self.lambda_flow}")

    @torch.no_grad()
    def _raft_flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute RAFT optical flow from img1 to img2.

        Args:
            img1, img2: [B, 3, H, W] in [0, 1].

        Returns:
            flow: [B, H, W, 2] (last-dim convention matching the corrector).
        """
        # RAFT expects [B, 3, H, W] in [0, 1] with H, W divisible by 8
        h, w = img1.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")

        flow_list = self.raft_model(img1, img2)
        flow = flow_list[-1]  # [B, 2, H_pad, W_pad]

        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :h, :w]

        return flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

    def _compute_flow_supervision_loss(self):
        """Compute L1 loss between corrected intermediate flows and RAFT pseudo-GT.

        The corrector predicts flows at LR resolution (virtual -> left bracket,
        virtual -> right bracket).  We run RAFT on GT frames at their native HR
        resolution (where RAFT is most accurate), then downscale the resulting
        flow to match the corrector's LR grid.

        For each gap between input frame i and i+1, and each sub-step k:
            raft_v2l = downsample(RAFT(gt_mid_hr, gt_left_hr))
            raft_v2r = downsample(RAFT(gt_mid_hr, gt_right_hr))
            loss += L1(corrected_v2l, raft_v2l) + L1(corrected_v2r, raft_v2r)

        Returns:
            Weighted flow loss scalar, or None if not applicable.
        """
        if self.raft_model is None or self.lambda_flow <= 0:
            return None

        net = self.get_bare_model(self.netG)
        if not hasattr(net, "_aux_interp_flows") or not net._aux_interp_flows:
            return None

        aux_flows = net._aux_interp_flows  # list of per-gap lists of dicts
        tsf = self.tsf
        # H_full is [B, T_full, C, H_hr, W_hr] — the full GT sequence
        gt = self.H_full
        B, T_full = gt.shape[:2]

        # Get LR spatial resolution from corrector flows
        sample_flow = aux_flows[0][0]["v2l"]  # [B, H_lr, W_lr, 2]
        h_lr, w_lr = sample_flow.shape[1], sample_flow.shape[2]

        total_loss = sample_flow.new_tensor(0.0)
        n_terms = 0

        for gap_idx, gap_flows in enumerate(aux_flows):
            # GT indices for left/right bracket frames
            gt_left_idx = gap_idx * tsf
            gt_right_idx = (gap_idx + 1) * tsf

            if gt_right_idx >= T_full:
                continue

            gt_left_hr = gt[:, gt_left_idx]  # [B, 3, H_hr, W_hr]
            gt_right_hr = gt[:, gt_right_idx]  # [B, 3, H_hr, W_hr]

            for k_idx, flow_dict in enumerate(gap_flows):
                # When interp_steps is set, gap_flows only contains entries
                # for the sampled steps, so map k_idx to the actual sub-step.
                if self.interp_steps is not None:
                    k = self.interp_steps[k_idx]
                else:
                    k = k_idx + 1  # sub-step index (1 .. tsf-1)

                # GT intermediate frame at native HR resolution
                gt_mid_idx = gt_left_idx + k
                if gt_mid_idx >= T_full:
                    continue
                gt_mid_hr = gt[:, gt_mid_idx]

                # RAFT at HR, then downscale flow to LR
                raft_v2l_hr = self._raft_flow(gt_mid_hr, gt_left_hr)  # [B, H_hr, W_hr, 2]
                raft_v2r_hr = self._raft_flow(gt_mid_hr, gt_right_hr)

                # Downscale flow to LR grid: resize spatially and scale magnitudes
                scale_h = h_lr / raft_v2l_hr.shape[1]
                scale_w = w_lr / raft_v2l_hr.shape[2]
                raft_v2l = F.interpolate(
                    raft_v2l_hr.permute(0, 3, 1, 2),
                    size=(h_lr, w_lr),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                raft_v2l[..., 0] *= scale_w
                raft_v2l[..., 1] *= scale_h
                raft_v2r = F.interpolate(
                    raft_v2r_hr.permute(0, 3, 1, 2),
                    size=(h_lr, w_lr),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                raft_v2r[..., 0] *= scale_w
                raft_v2r[..., 1] *= scale_h

                pred_v2l = flow_dict["v2l"]
                pred_v2r = flow_dict["v2r"]

                total_loss = total_loss + F.l1_loss(pred_v2l, raft_v2l) + F.l1_loss(pred_v2r, raft_v2r)
                n_terms += 2

        if n_terms == 0:
            return None

        return self.lambda_flow * total_loss / n_terms

    # ----------------------------------------
    # PSNR logging, aligned with sparse GT when present
    # ----------------------------------------
    def log_psnr(self):
        if self.gt_indices is not None:
            B, K = self.gt_indices.shape
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)
            E_sel = self.E[b_idx, self.gt_indices].detach().float()
            H_flat = self.H.detach().float().flatten(0, 1)
            E_flat = E_sel.flatten(0, 1)
        else:
            E_flat = self.E.detach().float().flatten(0, 1)
            H_flat = self.H.detach().float().flatten(0, 1)
        mse = nn.MSELoss()(E_flat, H_flat)
        psnr = 10 * torch.log10(1 / mse)
        self.log_dict["PSNR"] = psnr.item()

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        lr_multipliers = self.opt_train.get("G_lr_multipliers", None)
        if isinstance(lr_multipliers, dict) and len(lr_multipliers) > 0:
            base_lr = self.opt_train["G_optimizer_lr"]
            grouped = {}
            for name, param in self.netG.named_parameters():
                if not param.requires_grad:
                    print("Params [{:s}] will not optimize.".format(name))
                    continue

                mult = 1.0
                for key, value in lr_multipliers.items():
                    if key in name:
                        mult = float(value)
                        break
                grouped.setdefault(mult, []).append(param)

            G_optim_params = []
            for mult in sorted(grouped.keys()):
                params = grouped[mult]
                group_lr = base_lr * mult
                print(f"Optimizer group lr={group_lr:.6g} (x{mult}) with {len(params)} tensors")
                G_optim_params.append({"params": params, "lr": group_lr})

            if self.opt_train["G_optimizer_type"] == "adam":
                self.G_optimizer = Adam(
                    G_optim_params,
                    lr=base_lr,
                    betas=self.opt_train["G_optimizer_betas"],
                    weight_decay=self.opt_train["G_optimizer_wd"],
                )
            else:
                raise NotImplementedError
            return

        self.fix_keys = self.opt_train.get("fix_keys", [])
        if self.opt_train.get("fix_iter", 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train["fix_lr_mul"]
            print(f"Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.")
            if fix_lr_mul == 1:
                G_optim_params = self.netG.parameters()
            else:  # separate flow params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {"params": normal_params, "lr": self.opt_train["G_optimizer_lr"]},  # add normal params first
                    {"params": flow_params, "lr": self.opt_train["G_optimizer_lr"] * fix_lr_mul},
                ]

            if self.opt_train["G_optimizer_type"] == "adam":
                self.G_optimizer = Adam(
                    G_optim_params,
                    lr=self.opt_train["G_optimizer_lr"],
                    betas=self.opt_train["G_optimizer_betas"],
                    weight_decay=self.opt_train["G_optimizer_wd"],
                )
            else:
                raise NotImplementedError
        else:
            super(ModelELVSR, self).define_optimizer()

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.fix_iter:
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f"Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.")
                self.fix_unflagged = False
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f"Train all the parameters from {self.fix_iter} iters.")
                self.netG.requires_grad_(True)

        super(ModelELVSR, self).optimize_parameters(current_step)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def _n_out(self, n_in):
        """Number of output frames for n_in input frames."""
        if self.tsf > 1:
            return (n_in - 1) * self.tsf + 1
        return n_in

    def test(self):
        n = self.L.size(1)  # input frames
        tsf = self.tsf
        self.netG.eval()

        pad_seq = self.opt_train.get("pad_seq", False)
        # flip_seq is incompatible with tsf > 1 (junction creates artifacts)
        flip_seq = self.opt_train.get("flip_seq", False) and tsf == 1
        self.center_frame_only = self.opt_train.get("center_frame_only", False)

        if pad_seq:
            self.L = torch.cat([self.L, self.L[:, -1:, :, :, :]], dim=1)

        if flip_seq:
            self.L = torch.cat([self.L, self.L.flip(1)], dim=1)

        with torch.no_grad():
            self.E = self._test_video(self.L)

        if flip_seq:
            n_out = n  # tsf == 1 here
            output_1 = self.E[:, :n_out, :, :, :]
            output_2 = self.E[:, n_out:, :, :, :].flip(1)
            self.E = 0.5 * (output_1 + output_2)

        if pad_seq:
            # Keep only frames corresponding to the original n input frames
            n_out = self._n_out(n)
            self.E = self.E[:, :n_out, :, :, :]

        if self.center_frame_only:
            n_out = self.E.size(1)
            self.E = self.E[:, n_out // 2, :, :, :]

        self.netG.train()

    def _test_video(self, lq):
        """test the video as a whole or as clips (divided temporally)."""

        num_frame_testing = self.opt["val"].get("num_frame_testing", 0)
        tsf = self.tsf

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = self.opt["scale"]
            num_frame_overlapping = self.opt["val"].get("num_frame_overlapping", 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt["netG"].get("nonblind_denoising", False) else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]

            d_out = self._n_out(d)
            E = torch.zeros(b, d_out, c, h * sf, w * sf)
            W = torch.zeros(b, d_out, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
                out_clip = self._test_clip(lq_clip)
                out_clip_len = out_clip.size(1)
                out_clip_mask = torch.ones((b, out_clip_len, 1, 1, 1))

                # Map input clip start to output position
                out_start = d_idx * tsf if tsf > 1 else d_idx

                if not_overlap_border:
                    out_overlap = num_frame_overlapping * tsf // 2 if tsf > 1 else num_frame_overlapping // 2
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -out_overlap:, ...] *= 0
                        out_clip_mask[:, -out_overlap:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :out_overlap, ...] *= 0
                        out_clip_mask[:, :out_overlap, ...] *= 0

                E[:, out_start : out_start + out_clip_len, ...].add_(out_clip)
                W[:, out_start : out_start + out_clip_len, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = self.opt["val"].get("test_window_size", [6, 8, 8])
            d_old = lq.size(1)
            d_pad = (d_old // window_size[0] + 1) * window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq)
            d_out_old = self._n_out(d_old)
            output = output[:, :d_out_old, :, :, :]

        return output

    def _test_clip(self, lq):
        """test the clip as a whole or as patches."""

        sf = self.opt["scale"]
        window_size = self.opt["val"].get("test_window_size", [6, 8, 8])
        size_patch_testing = self.opt["val"].get("size_patch_testing", 0)
        assert size_patch_testing % window_size[-1] == 0, "testing patch size should be a multiple of window_size."

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = 20
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt["netG"].get("nonblind_denoising", False) else c
            d_out = self._n_out(d)
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, d_out, c, h * sf, w * sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx : h_idx + size_patch_testing, w_idx : w_idx + size_patch_testing]
                    if hasattr(self, "netE"):
                        out_patch = self.netE(in_patch).detach().cpu()
                    else:
                        out_patch = self.netG(in_patch).detach().cpu()

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2 :, :] *= 0
                            out_patch_mask[..., -overlap_size // 2 :, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2 :] *= 0
                            out_patch_mask[..., :, -overlap_size // 2 :] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., : overlap_size // 2, :] *= 0
                            out_patch_mask[..., : overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, : overlap_size // 2] *= 0
                            out_patch_mask[..., :, : overlap_size // 2] *= 0

                    E[
                        ...,
                        h_idx * sf : (h_idx + size_patch_testing) * sf,
                        w_idx * sf : (w_idx + size_patch_testing) * sf,
                    ].add_(out_patch)
                    W[
                        ...,
                        h_idx * sf : (h_idx + size_patch_testing) * sf,
                        w_idx * sf : (w_idx + size_patch_testing) * sf,
                    ].add_(out_patch_mask)
            output = E.div_(W)

        else:
            _, _, _, h_old, w_old = lq.size()
            h_pad = (h_old // window_size[1] + 1) * window_size[1] - h_old
            w_pad = (w_old // window_size[2] + 1) * window_size[2] - w_old

            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)

            if hasattr(self, "netE"):
                output = self.netE(lq).detach().cpu()
            else:
                output = self.netG(lq).detach().cpu()

            output = output[:, :, :, : h_old * sf, : w_old * sf]

        return output

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key="params"):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        self._print_different_keys_loading(network, state_dict, strict)
        network.load_state_dict(state_dict, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print("Current net - loaded net:")
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f"  {v}")
            print("Loaded net - current net:")
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f"  {v}")

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(
                        f"Size different, ignore [{k}]: crt_net: " f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)
