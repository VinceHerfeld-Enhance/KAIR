import torch
from torch.optim import Adam
from models.model_plain import ModelPlain
from collections import OrderedDict


# filepath: /home/vherfeld/Research/KAIR/models/ddp_model_elvsr.py


class DDPModelELVSR(ModelPlain):
    """Train video restoration with pixel loss, compatible with HuggingFace Accelerate DDP."""

    def __init__(self, opt):
        super(DDPModelELVSR, self).__init__(opt)
        self.fix_iter = self.opt_train.get("fix_iter", 0)
        self.fix_keys = self.opt_train.get("fix_keys", [])
        self.fix_unflagged = True
        # Will be set by the training script after accelerator.prepare()
        self.accelerator = None

    # ----------------------------------------
    # Utility: get bare model (unwrap DDP / Accelerate wrappers)
    # ----------------------------------------
    def get_bare_model(self, network):
        # Unwrap accelerate wrapper first if available
        if self.accelerator is not None:
            network = self.accelerator.unwrap_model(network)
        # Then fall back to the parent's unwrapping (handles DataParallel, DDP, etc.)
        if hasattr(network, "module"):
            network = network.module
        return network

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get("fix_keys", [])
        net = self.get_bare_model(self.netG)

        if self.opt_train.get("fix_iter", 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train["fix_lr_mul"]
            print(f"Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.")
            if fix_lr_mul == 1:
                G_optim_params = net.parameters()
            else:
                normal_params = []
                flow_params = []
                for name, param in net.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {"params": normal_params, "lr": self.opt_train["G_optimizer_lr"]},
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
            super(DDPModelELVSR, self).define_optimizer()

    # ----------------------------------------
    # feed data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # Data may already be on the correct device via accelerate's dataloader,
        # but during validation we may need to move it manually.
        self.L = data["L"]
        if not self.L.is_cuda and self.accelerator is not None:
            self.L = self.L.to(self.accelerator.device)
        elif not self.L.is_cuda:
            self.L = self.L.to(self.device)

        if need_H and "H" in data:
            self.H = data["H"]
            if not self.H.is_cuda and self.accelerator is not None:
                self.H = self.H.to(self.accelerator.device)
            elif not self.H.is_cuda:
                self.H = self.H.to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # Handle fix_keys logic (freeze/unfreeze certain parameters)
        if self.fix_iter:
            net = self.get_bare_model(self.netG)
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f"Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.")
                self.fix_unflagged = False
                for name, param in net.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f"Train all the parameters from {self.fix_iter} iters.")
                net.requires_grad_(True)

        # Forward
        self.G_optimizer.zero_grad()
        self.netG.train()
        self.E = self.netG(self.L)
        loss = self.lossfn(self.E, self.H)

        # Backward via accelerator (handles gradient scaling, sync, etc.)
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping (if configured)
        G_optimizer_clipgrad = self.opt_train.get("G_optimizer_clipgrad", 0)
        if G_optimizer_clipgrad > 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(
                    self.get_bare_model(self.netG).parameters(),
                    G_optimizer_clipgrad,
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.get_bare_model(self.netG).parameters(),
                    G_optimizer_clipgrad,
                )

        self.G_optimizer.step()

        # Store loss for logging
        self.log_dict["G_loss"] = loss.item()

    # ----------------------------------------
    # log psnr (training PSNR on current batch)
    # ----------------------------------------
    def log_psnr(self):
        with torch.no_grad():
            self.log_dict["Train_PSNR"] = (
                10 * torch.log10(1.0 / (self.lossfn(self.E.detach(), self.H.detach()) + 1e-8)).item()
            )

    # ----------------------------------------
    # current learning rate
    # ----------------------------------------
    def current_learning_rate(self):
        return self.G_optimizer.param_groups[0]["lr"]

    # ----------------------------------------
    # current visuals (move to cpu for evaluation)
    # ----------------------------------------
    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach().float().cpu()
        out_dict["E"] = self.E.detach().float().cpu()
        if hasattr(self, "H") and self.H is not None:
            out_dict["H"] = self.H.detach().float().cpu()
        return out_dict

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        n = self.L.size(1)
        net = self.get_bare_model(self.netG)
        net.eval()

        pad_seq = self.opt_train.get("pad_seq", False)
        flip_seq = self.opt_train.get("flip_seq", False)
        self.center_frame_only = self.opt_train.get("center_frame_only", False)

        if pad_seq:
            n = n + 1
            self.L = torch.cat([self.L, self.L[:, -1:, :, :, :]], dim=1)

        if flip_seq:
            self.L = torch.cat([self.L, self.L.flip(1)], dim=1)

        with torch.no_grad():
            self.E = self._test_video(self.L)

        if flip_seq:
            output_1 = self.E[:, :n, :, :, :]
            output_2 = self.E[:, n:, :, :, :].flip(1)
            self.E = 0.5 * (output_1 + output_2)

        if pad_seq:
            n = n - 1
            self.E = self.E[:, :n, :, :, :]

        if self.center_frame_only:
            self.E = self.E[:, n // 2, :, :, :]

        net.train()

    def _forward_net(self, lq):
        """Forward through the bare network (unwrapped). Used for inference."""
        net = self.get_bare_model(self.netG)
        if hasattr(self, "netE"):
            net_e = self.get_bare_model(self.netE)
            return net_e(lq).detach().cpu()
        return net(lq).detach().cpu()

    def _test_video(self, lq):
        """Test the video as a whole or as clips (divided temporally)."""
        num_frame_testing = self.opt["val"].get("num_frame_testing", 0)

        if num_frame_testing:
            sf = self.opt["scale"]
            num_frame_overlapping = self.opt["val"].get("num_frame_overlapping", 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt["netG"].get("nonblind_denoising", False) else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]
            E = torch.zeros(b, d, c, h * sf, w * sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
                out_clip = self._test_clip(lq_clip)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping // 2 :, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping // 2 :, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                        out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

                E[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip)
                W[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            window_size = self.opt["val"].get("test_window_size", [6, 8, 8])
            d_old = lq.size(1)
            d_pad = (d_old // window_size[0] + 1) * window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq)
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq):
        """Test the clip as a whole or as patches."""
        sf = self.opt["scale"]
        window_size = self.opt["val"].get("test_window_size", [6, 8, 8])
        size_patch_testing = self.opt["val"].get("size_patch_testing", 0)
        assert size_patch_testing % window_size[-1] == 0, "testing patch size should be a multiple of window_size."

        if size_patch_testing:
            overlap_size = 20
            not_overlap_border = True

            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt["netG"].get("nonblind_denoising", False) else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, d, c, h * sf, w * sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[
                        ...,
                        h_idx : h_idx + size_patch_testing,
                        w_idx : w_idx + size_patch_testing,
                    ]
                    out_patch = self._forward_net(in_patch)

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

            output = self._forward_net(lq)
            output = output[:, :, :, : h_old * sf, : w_old * sf]

        return output

    # ----------------------------------------
    # save network / optimizer / scheduler
    # ----------------------------------------
    def save(self, iter_label):
        """Save networks, optimizers, schedulers.
        The caller (training script) should unwrap netG before calling this
        via _unwrap_and_save, or we unwrap here as a safety measure.
        """
        self.save_network(self.save_dir, self.get_bare_model(self.netG), "G", iter_label)
        self.save_optimizer(self.save_dir, self.G_optimizer, "optimizerG", iter_label)
        if hasattr(self, "E_optimizer") and self.E_optimizer is not None:
            self.save_optimizer(self.save_dir, self.E_optimizer, "optimizerE", iter_label)
        if hasattr(self, "netE"):
            self.save_network(self.save_dir, self.get_bare_model(self.netE), "E", iter_label)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key="params"):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path, map_location="cpu")
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

        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(
                        f"Size different, ignore [{k}]: crt_net: " f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)


# Need this import for current_visuals
