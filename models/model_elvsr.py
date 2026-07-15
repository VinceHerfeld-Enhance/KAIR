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
        # Linear decay: lambda_flow decays from its initial value toward
        # ``flow_loss_min_weight`` over ``flow_loss_decay_steps`` training
        # steps.  Set ``flow_loss_decay_steps`` to 0 (default) to keep the
        # weight constant.  This lets the corrector bootstrap from RAFT early
        # on but specialise for VSR later, while ``flow_loss_min_weight`` keeps
        # a residual anchor so flows stay sane in ill-posed (textureless /
        # occluded) regions instead of drifting once the anchor decays.
        self.flow_loss_decay_steps = int(self.opt_train.get("flow_loss_decay_steps", 0))
        self.flow_loss_min_weight = float(self.opt_train.get("flow_loss_min_weight", 0.0))
        # Teacher-forcing of the interp_flow_net with RAFT pseudo-GT during
        # early training. ``teacher_flow_p0`` is the per-step probability of
        # replacing the corrector output with the RAFT flow at step 0; the
        # probability decays linearly to 0 over ``teacher_flow_decay_steps``.
        self.teacher_flow_p0 = float(self.opt_train.get("teacher_flow_p0", 0.0))
        self.teacher_flow_decay_steps = int(self.opt_train.get("teacher_flow_decay_steps", 20000))
        self.raft_model = None
        if self.tsf > 1 and (self.lambda_flow > 0 or self.teacher_flow_p0 > 0):
            self._init_raft()
        # Per-step caches populated in optimize_parameters
        self._current_pseudo_gt = None
        self._use_teacher_this_step = False

        # ---- Frozen V-JEPA for video perceptual loss ----
        self.lambda_vjepa = float(self.opt_train.get("lambda_vjepa", 0.0))
        self.vjepa_loss = None
        if self.lambda_vjepa > 0:
            self._init_vjepa()

        # ---- Edge-aware spatial smoothness on the predicted intermediate flow ----
        # Targets "flying pixels": incoherent per-pixel flow that forward-splats
        # content to scattered wrong locations. An image-edge-weighted first/second
        # order smoothness prior suppresses flow gradients WITHIN objects (flat image
        # regions) while RELAXING at object boundaries (image edges) — so it removes
        # scatter without forcing exact object mapping (a soft, edge-respecting prior,
        # not a hard segmentation constraint). Applied per hypothesis, so the K
        # multi-hypothesis trajectories stay diverse (each is internally coherent;
        # they may differ from one another). Set ``lambda_flow_smooth`` low and, if it
        # over-smooths, decay it like the flow supervision. ``flow_smooth_order`` 1 or
        # 2 (2 permits smooth gradients like rotation/zoom, penalizing only kinks);
        # ``flow_smooth_edge`` is the edge sensitivity alpha in exp(-alpha*|grad I|).
        self.lambda_flow_smooth = float(self.opt_train.get("lambda_flow_smooth", 0.0))
        self.flow_smooth_order = int(self.opt_train.get("flow_smooth_order", 1))
        self.flow_smooth_edge = float(self.opt_train.get("flow_smooth_edge", 10.0))

    # ----------------------------------------
    # feed L/H data, capturing sparse GT indices when present
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        super(ModelELVSR, self).feed_data(data, need_H)
        if "gt_indices" in data:
            self.gt_indices = data["gt_indices"].to(self.device, non_blocking=True)  # (B, K)
        else:
            self.gt_indices = None
        # Store full GT sequence for flow supervision (before any sparse indexing).
        # When the training loop pre-subsets GT, H already contains only the
        # needed frames and H_full_indices maps positions back to the original
        # full-sequence indexing. H_full == H in that case (same tensor).
        if "H_full" in data:
            self.H_full = data["H_full"].to(self.device, non_blocking=True)
        else:
            self.H_full = self.H  # when no sparse GT, H is the full sequence
        # Mapping from H_full positions → original absolute GT frame indices.
        # Used by _compute_pseudo_gt_flows to locate bracket/mid frames.
        self._h_full_indices = data.get("H_full_indices", None)

        # interp_steps to be set externally before optimize_parameters
        self.interp_steps = None

    # ----------------------------------------
    # feed L to netG, with optional interp_steps and oracle teacher flows
    # ----------------------------------------
    def netG_forward(self):
        teacher = self._current_pseudo_gt if self._use_teacher_this_step else None
        kwargs = {}
        if self.interp_steps is not None:
            kwargs["interp_steps"] = self.interp_steps
        if teacher is not None:
            kwargs["teacher_flows"] = teacher
        self.E = self.netG(self.L, **kwargs)

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
            self.log_dict["flow_loss"] = self._detach_log_value(flow_loss)
            G_loss = G_loss + flow_loss

        # ---- V-JEPA video perceptual loss ----
        vjepa_loss = self._compute_vjepa_loss()
        if vjepa_loss is not None:
            self.log_dict["vjepa_loss"] = self._detach_log_value(vjepa_loss)
            G_loss = G_loss + vjepa_loss

        # ---- Edge-aware spatial flow smoothness (anti flying-pixel) ----
        smooth_loss = self._compute_flow_smoothness_loss()
        if smooth_loss is not None:
            self.log_dict["flow_smooth_loss"] = self._detach_log_value(smooth_loss)
            G_loss = G_loss + smooth_loss

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
        for p in self.raft_model.parameters():
            p.requires_grad_(False)
        self.raft_model.to(self.device)
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

    def _raft_flow_lr(self, img1_hr, img2_hr, h_lr, w_lr):
        """RAFT(img1, img2) at HR, downsampled+scaled to LR grid."""
        flow_hr = self._raft_flow(img1_hr, img2_hr)  # [B, H_hr, W_hr, 2]
        scale_h = h_lr / flow_hr.shape[1]
        scale_w = w_lr / flow_hr.shape[2]
        flow_lr = F.interpolate(
            flow_hr.permute(0, 3, 1, 2), size=(h_lr, w_lr), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        flow_lr[..., 0] *= scale_w
        flow_lr[..., 1] *= scale_h
        return flow_lr

    @torch.no_grad()
    def _raft_bracket_flows(self, gt_mid_hr, gt_left_hr, gt_right_hr, h_lr, w_lr):
        """RAFT pseudo-GT flows between a mid frame and its L/R bracket.

        RAFT is run on the HR GT frames, then the resulting flows are
        downsampled to the LR grid and rescaled to LR-pixel units — the
        resolution the student's flows operate on. RAFT is run at HR (not on
        LR-downsampled frames) because torchvision's RAFT requires inputs of at
        least 128px (it downsamples by 8 and needs >=16px feature maps); typical
        LR patches (e.g. gt_size/sf = 64px) are too small. The four directional
        flows (mid->L, mid->R, L->mid, R->mid) are batched into one forward pass.
        ``gt_mid_hr`` may carry per-sample content (a different frame per batch
        row); the bracket frames are shared. Returns ``{'v2l','v2r','l2v','r2v'}``
        as [B, h_lr, w_lr, 2] flows in LR-pixel units.
        """
        mid, left, right = gt_mid_hr, gt_left_hr, gt_right_hr
        src = torch.cat([mid, mid, left, right], dim=0)
        dst = torch.cat([left, right, mid, mid], dim=0)
        flow_hr_4 = self._raft_flow(src, dst)  # [4B, H_hr, W_hr, 2] at HR res + HR units

        # Downsample HR flows to the LR grid and rescale magnitudes to LR units.
        h_hr, w_hr = flow_hr_4.shape[1], flow_hr_4.shape[2]
        flow_lr_4 = F.interpolate(
            flow_hr_4.permute(0, 3, 1, 2), size=(h_lr, w_lr), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        flow_lr_4[..., 0] *= w_lr / w_hr
        flow_lr_4[..., 1] *= h_lr / h_hr

        B = mid.shape[0]
        v2l, v2r, l2v, r2v = flow_lr_4.split(B, dim=0)
        return {"v2l": v2l, "v2r": v2r, "l2v": l2v, "r2v": r2v}

    @torch.no_grad()
    def _compute_pseudo_gt_flows(self):
        """Pre-compute RAFT pseudo-GT flows for all (gap, sub-step) pairs.

        Per-sample regime (``self.interp_steps`` is a ``[B, n_gaps, S]`` tensor):
        returns ``{gap_idx: [slot0, slot1, ...]}`` where each slot is a flow dict
        and entry ``b`` uses sample ``b``'s own intermediate frame. GT is laid out
        structurally — input frame ``g`` at position ``g*(S+1)`` and gap ``g`` slot
        ``s`` at ``g*(S+1)+1+s`` — so per-sample mid frames are read directly with
        no index map.

        Legacy regime (dict/list/None ``interp_steps``): returns a nested dict
        ``{gap_idx: {k: {'v2l','v2r','l2v','r2v'}}}``.

        Both are at LR resolution, ready for ``netG(..., teacher_flows=...)`` and
        reused by the flow-supervision loss. Returns ``None`` if RAFT is absent.

        The four directional RAFT calls per ``(gap, k)`` are batched into a
        single forward pass (B*4) and the result is split + downsampled in
        one shot.

        When the training loop pre-subsets GT (H_full_indices is set), frame
        positions are looked up via that index map instead of assuming H_full
        contains a contiguous full-length sequence.
        """
        if self.raft_model is None or self.tsf <= 1:
            return None
        gt = self.H_full
        h_lr, w_lr = self.L.shape[-2:]

        # ---- Per-sample regime: interp_steps is a [B, n_gaps, S] tensor ----
        # GT is in structural order (input frame g at g*(S+1); gap g slot s at
        # g*(S+1)+1+s), so each sample's own mid frame is read directly by
        # position — no absolute-index map needed.
        if torch.is_tensor(self.interp_steps):
            S = self.interp_steps.size(2)
            n_gaps = self.interp_steps.size(1)
            stride = S + 1  # one input-aligned frame + S mids per gap
            out = {}
            for gap_idx in range(n_gaps):
                gt_left_hr = gt[:, gap_idx * stride]
                gt_right_hr = gt[:, (gap_idx + 1) * stride]
                slots = [
                    self._raft_bracket_flows(gt[:, gap_idx * stride + 1 + s], gt_left_hr, gt_right_hr, h_lr, w_lr)
                    for s in range(S)
                ]
                out[gap_idx] = slots  # per-slot list (aligned by index, not keyed by k)
            return out

        # ---- Legacy regime: shared scalar sub-steps (dict / list / None) ----
        T_in = self.L.shape[1]
        default_sub_steps = list(range(1, self.tsf))

        # Build absolute-index → H_full-position lookup.
        if self._h_full_indices is not None:
            idx_map = {abs_idx: pos for pos, abs_idx in enumerate(self._h_full_indices)}
        else:
            idx_map = None

        def _get_frame(abs_idx):
            """Fetch a GT frame by its absolute index in the original sequence."""
            if idx_map is not None:
                pos = idx_map.get(abs_idx)
                if pos is None:
                    return None
                return gt[:, pos]
            if abs_idx >= gt.shape[1]:
                return None
            return gt[:, abs_idx]

        out = {}
        for gap_idx in range(T_in - 1):
            # Resolve sub-steps for this gap
            if isinstance(self.interp_steps, dict):
                sub_steps = self.interp_steps.get(gap_idx, default_sub_steps)
            elif self.interp_steps is not None:
                sub_steps = self.interp_steps
            else:
                sub_steps = default_sub_steps
            gt_left_idx = gap_idx * self.tsf
            gt_right_idx = (gap_idx + 1) * self.tsf
            gt_left_hr = _get_frame(gt_left_idx)
            gt_right_hr = _get_frame(gt_right_idx)
            if gt_left_hr is None or gt_right_hr is None:
                continue
            per_k = {}
            for k in sub_steps:
                gt_mid_hr = _get_frame(gt_left_idx + k)
                if gt_mid_hr is None:
                    continue
                per_k[k] = self._raft_bracket_flows(gt_mid_hr, gt_left_hr, gt_right_hr, h_lr, w_lr)
            if per_k:
                out[gap_idx] = per_k
        return out

    def _teacher_prob(self, current_step):
        """Linearly decay teacher-forcing probability from p0 to 0."""
        if self.teacher_flow_p0 <= 0 or self.teacher_flow_decay_steps <= 0:
            return 0.0
        frac = 1.0 - current_step / float(self.teacher_flow_decay_steps)
        return max(0.0, self.teacher_flow_p0 * frac)

    def _flow_loss_weight(self, current_step):
        """Return the (possibly decayed) flow-supervision weight."""
        if self.lambda_flow <= 0:
            return 0.0
        if self.flow_loss_decay_steps <= 0:
            return self.lambda_flow  # constant
        frac = 1.0 - current_step / float(self.flow_loss_decay_steps)
        # Decay toward the floor, not to 0, so a residual RAFT anchor remains.
        return max(self.flow_loss_min_weight, self.lambda_flow * frac)

    @staticmethod
    def _min_k_flow_loss(pred, gt):
        """Hindsight (multiple-choice) L1 between predicted flow(s) and the teacher.

        Args:
            pred: ``[B, K, H, W, 2]`` multi-flow hypotheses, or ``[B, H, W, 2]``
                  single flow.
            gt:   ``[B, H, W, 2]`` RAFT pseudo-GT flow.

        Takes a per-pixel ``min`` over the K hypotheses so only the
        best-matching one is pulled toward the teacher (Guzman-Rivera et al.,
        MCL), leaving the others free to specialise. Reduces *exactly* to plain
        L1 when ``K == 1`` (or a single flow is passed), so single-flow configs
        are unaffected.
        """
        if pred.dim() == 4:
            return F.l1_loss(pred, gt)
        # pred [B, K, H, W, 2], gt [B, H, W, 2]; mean over the 2 flow components
        # keeps the scale identical to F.l1_loss in the K == 1 case.
        d = (pred - gt.unsqueeze(1)).abs().mean(dim=-1)  # [B, K, H, W]
        return d.min(dim=1).values.mean()

    def _compute_flow_supervision_loss(self):
        """min-over-K flow loss between refined interpolation flows and RAFT pseudo-GT.

        The spatiotemporal model may expose either:
            - legacy ``v2l`` / ``v2r`` virtual-to-bracket flows, or
            - refined ``l2v`` / ``r2v`` bracket-to-virtual splatting flows.

        The latter is the current learned target after unifying flow
        correction into the STVSR flow refiner. Each may carry an extra
        ``n_flows_per_frame`` hypothesis axis, reduced by :meth:`_min_k_flow_loss`.
        """
        lam = self._current_flow_loss_weight
        if self.raft_model is None or lam <= 0:
            return None
        net = self.get_bare_model(self.netG)
        if not hasattr(net, "_aux_interp_flows") or not net._aux_interp_flows:
            return None
        pseudo = self._current_pseudo_gt
        if pseudo is None:
            return None

        aux_flows = net._aux_interp_flows  # list of per-gap lists of dicts
        per_sample = torch.is_tensor(self.interp_steps)
        total = self.L.new_tensor(0.0)
        n_terms = 0
        for gap_idx, gap_flows in enumerate(aux_flows):
            if gap_idx not in pseudo:
                continue
            gap_pseudo = pseudo[gap_idx]  # per-sample: list[slot]; legacy: {k: flows}
            for k_idx, flow_dict in enumerate(gap_flows):
                if per_sample:
                    # Network slots and pseudo-GT slots are emitted in the same
                    # order, so they match directly by index.
                    if k_idx >= len(gap_pseudo):
                        continue
                    gt_flows = gap_pseudo[k_idx]
                else:
                    # Legacy: resolve the scalar sub-step k for this slot.
                    if isinstance(self.interp_steps, dict):
                        gap_steps = self.interp_steps.get(gap_idx, list(range(1, self.tsf)))
                        k = gap_steps[k_idx] if k_idx < len(gap_steps) else k_idx + 1
                    elif self.interp_steps is not None:
                        k = self.interp_steps[k_idx]
                    else:
                        k = k_idx + 1
                    if k not in gap_pseudo:
                        continue
                    gt_flows = gap_pseudo[k]
                if "l2v" in flow_dict and "r2v" in flow_dict:
                    total = total + self._min_k_flow_loss(flow_dict["l2v"], gt_flows["l2v"])
                    total = total + self._min_k_flow_loss(flow_dict["r2v"], gt_flows["r2v"])
                    n_terms += 2
                elif "v2l" in flow_dict and "v2r" in flow_dict:
                    total = total + self._min_k_flow_loss(flow_dict["v2l"], gt_flows["v2l"])
                    total = total + self._min_k_flow_loss(flow_dict["v2r"], gt_flows["v2r"])
                    n_terms += 2
        if n_terms == 0:
            return None
        return lam * total / n_terms

    @staticmethod
    def _edge_aware_flow_smoothness(flow, img, order=1, edge=10.0):
        """Edge-aware first/second-order spatial smoothness of a flow field.

        Args:
            flow: ``[B, K, H, W, 2]`` multi-hypothesis flow, or ``[B, H, W, 2]``.
            img:  ``[B, C, H, W]`` source-grid image; its gradients relax the
                  penalty at object boundaries (edge-aware weighting).
            order: 1 (penalize flow gradient) or 2 (penalize curvature — permits
                   smoothly-varying flow like rotation/zoom, penalizes only kinks).
            edge:  sensitivity alpha in ``exp(-alpha * |grad img|)``.

        The K axis is folded into the batch, so smoothness is enforced
        INDEPENDENTLY per hypothesis — each trajectory is made internally coherent
        while the hypotheses stay free to differ from one another.
        """
        if flow.dim() == 4:
            flow = flow.unsqueeze(1)
        B, K, H, W, _ = flow.shape
        f = flow.permute(0, 1, 4, 2, 3).reshape(B * K, 2, H, W)  # [BK, 2, H, W]
        if img.shape[-2:] != (H, W):
            img = F.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)
        img = img.unsqueeze(1).expand(B, K, -1, -1, -1).reshape(B * K, img.shape[1], H, W)

        def dx(t):
            return t[:, :, :, 1:] - t[:, :, :, :-1]

        def dy(t):
            return t[:, :, 1:, :] - t[:, :, :-1, :]

        wx = torch.exp(-edge * dx(img).abs().mean(1, keepdim=True))
        wy = torch.exp(-edge * dy(img).abs().mean(1, keepdim=True))
        fx, fy = dx(f), dy(f)
        if order >= 2:
            fx, wx = dx(fx), wx[:, :, :, 1:]
            fy, wy = dy(fy), wy[:, :, 1:, :]
        loss_x = (fx.abs().mean(1, keepdim=True) * wx).mean()
        loss_y = (fy.abs().mean(1, keepdim=True) * wy).mean()
        return loss_x + loss_y

    def _compute_flow_smoothness_loss(self):
        """Edge-aware spatial smoothness on the network's predicted interp flows.

        Independent of the RAFT teacher — uses only the exposed interp flows and
        the input LR frames — so it applies even when flow supervision is off or
        has decayed. Directly penalizes flying pixels (incoherent per-pixel flow),
        with an image-edge weight that permits genuine motion discontinuities at
        object boundaries. Bracket-anchored ``l2v``/``r2v`` are on the source (LR
        bracket) grids, so their edge weights are exact; legacy virtual-anchored
        ``v2l``/``v2r`` are weighted by the nearer bracket as a proxy (no image
        exists at the virtual time).
        """
        if self.lambda_flow_smooth <= 0:
            return None
        net = self.get_bare_model(self.netG)
        if not hasattr(net, "_aux_interp_flows") or not net._aux_interp_flows:
            return None
        t_in = self.L.shape[1]
        order, edge = self.flow_smooth_order, self.flow_smooth_edge
        total = self.L.new_tensor(0.0)
        n_terms = 0
        for gap_idx, gap_flows in enumerate(net._aux_interp_flows):
            if gap_idx + 1 >= t_in:
                continue
            img_l, img_r = self.L[:, gap_idx], self.L[:, gap_idx + 1]
            for flow_dict in gap_flows:
                if "l2v" in flow_dict and "r2v" in flow_dict:
                    total = total + self._edge_aware_flow_smoothness(flow_dict["l2v"], img_l, order, edge)
                    total = total + self._edge_aware_flow_smoothness(flow_dict["r2v"], img_r, order, edge)
                    n_terms += 2
                elif "v2l" in flow_dict and "v2r" in flow_dict:
                    total = total + self._edge_aware_flow_smoothness(flow_dict["v2l"], img_l, order, edge)
                    total = total + self._edge_aware_flow_smoothness(flow_dict["v2r"], img_r, order, edge)
                    n_terms += 2
        if n_terms == 0:
            return None
        return self.lambda_flow_smooth * total / n_terms

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
        self.log_dict["PSNR"] = self._detach_log_value(psnr)

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
            elif self.opt_train["G_optimizer_type"] == "adamw":
                self.G_optimizer = torch.optim.AdamW(
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
            elif self.opt_train["G_optimizer_type"] == "adamw":
                self.G_optimizer = torch.optim.AdamW(
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

        # Pre-compute RAFT pseudo-GT once and reuse it for (a) optional
        # teacher-forcing of the corrector and (b) the flow supervision loss.
        # Skip entirely when neither client needs it (e.g. after teacher decay
        # if lambda_flow==0).
        self._current_pseudo_gt = None
        self._use_teacher_this_step = False
        self._current_flow_loss_weight = self._flow_loss_weight(current_step) if self.raft_model is not None else 0.0
        p = self._teacher_prob(current_step) if self.raft_model is not None else 0.0
        if self.raft_model is not None and self.tsf > 1 and (self._current_flow_loss_weight > 0 or p > 0):
            self._current_pseudo_gt = self._compute_pseudo_gt_flows()
            if p > 0 and torch.rand(()).item() < p:
                self._use_teacher_this_step = True
                self.log_dict["teacher_forced"] = 1.0
            else:
                self.log_dict["teacher_forced"] = 0.0
            self.log_dict["teacher_prob"] = p
            self.log_dict["flow_loss_weight"] = self._current_flow_loss_weight

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
            with self._autocast_context():
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
        size_patch_testing = self.opt["val"].get("size_patch_testing", None)

        if size_patch_testing:
            assert size_patch_testing % window_size[-1] == 0, "testing patch size should be a multiple of window_size."

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
                        with self._autocast_context():
                            out_patch = self.netE(in_patch).detach().cpu()
                    else:
                        with self._autocast_context():
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
                with self._autocast_context():
                    output = self.netE(lq).detach().cpu()
            else:
                with self._autocast_context():
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
