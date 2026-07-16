import torch

from models.model_plain import ModelPlain
from models.model_elvsr import ModelELVSR


class ModelBFSTVSR(ModelELVSR):
    """Train BF-STVSR (CVPR 2025) from scratch inside the KAIR harness.

    Reuses ``ModelELVSR`` for the sparse-GT / tsf plumbing (``feed_data``,
    ``test``, ``_n_out``) but replaces the training forward/loss with BF-STVSR's
    own recipe:

      * BF-STVSR runs its OWN internal (frozen) RAFT teacher on the HR GT frames
        for scheduled-sampling of the forward-splat flow, so ``ModelELVSR``'s RAFT
        pseudo-GT path is intentionally left dormant (do not set
        ``lambda_flow_supervision`` / ``teacher_flow_p0`` — keep ``raft_model`` None).
      * Scheduled sampling: with probability ``max(0, 1 - step/teacher_decay)`` the
        GT flow is used to warp features (``use_GT=True``); this decays to 0.
      * Optional flow-supervision loss (the ``bfstvsr_w_flow`` variant):
        ``bfstvsr_flow_loss_weight * (1 - (step % teacher_decay)/teacher_decay)``
        Charbonnier between the predicted (student) and RAFT (teacher) HR flows.

    Config keys (under ``opt["train"]``):
      * ``bfstvsr_teacher`` (bool, default True) — enable GT-flow scheduled sampling.
      * ``bfstvsr_teacher_decay`` (int, default 150000) — decay horizon (native period).
      * ``bfstvsr_flow_loss_weight`` (float, default 0.0) — 0.1 reproduces w_flow, 0 the plain variant.
    """

    def __init__(self, opt):
        super(ModelBFSTVSR, self).__init__(opt)
        t = self.opt_train
        self.bfstvsr_teacher = bool(t.get("bfstvsr_teacher", True))
        self.bfstvsr_teacher_decay = max(1, int(t.get("bfstvsr_teacher_decay", 150000)))
        self.bfstvsr_flow_loss_weight = float(t.get("bfstvsr_flow_loss_weight", 0.0))
        # Per-step caches populated in optimize_parameters.
        self._bf_use_gt = False
        self._bf_flow_w = 0.0

    # ----------------------------------------
    # scheduling
    # ----------------------------------------
    def _bf_teacher_prob(self, step):
        if not self.bfstvsr_teacher:
            return 0.0
        return max(0.0, 1.0 - step / self.bfstvsr_teacher_decay)

    def _bf_flow_weight(self, step):
        if self.bfstvsr_flow_loss_weight <= 0:
            return 0.0
        d = self.bfstvsr_teacher_decay
        return self.bfstvsr_flow_loss_weight * max(0.0, 1.0 - (step % d) / d)

    def optimize_parameters(self, current_step):
        # Decide this step's teacher-forcing / flow-loss weight, then run the plain
        # optimisation loop. We call ModelPlain (not ModelELVSR) directly to bypass
        # ELVSR's RAFT pseudo-GT machinery — BF-STVSR uses its own internal RAFT.
        p = self._bf_teacher_prob(current_step)
        self._bf_use_gt = bool(self.bfstvsr_teacher and p > 0 and torch.rand(()).item() < p)
        self._bf_flow_w = self._bf_flow_weight(current_step)
        self.log_dict["bf_teacher_prob"] = p
        self.log_dict["bf_use_gt"] = float(self._bf_use_gt)
        self.log_dict["bf_flow_w"] = self._bf_flow_w
        ModelPlain.optimize_parameters(self, current_step)

    # ----------------------------------------
    # forward (train): feed GT frames + scheduled use_GT to the BF-STVSR net
    # ----------------------------------------
    def netG_forward(self):
        kwargs = {
            "interp_steps": self.interp_steps,
            "teacher_enabled": self.bfstvsr_teacher,
            "use_GT": self._bf_use_gt,
        }
        if self.bfstvsr_teacher:
            # HR GT frames (ascending structural order) — the adapter gathers the
            # per-gap endpoint + intermediate frames for the internal RAFT teacher.
            kwargs["gt_frames"] = self.H
        # cupy / DCN kernels require fp32; force-disable autocast around the net
        # even if AMP is enabled elsewhere in the harness.
        with torch.autocast(device_type="cuda", enabled=False):
            self.E = self.netG(self.L, **kwargs)

    # ----------------------------------------
    # loss: pixel loss (structural order, 1:1 with H) + optional flow supervision
    # ----------------------------------------
    def compute_G_loss(self):
        if self.gt_indices is not None:
            B, K = self.gt_indices.shape
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)
            E_sel = self.E[b_idx, self.gt_indices]
            G_loss = self.G_lossfn_weight * self.G_lossfn(E_sel, self.H)
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

        if self._bf_flow_w > 0:
            net = self.get_bare_model(self.netG)
            flows = getattr(net, "last_flows", None)
            if flows is not None:
                student, teacher = flows  # teacher has no grad (RAFT runs under no_grad)
                flow_loss = self._bf_flow_w * self.G_lossfn(student, teacher)
                self.log_dict["bf_flow_loss"] = self._detach_log_value(flow_loss)
                G_loss = G_loss + flow_loss

        return G_loss
