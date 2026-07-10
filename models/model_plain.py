from collections import OrderedDict
from contextlib import nullcontext
import glob
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.amp import GradScaler, autocast

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierFourierLoss, CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt["train"]  # training option
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        self.netG = self.maybe_compile_forward(self.netG, module_name="netG")
        if self.opt_train["E_decay"] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    def _resolve_amp_dtype(self, dtype_name):
        if dtype_name is None:
            return torch.float16
        key = str(dtype_name).strip().lower()
        if key in ("float16", "fp16", "half"):
            return torch.float16
        if key in ("bfloat16", "bf16"):
            return torch.bfloat16
        print(f"Unknown amp_dtype={dtype_name}, falling back to float16.")
        return torch.float16

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)

    def _detach_log_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        return value

    def _materialize_log_dict(self):
        materialized = OrderedDict()
        for key, value in self.log_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    materialized[key] = value.item()
                else:
                    materialized[key] = value.detach().float().mean().item()
            else:
                materialized[key] = value
        return materialized

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

        # mixed precision
        amp_requested = bool(self.opt_train.get("use_amp", False))
        self.use_amp = bool(amp_requested and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype(self.opt_train.get("amp_dtype", "float16"))
        if self.use_amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("AMP bf16 requested but unsupported on this GPU. Falling back to float16.")
            self.amp_dtype = torch.float16
        if amp_requested and not self.use_amp:
            print("AMP requested but CUDA is unavailable. Running in full precision.")

        if self.use_amp:
            self.scaler = GradScaler("cuda", enabled=True)
            self.load_scaler()
            print(f"Mixed precision (AMP) training enabled (dtype={self.amp_dtype}).")
        else:
            self.scaler = None

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt["path"]["pretrained_netG"]
        if load_path_G is not None:
            print("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train["G_param_strict"], param_key="params")
        load_path_E = self.opt["path"]["pretrained_netE"]
        if self.opt_train["E_decay"] > 0:
            if load_path_E is not None:
                print("Loading model for E [{:s}] ...".format(load_path_E))
                self.load_network(
                    load_path_E, self.netE, strict=self.opt_train["E_param_strict"], param_key="params_ema"
                )
            else:
                print("Copying model for E ...")
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt["path"]["pretrained_optimizerG"]
        if load_path_optimizerG is not None and self.opt_train["G_optimizer_reuse"]:
            print("Loading optimizerG [{:s}] ...".format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, "G", iter_label)
        if self.opt_train["E_decay"] > 0:
            self.save_network(self.save_dir, self.netE, "E", iter_label)
        if self.opt_train["G_optimizer_reuse"]:
            self.save_optimizer(self.save_dir, self.G_optimizer, "optimizerG", iter_label)
        if self.use_amp:
            scaler_path = os.path.join(self.save_dir, "{}_{}.pth".format(iter_label, "scaler"))
            torch.save(self.scaler.state_dict(), scaler_path)

    def load_scaler(self):
        scaler_files = glob.glob(os.path.join(self.opt["path"]["models"], "*_scaler.pth"))
        if scaler_files:
            latest = max(scaler_files, key=os.path.getmtime)
            print("Loading scaler [{:s}] ...".format(latest))
            self.scaler.load_state_dict(torch.load(latest, weights_only=True))

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train["G_lossfn_type"]
        if G_lossfn_type == "l1":
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == "l2":
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == "l2sum":
            self.G_lossfn = nn.MSELoss(reduction="sum").to(self.device)
        elif G_lossfn_type == "ssim":
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == "charbonnier":
            self.G_lossfn = CharbonnierLoss(self.opt_train["G_charbonnier_eps"]).to(self.device)
        elif G_lossfn_type in ["charbonnier_fft", "charbonnier_hf"]:
            self.G_lossfn = CharbonnierFourierLoss(
                eps=self.opt_train["G_charbonnier_eps"],
                fft_weight=self.opt_train["G_fft_loss_weight"],
                fft_loss_type=self.opt_train["G_fft_loss_type"],
                fft_mask_radius=self.opt_train["G_fft_mask_radius"],
                fft_mask_ratio=self.opt_train["G_fft_mask_ratio"],
            ).to(self.device)
        else:
            raise NotImplementedError("Loss type [{:s}] is not found.".format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train["G_lossfn_weight"]

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))
        if self.opt_train["G_optimizer_type"] == "adam":
            self.G_optimizer = Adam(
                G_optim_params,
                lr=self.opt_train["G_optimizer_lr"],
                betas=self.opt_train["G_optimizer_betas"],
                weight_decay=self.opt_train["G_optimizer_wd"],
            )
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train["G_scheduler_type"] == "MultiStepLR":
            self.schedulers.append(
                lr_scheduler.MultiStepLR(
                    self.G_optimizer, self.opt_train["G_scheduler_milestones"], self.opt_train["G_scheduler_gamma"]
                )
            )
        elif self.opt_train["G_scheduler_type"] == "CosineAnnealingWarmRestarts":
            self.schedulers.append(
                lr_scheduler.CosineAnnealingWarmRestarts(
                    self.G_optimizer,
                    self.opt_train["G_scheduler_periods"],
                    self.opt_train["G_scheduler_restart_weights"],
                    self.opt_train["G_scheduler_eta_min"],
                )
            )
        elif self.opt_train["G_scheduler_type"] == "CosineAnnealingRestartLR":
            from basicsr.models.lr_scheduler import CosineAnnealingRestartLR

            self.schedulers.append(
                CosineAnnealingRestartLR(
                    self.G_optimizer,
                    periods=self.opt_train["G_scheduler_periods"],
                    restart_weights=self.opt_train["G_scheduler_restart_weights"],
                    eta_min=self.opt_train["G_scheduler_eta_min"],
                )
            )
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data["L"].to(self.device, non_blocking=True)
        if need_H:
            self.H = data["H"].to(self.device, non_blocking=True)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # compute generator loss (can be overridden by subclasses)
    # ----------------------------------------
    def compute_G_loss(self):
        return self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()

        # When training under HuggingFace Accelerate, mixed precision, gradient
        # scaling and grad-sync are owned by the accelerator: backward and
        # clipping route through it and the model's own AMP scaler is bypassed.
        accelerator = getattr(self, "accelerator", None)

        with self._autocast_context():
            self.netG_forward()
            G_loss = self.compute_G_loss()

        # Fail loud on a non-finite loss instead of letting NaN/inf flow through
        # backward() and silently corrupt the weights (Adam moments never recover).
        # This surfaces upstream numerical bugs at the step they first appear.
        if not torch.isfinite(G_loss):
            raise FloatingPointError(
                f"Non-finite G_loss ({G_loss.item()}) at step {current_step}; "
                "aborting before optimizer step corrupts the weights."
            )

        if accelerator is not None:
            accelerator.backward(G_loss)
        elif self.use_amp:
            self.scaler.scale(G_loss).backward()
        else:
            G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train["G_optimizer_clipgrad"] if self.opt_train["G_optimizer_clipgrad"] else 0
        if G_optimizer_clipgrad > 0:
            if accelerator is not None:
                accelerator.clip_grad_norm_(self.netG.parameters(), max_norm=G_optimizer_clipgrad)
            else:
                if self.use_amp:
                    self.scaler.unscale_(self.G_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.netG.parameters(), max_norm=self.opt_train["G_optimizer_clipgrad"], norm_type=2
                )

        if accelerator is not None:
            self.G_optimizer.step()
        elif self.use_amp:
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        else:
            self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = (
            self.opt_train["G_regularizer_orthstep"] if self.opt_train["G_regularizer_orthstep"] else 0
        )
        if (
            G_regularizer_orthstep > 0
            and current_step % G_regularizer_orthstep == 0
            and current_step % self.opt["train"]["checkpoint_save"] != 0
        ):
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = (
            self.opt_train["G_regularizer_clipstep"] if self.opt_train["G_regularizer_clipstep"] else 0
        )
        if (
            G_regularizer_clipstep > 0
            and current_step % G_regularizer_clipstep == 0
            and current_step % self.opt["train"]["checkpoint_save"] != 0
        ):
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict["G_loss"] = self._detach_log_value(G_loss)

        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            with self._autocast_context():
                self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            with self._autocast_context():
                self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt["scale"], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self._materialize_log_dict()

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach()[0].float().cpu()
        out_dict["E"] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach().float().cpu()
        out_dict["E"] = self.E.detach().float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    # ----------------------------------------
    # Compute PSNR
    # ----------------------------------------
    def log_psnr(self):
        # Flatten the first two dimensions (batch and sequence) for visualization
        E_flat = self.E.detach().float().flatten(0, 1)
        H_flat = self.H.detach().float().flatten(0, 1)
        mse = nn.MSELoss()(E_flat, H_flat)
        psnr = 10 * torch.log10(1 / mse)
        self.log_dict["PSNR"] = self._detach_log_value(psnr)
