import os
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ModelBase:
    def __init__(self, opt):
        self.opt = opt  # opt
        self.save_dir = opt["path"]["models"]  # save models
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")

        print("Device: {}".format(self.device))
        self.is_train = opt["is_train"]  # training or not
        self.schedulers = []  # schedulers
        # HuggingFace Accelerate handle. Left None for single-GPU / native-DDP
        # training; the accelerate launcher sets it on the model so the backward
        # pass, grad clipping and checkpoint unwrapping route through it.
        self.accelerator = None

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. Wraps model with DistributedDataParallel (DDP) or DataParallel (DP)."""
        network = network.to(self.device)

        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", True)
            use_static_graph = self.opt.get("use_static_graph", False)

            # Get the correct GPU for this process
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            network.to(device)

            network = DistributedDataParallel(
                network,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
            )

            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()

        elif self.opt.get("use_dp", False) and torch.cuda.device_count() > 1:
            # Only wrap in DataParallel when explicitly requested. Under Accelerate
            # (opt["dist"] is False but accelerator.prepare wraps netG in DDP), an
            # implicit DataParallel here would double-wrap and scatter params to
            # cuda:0, breaking multi-process runs (params land on cuda:<local_rank>).
            network = DataParallel(network)

        return network

    def _torch_compile_cfg(self):
        """Return normalized torch.compile config dict."""
        train_opt = self.opt.get("train", {}) if isinstance(self.opt, dict) else {}
        cfg = train_opt.get("torch_compile", None)

        if isinstance(cfg, bool):
            cfg = {"enable": cfg}
        elif cfg is None:
            cfg = {"enable": bool(train_opt.get("torch_compile_enable", False))}
        elif not isinstance(cfg, dict):
            cfg = {"enable": False}

        cfg.setdefault("enable", False)
        cfg.setdefault("backend", "inductor")
        cfg.setdefault("mode", "reduce-overhead")
        cfg.setdefault("dynamic", True)
        cfg.setdefault("fullgraph", False)
        cfg.setdefault("disable", False)
        cfg.setdefault("cuda_only", True)
        return cfg

    def maybe_compile_forward(self, module, module_name="module", compile_cfg=None):
        """Compile a module forward path when enabled in options.

        This compiles only ``forward`` instead of wrapping the entire module,
        which keeps checkpoint keys unchanged.
        """
        cfg = self._torch_compile_cfg() if compile_cfg is None else dict(compile_cfg)
        if not cfg.get("enable", False):
            return module
        if cfg.get("disable", False):
            return module
        if cfg.get("cuda_only", True) and self.device.type != "cuda":
            print(f"[torch.compile] Skip {module_name}: CUDA-only requested on device={self.device}.")
            return module
        if not hasattr(torch, "compile"):
            print(f"[torch.compile] Skip {module_name}: torch.compile is unavailable in this PyTorch build.")
            return module

        bare = self.get_bare_model(module)
        if getattr(bare, "_forward_is_compiled", False):
            return module

        try:
            bare.forward = torch.compile(
                bare.forward,
                backend=cfg.get("backend", "inductor"),
                mode=cfg.get("mode", "reduce-overhead"),
                dynamic=bool(cfg.get("dynamic", True)),
                fullgraph=bool(cfg.get("fullgraph", False)),
                disable=bool(cfg.get("disable", False)),
            )
            bare._forward_is_compiled = True
            print(
                "[torch.compile] Compiled {}.forward "
                "(backend={}, mode={}, dynamic={}, fullgraph={})".format(
                    module_name,
                    cfg.get("backend", "inductor"),
                    cfg.get("mode", "reduce-overhead"),
                    bool(cfg.get("dynamic", True)),
                    bool(cfg.get("fullgraph", False)),
                )
            )
        except Exception as exc:
            print(f"[torch.compile] Failed to compile {module_name}.forward: {exc}")

        return module

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = "\n"
        msg += "Networks name: {}".format(network.__class__.__name__) + "\n"
        msg += "Params number: {}".format(sum(map(lambda x: x.numel(), network.parameters()))) + "\n"
        msg += "Net structure:\n{}".format(str(network)) + "\n"
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = "\n"
        msg += (
            " | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}".format("mean", "min", "max", "std", "shape", "param_name")
            + "\n"
        )
        for name, param in network.state_dict().items():
            if not "num_batches_tracked" in name:
                v = param.data.clone().float()
                msg += (
                    " | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}".format(
                        v.mean(), v.min(), v.max(), v.std(), v.shape, name
                    )
                    + "\n"
                )
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key="params"):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for (key_old, param_old), (key, param) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(
            torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        )

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
