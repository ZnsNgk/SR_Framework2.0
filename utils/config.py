import os
import torch
from torch.cuda import is_available
from torch import device
from .logs import log
from .loss_func import get_loss_func
from .bool import get_bool

class sys_config():
    def __init__(self, args, cfg, data_root, local_rank, train=True):
        self.train = train
        self.model = args
        self.data_root = data_root
        self.data_root = self.data_root.replace("\\", "/")
        if self.data_root[-1] != "/":
            self.data_root = self.data_root + "/"
        self.model_name = cfg["model_name"]
        self.model_mode = cfg["model_mode"]
        self.color_channel = cfg["color_channel"]
        self.Epoch = cfg["Epoch"]
        self.scale_pos = "init"
        self.device = "cuda:0"
        self.device_in_prog = None
        self.model_args = None
        self.save_step = cfg["save_step"]
        self.scale_factor = cfg["scale_factor"]
        self.dataset = cfg["dataset"]
        self.patch_size = cfg["patch_size"]
        self.__set_scale()
        self.optim_args = None
        self.parallel = False
        self.DD_parallel = False
        self.seed = None
        self.parallel_mode = None
        self.backend = None
        if "device" in cfg:
            self.device = cfg["device"]
        if "model_args" in cfg:
            self.model_args = cfg["model_args"]
        if "scale_position" in cfg:
            self.scale_pos = cfg["scale_position"]
        if self.train:
            self.batch_size = cfg["batch_size"]
            self.mini_batch = 0
            self.weight_init = cfg["weight_init"]
            self.loss_function = cfg["loss_function"]
            self.optim = cfg["optimizer"]
            self.loss_args = None
            if "loss_args" in cfg:
                self.loss_args = cfg["loss_args"]
            if "optim_args" in cfg:
                self.optim_args = cfg["optim_args"]
            if "mini_batch" in cfg:
                self.mini_batch = cfg["mini_batch"]
            if "parallel_opts" in cfg:
                if "backend" in cfg["parallel_opts"]:
                    self.backend = cfg["parallel_opts"]["backend"]
                if "parallel_mode" in cfg["parallel_opts"]:
                    self.parallel_mode = cfg["parallel_opts"]["parallel_mode"]
                    if self.parallel_mode == "DDP":
                        self.DD_parallel = True
                        self.local_rank = local_rank
                    elif self.DD_parallel == "DP":
                        self.DD_parallel = False
                        self.backend = None
        if "seed" in cfg:
            self.seed = cfg["seed"]
        self.__check_cuda()
        if self.DD_parallel and self.train:
            torch.distributed.init_process_group(backend=self.backend)
            self.device_in_prog = device(self.device_in_prog, self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            self.device_in_prog = device(self.device_in_prog)
    def __check_cuda(self):
        if "cuda" in self.device:
            cuda_idx = self.device.split(':')
            cuda_idx = cuda_idx[1].replace(' ', '')
            if self.train:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx[0]
            if not is_available():
                print(self.device + " is not useable, now try to use cpu!")
                self.device = "cpu"
                self.device_in_prog = 'cpu'
            else:
                if not cuda_idx == "0":
                    cuda_idx_int = []
                    if ',' in cuda_idx:
                        idx = cuda_idx.split(',')
                        for name in idx:
                            cuda_idx_int.append(int(name))
                    else:
                        cuda_idx_int.append(int(cuda_idx))
                    if len(cuda_idx_int) > 1:
                        if self.train:
                            self.parallel = True
                            if self.DD_parallel:
                                self.parallel_mode = "DDP"
                                self.n_GPUs = len(cuda_idx_int)
                            else:
                                self.parallel_mode = "DP"
                            self.device_in_prog = "cuda"
                        else:
                            self.device_in_prog = "cuda:0"
                    else:
                        self.device_in_prog = "cuda:0"
                else:
                    self.device_in_prog = "cuda:0"
        elif ("mkldnn" or "opengl" or "opencl" or "ideep" or "hip" or "msnpu" or "xla" or "mps") in self.device:
            self.device_in_prog = self.device
        else:
            self.device = "cpu"
            self.device_in_prog = "cpu"
        if not self.parallel:
            self.parallel_mode = None
            self.backend = None
    def __set_scale(self):
        if not isinstance(self.scale_factor, list):
            scale_list = []
            scale_list.append(self.scale_factor)
            self.scale_factor = scale_list
    def show(self):
        if self.DD_parallel and (not self.local_rank == 0):
            return
        log("-------------This is system config--------------", self.model_name)
        log("Model: " + self.model, self.model_name)
        if self.data_root != "./":
            log("Data Root: " + self.data_root, self.model_name)
        log("Dataset: " + self.dataset, self.model_name)
        log("Upsample Position: " + self.model_mode, self.model_name)
        log("Color Channel: " + self.color_channel, self.model_name)
        log("Batch Size: " + str(self.batch_size), self.model_name)
        if self.mini_batch != 0:
            log("Mini Batch: " + str(self.mini_batch), self.model_name)
        log("Patch Size: " + str(self.patch_size), self.model_name)
        log("Training Epoch: " + str(self.Epoch), self.model_name)
        log("Training Device:" + str(self.device), self.model_name)
        if self.parallel_mode:
            log("Parallel mode:" + str(self.parallel_mode), self.model_name)
        if self.backend and self.DD_parallel:
            log("Parallel init backend:" + str(self.backend), self.model_name)
        log("Training Scale: " + str(self.scale_factor), self.model_name)
        log("Trained Model Save Step: " + str(self.save_step), self.model_name)
        log("Weight Init: " + self.weight_init, self.model_name)
        log("Loss Function: " + self.loss_function, self.model_name)
        log("Optimizer: " + self.optim, self.model_name)
        log("Position of Upsample Method in Model: " + self.scale_pos, self.model_name)
        if self.model_args != None:
            log("Model args: " + str(self.model_args), self.model_name)
        if self.loss_args != None:
            log("Loss function args: " + str(self.loss_args), self.model_name)
        if self.optim_args != None:
            log("Optimizer args: " + str(self.optim_args), self.model_name)
        if self.seed != None:
            log("Seed: " + str(self.seed), self.model_name)
    def get_loss(self):
        if self.loss_args == None:
            return get_loss_func(self.loss_function)
        else:
            return get_loss_func(self.loss_function, self.loss_args)
    def set_test_config(self, args, test_cfg):
        self.scale_factor = list(dict.fromkeys(self.scale_factor))
        self.test_color_channel = test_cfg["color_channel"]
        self.test_all = False
        self.test_best = args.best
        self.shave = 0
        self.shave_is_scale = False
        self.patch = None
        self.indicators = ['PSNR', 'SSIM']
        if "shave" in test_cfg:
            if test_cfg["shave"] == "scale":
                self.shave_is_scale = True
            else:
                self.shave = test_cfg["shave"]
        if "patch" in test_cfg:
            if test_cfg["patch"] == 0:
                self.patch = None
            else:
                self.patch = test_cfg["patch"]
        if "indicators" in test_cfg:
            self.indicators = test_cfg["indicators"]
        if args.all:
            self.drew = get_bool(test_cfg["drew_pic"])
            self.test_all = True
        else:
            self.drew = False
        self.test_dataset = test_cfg["test_dataset"]
        self.test_file = ""
        if args.once != None:
            self.test_file = args.once
        if args.dataset != None:
            self.test_dataset = [args.dataset]

class val_config(sys_config):
    def __init__(self, cfg={}, default=False):
        self.use_val = False
        self.dataset = ""
        self.multi_device = False
        self.split = 0
        if default:
            pass
        else:
            if "use_val" in cfg:
                self.use_val = get_bool(cfg["use_val"])
            if "val_dataset" in cfg:
                self.dataset = cfg["val_dataset"]
            if "split" in cfg:
                self.split = cfg["split"]
            if "multi_device" in cfg:
                if torch.cuda.is_available() and self.split != 0:
                    self.multi_device = get_bool(cfg["multi_device"])
    
    def set_val_data(self, val_data):
        self.val_data = val_data
    
    def show(self, name, DD_parallel, local_rank):
        if DD_parallel and (not local_rank == 0):
            return
        log("-------------This is validation config--------------", name)
        log("Enable validation: " + str(self.use_val), name)
        if self.use_val:
            log("Validation dataset: " + str(self.dataset), name)
            log("Validation multi device: " + str(self.multi_device), name)
            if self.split != 0:
                log("Validation split: " + str(self.split), name)