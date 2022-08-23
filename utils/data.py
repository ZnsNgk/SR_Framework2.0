import torch
import os, random
import cv2
import numpy, math
from torch.utils.data import Dataset, DataLoader
from .bool import get_bool
from.logs import log

class SR_dataset_RGB(Dataset):
    def __init__(self, HR_folder, LR_folder, scale, patch_size, opts, batchsize, train=True, is_post=True, normal=False, is_real=False, test_patch=None, n=1):
        self.scale = scale
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.is_real = is_real
        if self.is_real:
            self.HR_folder += (str(self.scale) + '/')
            self.LR_folder += (str(self.scale) + '/')
        else:
            self.LR_folder += (str(self.scale) + '/')
        self.hr_img = os.listdir(self.HR_folder)
        self.lr_img = os.listdir(self.LR_folder)
        assert self.hr_img == self.lr_img
        self.patch_size = patch_size
        self.train = train
        self.is_post = is_post
        self.normal = normal
        self.cut = test_patch
        self.opts = opts
        if self.opts["repeat_factor"] == 0:
            self.repeat = 1
        else:
            self.repeat = int(self.opts["repeat_factor"] / (len(self.hr_img)/batchsize)) * n
        self.hflip = get_bool(self.opts["horizontal_flip"])
        self.wflip = get_bool(self.opts["vertical_flip"])
        self.rotate = get_bool(self.opts["rotate"])
    def get_idx(self, idx):
        if self.train:
            return idx % len(self.hr_img)
        else:
            return idx
    def __len__(self):
        if self.train:
            return len(self.hr_img) * self.repeat
        else:
            return len(self.hr_img)
    def get_patch(self, lr, hr):
        ih, iw = lr.shape[:2]
        ip = self.patch_size
        ix = random.randrange(0, (iw-ip))
        iy = random.randrange(0, (ih-ip))
        if self.is_post:
            tp = int(self.scale* self.patch_size)
            tx, ty = int(self.scale * ix), int(self.scale * iy)
        else:
            tx, ty = ix, iy
            tp = ip
        return lr[iy:iy + ip, ix:ix + ip, :], hr[ty:ty + tp, tx:tx + tp, :]
        # h, w, _ = lr.shape
        # randh = random.randint(0, h - self.patch_size)
        # randw = random.randint(0, w - self.patch_size)
        # toh = randh + self.patch_size
        # tow = randw + self.patch_size
        # lr_patch = lr[randh:toh, randw:tow ,:]
        # if self.is_post:
        #     hr_patch = hr[randh*self.scale : toh*self.scale, randw*self.scale : tow*self.scale, :]
        # else:
        #     hr_patch = hr[randh : toh, randw : tow, :]
        # return lr_patch,  hr_patch
    def cut_pic(self, lr):
        [c, h_lr, w_lr] = lr.shape
        h_n = math.ceil(h_lr / self.cut)
        w_n = math.ceil(w_lr / self.cut)
        patch_num = h_n * w_n
        lr_patch = torch.zeros([patch_num, c, self.cut, self.cut])
        n = 0
        for i in range(h_n):
            for j in range(w_n):
                if ((i + 1) == h_n) and ((j + 1) == w_n):
                    lr_patch[n, :, :, :] = lr[ :, h_lr-self.cut:, w_lr-self.cut:]
                elif (i + 1) == h_n:
                    lr_patch[n, :, :, :] = lr[:, h_lr-self.cut:, j*self.cut:(j+1)*self.cut]
                elif (j + 1) == w_n:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, w_lr-self.cut:]
                else:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, j*self.cut:(j+1)*self.cut]
                n += 1
        return lr_patch, [h_lr, w_lr]
    def augment(self, *args):
        hflip = self.hflip and random.random() < 0.5
        vflip = self.wflip and random.random() < 0.5
        rot90 = self.rotate and random.random() < 0.5
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            return img.copy()
        return [_augment(a) for a in args]
    def __getitem__(self, index):
        index = self.get_idx(index)
        hr_name = self.hr_img[index]
        lr_name = self.lr_img[index]
        hr_path = os.path.join(self.HR_folder, hr_name)
        lr_path = os.path.join(self.LR_folder, lr_name)
        hr = cv2.imread(hr_path)
        lr = cv2.imread(lr_path)
        if self.opts["color_seq"] == "RGB":
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        if self.train:
            lr, hr = self.get_patch(lr, hr)
            hr, lr = self.augment(hr, lr)
        hr = torch.from_numpy(hr)
        lr = torch.from_numpy(lr)
        hr = hr.permute(2,0,1).float()
        lr = lr.permute(2,0,1).float()
        if self.normal:
            hr = hr / 255.
            lr = lr / 255.
        if not self.cut == None:
            lr, shape = self.cut_pic(lr)
            return lr, hr, shape
        return lr, hr

class SR_dataset_Y(SR_dataset_RGB):
    def cut_pic(self, lr):
        [c, h_lr, w_lr] = lr.shape
        h_n = math.ceil(h_lr / self.cut)
        w_n = math.ceil(w_lr / self.cut)
        patch_num = h_n * w_n
        lr_patch = torch.zeros([patch_num, c, self.cut, self.cut])
        n = 0
        for i in range(h_n):
            for j in range(w_n):
                if ((i + 1) == h_n) and ((j + 1) == w_n):
                    lr_patch[n, :, :, :] = lr[ :, h_lr-self.cut:, w_lr-self.cut:]
                elif (i + 1) == h_n:
                    lr_patch[n, :, :, :] = lr[:, h_lr-self.cut:, j*self.cut:(j+1)*self.cut]
                elif (j + 1) == w_n:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, w_lr-self.cut:]
                else:
                    lr_patch[n, :, :, :] = lr[:, i*self.cut:(i+1)*self.cut, j*self.cut:(j+1)*self.cut]
                n += 1
        return lr_patch, [h_lr, w_lr]
    def __get_Y(self, img):
        m = numpy.array([24.966, 128.553, 65.481], dtype='float32')
        shape = img.shape
        if len(shape) == 3:
            img = img.reshape((shape[0] * shape[1], 3))
            shape = shape[0 : 2]
        y = numpy.dot(img, m.transpose() / 255.)
        y += 16.
        y = y.reshape(shape)
        return y
    def __getitem__(self, index):
        index = self.get_idx(index)
        hr_name = self.hr_img[index]
        lr_name = self.lr_img[index]
        hr_path = os.path.join(self.HR_folder, hr_name)
        lr_path = os.path.join(self.LR_folder, lr_name)
        hr = cv2.imread(hr_path)
        lr = cv2.imread(lr_path)
        if self.train:
            lr, hr = self.get_patch(lr, hr)
            hr, lr = self.augment(hr, lr)
        hr = self.__get_Y(hr)
        lr = self.__get_Y(lr)
        hr = torch.from_numpy(hr)
        lr = torch.from_numpy(lr)
        hr = hr.unsqueeze(0).float()
        lr = lr.unsqueeze(0).float()
        if self.normal:
            hr = hr / 255.
            lr = lr / 255.
        if not self.cut == None:
            lr, shape = self.cut_pic(lr)
            return lr, hr, shape
        return lr, hr

class SR_demo(Dataset):
    def __init__(self, folder, scale, color_seq, normal=False, is_input=False, is_Y=False):
        self.folder = folder
        self.scale = scale
        self.normal = normal
        self.is_input = is_input
        self.is_Y = is_Y
        self.color_seq = color_seq
        if not self.is_input:
            if is_Y:
                self.folder += '_Y'
            self.folder += '_LR/'
            self.folder += str(self.scale)
        self.folder += '/'
        self.img = os.listdir(self.folder)
    def get_Y(self, img):
        if self.color_seq == "RGB":
            m = numpy.array([65.481, 128.553, 24.966], dtype='float32')
        elif self.color_seq == "BGR":
            m = numpy.array([24.966, 128.553, 65.481], dtype='float32')
        shape = img.shape
        if len(shape) == 3:
            img = img.reshape((shape[0] * shape[1], 3))
            shape = shape[0 : 2]
        y = numpy.dot(img, m.transpose() / 255.)
        y += 16.
        y = y.reshape(shape)
        return y
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        name = self.img[index]
        path = os.path.join(self.folder, name)
        if self.is_Y and (not self.is_input):
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread(path)
            if self.color_seq == "RGB":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.is_input and self.is_Y:
            im = self.get_Y(im)
        im = torch.from_numpy(im)
        if self.is_Y:
            im = im.unsqueeze(0).float()
        else:
            im = im.permute(2,0,1).float()
        if self.normal:
            im = im / 255.
        return im, name

def get_demo_loader(folder, scale, color_seq, normal, is_input, is_Y):
    data = SR_demo(folder, scale, color_seq, normal, is_input, is_Y)
    return DataLoader(data, 1, False, num_workers=0, drop_last=False, pin_memory=False)

class Data():
    def __init__(self, sys_conf, data_config, train=True, val=False, test_patch=None, val_dataset=None):
        self.train = train
        self.val = val
        self.model_name = sys_conf.model_name
        self.root = sys_conf.data_root
        self.dataset = sys_conf.dataset
        self.batch_size = (sys_conf.batch_size if train else 1)
        self.scale = 1
        self.normal = get_bool(data_config["normalize"])
        self.shuffle = (get_bool(data_config["shuffle"]) if self.train else False)
        self.n = 1
        if self.train:
            self.DD_parallel = sys_conf.DD_parallel
            if self.DD_parallel:
                self.n_GPUs = sys_conf.n_GPUs
                self.local_rank = sys_conf.local_rank
                self.n = self.n_GPUs
                self.batch_size = self.batch_size // self.n_GPUs
        self.pic_pair = False
        self.test_patch = test_patch
        if self.val:
            self.dataset = val_dataset
        if sys_conf.color_channel == "RGB":
            self.color_is_RGB = True
        elif sys_conf.color_channel == "Y":
            self.color_is_RGB = False
        if sys_conf.model_mode == "pre":
            self.is_post = False
        elif sys_conf.model_mode == "post":
            self.is_post = True
        self.patch_size = sys_conf.patch_size
        self.num_workers = (data_config["num_workers"] if self.train else 0)
        self.drop_last = False
        if "drop_last" in data_config:
            self.drop_last = (get_bool(data_config["drop_last"])if self.train else False)
        self.pin_memory = True
        if "pic_pair" in data_config:
            self.pic_pair = get_bool(data_config["pic_pair"])
        if "pin_memory" in data_config:
            self.pin_memory = get_bool(data_config["pin_memory"])
        self.opts = {}
        if "data_opts" in data_config:
            self.opts = data_config["data_opts"]
        self.__check_dataopts()
    def show(self):
        if self.DD_parallel and (not self.local_rank == 0):
            return
        log("--------This is dataset and dataloader config--------", self.model_name)
        log("Dataloader num workers: " + str(self.num_workers), self.model_name)
        log("Dataset is pair: " + str(self.pic_pair), self.model_name)
        log("Shuffle: " + str(self.shuffle), self.model_name)
        log("Drop the last batch: " + str(self.drop_last), self.model_name)
        log("Using pin menory: " + str(self.pin_memory), self.model_name)
        log("Using normalization: " + str(self.normal), self.model_name)
        log("Data opts: " + str(self.opts), self.model_name)
    def __check_dataopts(self):
        keys = ["color_seq", "rotate", "horizontal_flip", "vertical_flip", "repeat_factor"]
        for key in keys:
            if not key in self.opts:
                if key == "color_seq":
                    self.opts[key] = "RGB"
                elif key == "rotate":
                    self.opts[key] = "False"
                elif key == "repeat_factor":
                    self.opts[key] = 1000
                elif "flip" in key:
                    self.opts[key] = "False"
        if (not self.opts["color_seq"] == "RGB") and (not self.opts["color_seq"] == "BGR"):
            raise ValueError("Color_seq only can be set in RGB or BGR")
    def __set_dataset_path(self):
        if self.root == "./":
            self.root += "data/"
        if not self.pic_pair:
            HR_folder = self.root
            if self.train:
                HR_folder += 'train/'
            else:
                if self.val:
                    HR_folder += 'val/'
                else:
                    HR_folder += 'test/'
            HR_folder += self.dataset
            LR_folder = HR_folder + '_LR/'
            HR_folder += '/'
        else:
            HR_folder = self.root
            LR_folder = self.root
            if self.train:
                HR_folder += 'train/'
                LR_folder += 'train/'
            else:
                if self.val:
                    HR_folder += 'val/'
                    LR_folder += 'val/'
                else:
                    HR_folder += 'test/'
                    LR_folder += 'test/'
            HR_folder += self.dataset
            LR_folder += self.dataset
            HR_folder += '/HR'
            LR_folder += '/LR'
            HR_folder += '/'
            LR_folder += '/'
        return LR_folder, HR_folder
    def update_scale(self, scale):
        self.scale = scale
    def get_loader(self):
        LR_folder, HR_folder = self.__set_dataset_path()
        if self.color_is_RGB:
            data = SR_dataset_RGB(HR_folder, LR_folder, self.scale, self.patch_size, self.opts, self.batch_size, self.train, self.is_post, self.normal, self.pic_pair, self.test_patch, self.n)
        else:
            data = SR_dataset_Y(HR_folder, LR_folder, self.scale, self.patch_size, self.opts, self.batch_size, self.train, self.is_post, self.normal, self.pic_pair, self.test_patch, self.n)
        if self.train:
            if self.DD_parallel:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(data, shuffle=self.shuffle)
                loader = DataLoader(data, self.batch_size, shuffle=False, sampler=sampler, num_workers=self.num_workers, 
                                drop_last=self.drop_last, pin_memory=self.pin_memory)
                return loader, sampler
        loader = DataLoader(data, self.batch_size, self.shuffle, num_workers=self.num_workers, 
                            drop_last=self.drop_last, pin_memory=self.pin_memory)
        return loader
    def update_dataset(self, dataset):
        self.dataset = dataset