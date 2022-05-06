import torch
import numpy, random
import utils
import shutil
import os
from tqdm import tqdm
import models

class Trainer():
    def __init__(self, sys_conf, data, val, lr_conf, break_point):
        self.sys_conf = sys_conf
        self.data = data
        self.init_lr = lr_conf["init_learning_rate"]
        self.decay_mode = lr_conf["decay_mode"]
        self.lr_reset = True
        self.per_epoch = None
        self.decay_rate = None
        self.eta_min = None
        self.break_point = break_point
        self.is_break = self.__check_breakpoint()
        self.loss_func = self.sys_conf.get_loss()
        self.curr_scale = 0
        self.curr_scale_list_pos = 0
        self.cal_psnr = utils.get_loss_PSNR(self.sys_conf.loss_function, self.data.normal)
        self.val = val
        if "per_epoch" in lr_conf:
            self.per_epoch = lr_conf["per_epoch"]
        if "decay_rate" in lr_conf:
            self.decay_rate = lr_conf["decay_rate"]
        if "eta_min" in lr_conf:
            self.eta_min = lr_conf["eta_min"]
        if "learning_rate_reset" in lr_conf:
            self.lr_reset = utils.get_bool(lr_conf["learning_rate_reset"])
        if self.is_break:
            utils.log("The train will be continue!", self.sys_conf.model_name)
        else:
            self.__check_folder()
            utils.check_log_file(self.sys_conf.model_name)
            self.show()
    def __set_seed(self):
        utils.log("-------------Now Setting Seed-------------", self.sys_conf.model_name)
        random.seed(self.sys_conf.seed)
        utils.log("Random seed is set with "+ str(self.sys_conf.seed), self.sys_conf.model_name, True)
        numpy.random.seed(self.sys_conf.seed)
        utils.log("Numpy seed is set with "+ str(self.sys_conf.seed), self.sys_conf.model_name, True)
        torch.manual_seed(self.sys_conf.seed)
        try:
            torch.cuda.manual_seed_all(self.sys_conf.seed)
            utils.log("GPU seed is set with "+ str(self.sys_conf.seed), self.sys_conf.model_name, True)
        except:
            utils.log("GPU seeding failed!"+ str(self.sys_conf.seed), self.sys_conf.model_name, True)
        utils.log("Torch seed is set with "+ str(self.sys_conf.seed), self.sys_conf.model_name, True)
    def show(self):
        self.sys_conf.show()
        self.data.show()
        self.val.show(self.sys_conf.model_name)
        utils.log("--------This is learning rate and decay config-------", self.sys_conf.model_name)
        utils.log("Init learning rate: " + str(self.init_lr), self.sys_conf.model_name)
        utils.log("Learning rate reset per scale: " + str(self.lr_reset), self.sys_conf.model_name)
        utils.log("Learning rate decay mode: " + str(self.decay_mode), self.sys_conf.model_name)
        if self.per_epoch != None:
            utils.log("The learning rate will decay every " + str(self.per_epoch) + " epochs", self.sys_conf.model_name)
        if self.decay_rate != None:
            utils.log("Learnging rate decay rate is " + str(self.decay_rate), self.sys_conf.model_name)
        if self.eta_min != None:
            utils.log("The minimum learning rate is " + str(self.eta_min), self.sys_conf.model_name)
    def __check_breakpoint(self):
        if self.break_point != None:
            self.break_path = './trained_model/' + self.sys_conf.model_name + '/' + self.break_point
            state_list = self.break_point.split('_')
            self.breakpoint_scale = int(state_list[1].replace('x', ''))
            self.breakpoint_epoch = int(state_list[2].replace('.pth', ''))
            self.para = torch.load(self.break_path)
            utils.log("Setting breakpoint at " + self.break_point, self.sys_conf.model_name)
            return True
        return False
    def __get_model(self):
        if self.data.is_post:
            if self.sys_conf.scale_pos == "init":
                if self.sys_conf.model_args == None:
                    model = models.get_model(self.sys_conf.model, scale=self.curr_scale)
                else:
                    model = models.get_model(self.sys_conf.model, scale=self.curr_scale, **self.sys_conf.model_args)
            elif self.sys_conf.scale_pos == "forward":
                if self.sys_conf.model_args == None:
                    model = models.get_model(self.sys_conf.model)
                else:
                    model = models.get_model(self.sys_conf.model, **self.sys_conf.model_args)
        else:
            if self.sys_conf.model_args == None:
                model = models.get_model(self.sys_conf.model)
            else:
                model = models.get_model(self.sys_conf.model, **self.sys_conf.model_args)
        if self.sys_conf.parallel:
            model = torch.nn.parallel.DataParallel(model)
        return model
    def __check_folder(self):
        trained_model_path = './trained_model/' + self.sys_conf.model_name + '/'
        if not os.path.exists(trained_model_path):
            os.makedirs(trained_model_path)
        else:
            print("WARNING: The current directory already exists. Do you want to override it?")
            s = input("[yes/no]\n>>>")
            if s == "no":
                exit()
            elif s == "yes":
                shutil.rmtree(trained_model_path)
                os.makedirs(trained_model_path)
            else:
                raise NameError("WRONG INPUT!")
    def __update_scale(self, curr_scale):
        self.curr_scale = curr_scale
        self.data.update_scale(curr_scale)
        if self.val.use_val:
            self.val.val_data.update_scale(curr_scale)
    def __set_optim(self, m):
        trainable = filter(lambda x: x.requires_grad, m.parameters())
        if self.sys_conf.optim_args == None:
            return utils.get_optimizer(self.sys_conf.optim, trainable, self.init_lr)
        else:
            return utils.get_optimizer(self.sys_conf.optim, trainable, self.init_lr, **self.sys_conf.optim_args)
    def _set_scheduler(self, optim):
        return utils.get_scheduler(self.decay_mode, optim, step_size=self.per_epoch, gamma=self.decay_rate, eta_min=self.eta_min)
    def train(self):
        if self.sys_conf.seed != None:
            self.__set_seed()
        if self.is_break:
            position = self.sys_conf.scale_factor.index(self.breakpoint_scale)
            for _ in range(position):
                 self.sys_conf.scale_factor.pop(0)
                 self.curr_scale_list_pos += 1
        else:
            utils.log("-------------Now Starting train-------------", self.sys_conf.model_name)
            utils.log("Training is started", self.sys_conf.model_name, True)
        torch.set_grad_enabled(True)
        if self.sys_conf.model_mode == "post":
            if self.sys_conf.scale_pos == "init":
                self.__train_scale_pos_is_init()
            elif self.sys_conf.scale_pos == "forward":
                self.__train_scale_pos_is_forward()
            else:
                raise NameError("WRONG MODEL SCALE POSITION!")
        elif self.sys_conf.model_mode == "pre":
            self.__train_model_mode_is_pre()
        else:
            raise NameError("WRONG MODEL MODE!")
    def __train_scale_pos_is_forward(self):
        net = self.__get_model()
        loss_func = self.sys_conf.get_loss()
        if not self.lr_reset:
            optim = self.__set_optim(net)
            scheduler = self._set_scheduler(optim)
        if self.is_break:
            para = torch.load(self.break_path)
            if self.sys_conf.parallel:
                net.module.load_state_dict(para)
            else:
                net.load_state_dict(para)
        else:
            utils.init_weights(self.sys_conf.weight_init, net, self.sys_conf.model_name)
        net = net.to(self.sys_conf.device_in_prog)
        net.train()
        for scale in self.sys_conf.scale_factor:
            self.curr_scale_list_pos += 1
            if self.lr_reset:
                optim = self.__set_optim(net)
                scheduler = self._set_scheduler(optim)
            else:
                if self.curr_scale_list_pos > 1:
                    if self.is_break:
                        for _ in range(self.sys_conf.Epoch):
                            scheduler.step()
                        optim_state = optim.state_dict()
                        scheduler_state = scheduler.state_dict()
                    else:
                        optim.load_state_dict(optim_state)
                        scheduler.load_state_dict(scheduler_state)
            self.__update_scale(scale)
            train_data = self.data.get_loader()
            if self.val.use_val:
                val_loader = self.val.val_data.get_loader()
            for epoch in range(1, self.sys_conf.Epoch + 1):
                if self.is_break and epoch <= self.breakpoint_epoch:
                    scheduler.step()
                    continue
                running_loss = 0.0
                i = 0
                with tqdm(train_data, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                    for lr, hr in t:
                        i += 1
                        optim.zero_grad()
                        if self.sys_conf.parallel:
                            lr = lr.to(self.sys_conf.device_in_prog)
                            hr = hr.cuda()
                        else:
                            lr = lr.to(self.sys_conf.device_in_prog)
                            hr = hr.to(self.sys_conf.device_in_prog)
                        sr = net(lr, scale)
                        loss = loss_func(sr, hr)
                        loss.backward()
                        optim.step()
                        running_loss += float(loss)
                        t.set_postfix(loss = float(loss))
                avg_loss = running_loss / i
                psnr = self.cal_psnr(avg_loss)
                scheduler.step()
                if self.val.use_val:
                    if epoch == 1:
                        best_loss = numpy.Inf
                    else:
                        try:
                            best_loss = torch.load('./trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                            best_loss = best_loss["loss"]
                        except:
                            utils.log("Cannot find best log file, reset best loss to Inf", self.sys_conf.model_name, True)
                            best_loss = numpy.Inf
                    val_loss = 0.0
                    v = 0
                    with torch.no_grad():
                        with tqdm(val_loader, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                            v += 1
                            for lr, hr in t:
                                if self.sys_conf.parallel:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.cuda()
                                else:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.to(self.sys_conf.device_in_prog)
                            sr = net(lr, scale)
                            loss = loss_func(sr, hr)
                            val_loss += float(loss)
                            t.set_postfix(loss = float(loss))
                    val_avg_loss = val_loss / v
                    val_psnr = self.cal_psnr(val_avg_loss)
                    if val_avg_loss < best_loss:
                        best_loss = {"loss": val_avg_loss}
                        torch.save(best_loss, './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                        utils.log("Best loss update to "+str(round(val_avg_loss, 4))+", saving parameters to x"+str(self.curr_scale) + "_best.pth", self.sys_conf.model_name, True)
                        PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_best.pth'
                        if self.sys_conf.parallel:
                            torch.save(net.module.state_dict(), PATH)
                        else:
                            torch.save(net.state_dict(), PATH)
                s = "Epoch: " + str(epoch) + "| running loss: " + str(round(avg_loss, 4)) + "| PSNR: " + str(round(psnr, 2)) + "db"
                if self.val.use_val:
                    s += "|val loss: " + str(round(val_avg_loss, 4)) + "| val PSNR: " + str(round(val_psnr, 2)) + "db"
                s += "| Curr lr: " + str(round(optim.state_dict()['param_groups'][0]['lr'], 6)) + "| Now scale: " + str(self.curr_scale)
                utils.log(s, self.sys_conf.model_name, True)
                if epoch % self.sys_conf.save_step == 0:
                    PATH = './trained_model/' + self.sys_conf.model_name + '/net_x'+ str(self.curr_scale) + '_' + str(epoch) + '.pth'
                    if self.sys_conf.parallel:
                        torch.save(net.module.state_dict(), PATH)
                    else:
                        torch.save(net.state_dict(), PATH)
            if self.curr_scale_list_pos == 1:
                optim_state = optim.state_dict()
                scheduler_state = scheduler.state_dict()
            utils.log("Finished trained scale " + str(self.curr_scale), self.sys_conf.model_name, True)
            PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '.pkl'
            if self.sys_conf.parallel:
                torch.save(net.module, PATH)
            else:
                torch.save(net, PATH)
            self.is_break = False
    def __train_scale_pos_is_init(self):
        loss_func = self.sys_conf.get_loss()
        for scale in self.sys_conf.scale_factor:
            self.__update_scale(scale)
            net = self.__get_model()
            net.train()
            if self.is_break:
                para = torch.load(self.break_path)
                if self.sys_conf.parallel:
                    net.module.load_state_dict(para)
                else:
                    net.load_state_dict(para)
            else:
                utils.init_weights(self.sys_conf.weight_init, net, self.sys_conf.model_name)
            optim = self.__set_optim(net)
            scheduler = self._set_scheduler(optim)
            net = net.to(self.sys_conf.device_in_prog)
            train_data = self.data.get_loader()
            if self.val.use_val:
                val_loader = self.val.val_data.get_loader()
            for epoch in range(1, self.sys_conf.Epoch + 1):
                if self.is_break and epoch <= self.breakpoint_epoch:
                    scheduler.step()
                    continue
                running_loss = 0.0
                i = 0
                with tqdm(train_data, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                    for lr, hr in t:
                        i += 1
                        optim.zero_grad()
                        if self.sys_conf.parallel:
                            lr = lr.cuda()
                            hr = hr.cuda()
                        else:
                            lr = lr.to(self.sys_conf.device_in_prog)
                            hr = hr.to(self.sys_conf.device_in_prog)
                        sr = net(lr)
                        loss = loss_func(sr, hr)
                        loss.backward()
                        optim.step()
                        running_loss += float(loss)
                        t.set_postfix(loss = float(loss))
                avg_loss = running_loss / i
                psnr = self.cal_psnr(avg_loss)
                if self.val.use_val:
                    if epoch == 1:
                        best_loss = numpy.Inf
                    else:
                        try:
                            best_loss = torch.load('./trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                            best_loss = best_loss["loss"]
                        except:
                            utils.log("Cannot find best log file, reset best loss to Inf", self.sys_conf.model_name, True)
                            best_loss = numpy.Inf
                    val_loss = 0.0
                    v = 0
                    with torch.no_grad():
                        with tqdm(val_loader, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                            v += 1
                            for lr, hr in t:
                                if self.sys_conf.parallel:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.cuda()
                                else:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.to(self.sys_conf.device_in_prog)
                            sr = net(lr)
                            loss = loss_func(sr, hr)
                            val_loss += float(loss)
                            t.set_postfix(loss = float(loss))
                    val_avg_loss = val_loss / v
                    val_psnr = self.cal_psnr(val_avg_loss)
                    if val_avg_loss < best_loss:
                        best_loss = {"loss": val_avg_loss}
                        torch.save(best_loss, './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                        utils.log("Best loss update to "+str(round(val_avg_loss, 4))+", saving parameters to x"+str(self.curr_scale) + "_best.pth", self.sys_conf.model_name, True)
                        PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_best.pth'
                        if self.sys_conf.parallel:
                            torch.save(net.module.state_dict(), PATH)
                        else:
                            torch.save(net.state_dict(), PATH)
                s = "Epoch: " + str(epoch) + "| running loss: " + str(round(avg_loss, 4)) + "| PSNR: " + str(round(psnr, 2)) + "db"
                if self.val.use_val:
                    s += "|val loss: " + str(round(val_avg_loss, 4)) + "| val PSNR: " + str(round(val_psnr, 2)) + "db"
                s += "| Curr lr: " + str(round(optim.state_dict()['param_groups'][0]['lr'], 6)) + "| Now scale: " + str(self.curr_scale)
                utils.log(s, self.sys_conf.model_name, True)
                scheduler.step()
                if epoch % self.sys_conf.save_step == 0:
                    PATH = './trained_model/' + self.sys_conf.model_name + '/net_x'+ str(self.curr_scale) + '_' + str(epoch) + '.pth'
                    if self.sys_conf.parallel:
                        torch.save(net.module.state_dict(), PATH)
                    else:
                        torch.save(net.state_dict(), PATH)
            utils.log("Finished trained scale " + str(self.curr_scale), self.sys_conf.model_name, True)
            PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '.pkl'
            if self.sys_conf.parallel:
                torch.save(net.module, PATH)
            else:
                torch.save(net, PATH)
            self.is_break = False
    def __train_model_mode_is_pre(self):
        net = self.__get_model()
        loss_func = self.sys_conf.get_loss()
        if not self.lr_reset:
            optim = self.__set_optim(net)
            scheduler = self._set_scheduler(optim)
        if self.is_break:
            para = torch.load(self.break_path)
            if self.sys_conf.parallel:
                net.module.load_state_dict(para)
            else:
                net.load_state_dict(para)
        else:
            utils.init_weights(self.sys_conf.weight_init, net, self.sys_conf.model_name)
        net = net.to(self.sys_conf.device_in_prog)
        net.train()
        for scale in self.sys_conf.scale_factor:
            self.curr_scale_list_pos += 1
            self.__update_scale(scale)
            if self.lr_reset:
                optim = self.__set_optim(net)
                scheduler = self._set_scheduler(optim)
            else:
                if self.curr_scale_list_pos > 1:
                    if self.is_break:
                        for _ in range(self.sys_conf.Epoch):
                            scheduler.step()
                        optim_state = optim.state_dict()
                        scheduler_state = scheduler.state_dict()
                    else:
                        optim.load_state_dict(optim_state)
                        scheduler.load_state_dict(scheduler_state)
            train_data = self.data.get_loader()
            if self.val.use_val:
                val_loader = self.val.val_data.get_loader()
            for epoch in range(1, self.sys_conf.Epoch + 1):
                if self.is_break and epoch <= self.breakpoint_epoch:
                    scheduler.step()
                    continue
                running_loss = 0.0
                i = 0
                with tqdm(train_data, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                    for lr, hr in t:
                        i += 1
                        optim.zero_grad()
                        if self.sys_conf.parallel:
                            lr = lr.cuda()
                            hr = hr.cuda()
                        else:
                            lr = lr.to(self.sys_conf.device_in_prog)
                            hr = hr.to(self.sys_conf.device_in_prog)
                        sr = net(lr)
                        loss = loss_func(sr, hr)
                        loss.backward()
                        optim.step()
                        running_loss += float(loss)
                        t.set_postfix(loss = float(loss))
                avg_loss = running_loss / i
                psnr = self.cal_psnr(avg_loss)
                if self.val.use_val:
                    if epoch == 1:
                        best_loss = numpy.Inf
                    else:
                        try:
                            best_loss = torch.load('./trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                            best_loss = best_loss["loss"]
                        except:
                            utils.log("Cannot find best log file, reset best loss to Inf", self.sys_conf.model_name, True)
                            best_loss = numpy.Inf
                    val_loss = 0.0
                    v = 0
                    with torch.no_grad():
                        with tqdm(val_loader, desc="Epoch "+str(epoch), ncols=100, leave=False) as t:
                            v += 1
                            for lr, hr in t:
                                if self.sys_conf.parallel:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.cuda()
                                else:
                                    lr = lr.to(self.sys_conf.device_in_prog)
                                    hr = hr.to(self.sys_conf.device_in_prog)
                            sr = net(lr)
                            loss = loss_func(sr, hr)
                            val_loss += float(loss)
                            t.set_postfix(loss = float(loss))
                    val_avg_loss = val_loss / v
                    val_psnr = self.cal_psnr(val_avg_loss)
                    if val_avg_loss < best_loss:
                        best_loss = {"loss": val_avg_loss}
                        torch.save(best_loss, './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_loss.pth')
                        utils.log("Best loss update to "+str(round(val_avg_loss, 4))+", saving parameters to x"+str(self.curr_scale) + "_best.pth", self.sys_conf.model_name, True)
                        PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '_best.pth'
                        if self.sys_conf.parallel:
                            torch.save(net.module.state_dict(), PATH)
                        else:
                            torch.save(net.state_dict(), PATH)
                s = "Epoch: " + str(epoch) + "| running loss: " + str(round(avg_loss, 4)) + "| PSNR: " + str(round(psnr, 2)) + "db"
                if self.val.use_val:
                    s += "|val loss: " + str(round(val_avg_loss, 4)) + "| val PSNR: " + str(round(val_psnr, 2)) + "db"
                s += "| Curr lr: " + str(round(optim.state_dict()['param_groups'][0]['lr'], 6)) + "| Now scale: " + str(self.curr_scale)
                utils.log(s, self.sys_conf.model_name, True)
                scheduler.step()
                if epoch % self.sys_conf.save_step == 0:
                    PATH = './trained_model/' + self.sys_conf.model_name + '/net_x'+ str(self.curr_scale) + '_' + str(epoch) + '.pth'
                    if self.sys_conf.parallel:
                        torch.save(net.module.state_dict(), PATH)
                    else:
                        torch.save(net.state_dict(), PATH)
            if self.curr_scale_list_pos == 1:
                optim_state = optim.state_dict()
                scheduler_state = scheduler.state_dict()
            utils.log("Finished trained scale " + str(self.curr_scale), self.sys_conf.model_name, True)
            PATH = './trained_model/' + self.sys_conf.model_name + '/x'+ str(self.curr_scale) + '.pkl'
            if self.sys_conf.parallel:
                torch.save(net.module, PATH)
            else:
                torch.save(net, PATH)
            self.is_break = False
