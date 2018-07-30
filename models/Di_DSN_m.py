import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class Di_DSN_Model(BaseModel):
    def name(self):
        return 'Di_DSN_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Re', 'AC']


        self.model_names = ['ZE', 'AC', 'De']
        self.load_model_names_DA = ['SE', 'C']


        # load/define networks                
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC = networks.define_C(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netZE = networks.define_ZE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netAC = networks.define_AC(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netDe = networks.define_De(opt.output_nc, opt.input_nc, opt.which_model_netDe, opt.which_method, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_networks_DA(opt.which_usename_DA, opt.which_epochs_DA)
        self.nets = [self.netZE, self.netDe, self.netAC]



        if self.isTrain:
                                                
            # initialize optimizers
            #self.optimizer_AC = torch.optim.Adam((itertools.chain(self.netAC.parameters()), lr=opt.lr)
            #self.optimizer_Re = torch.optim.SGD(itertools.chain(self.netZE.parameters(), self.netDe.parameters(), self.netAC.parameters()), lr=opt.lr, momentum = 0.9)
            self.AC_params = list(self.netAC.parameters())
            self.Re_params = list(self.netZE.parameters()) + list(self.netDe.parameters()) + list(self.netAC.parameters())
            self.optimizer_Re = torch.optim.Adam(self.Re_params, lr=opt.lr)
            self.optimizer_AC = torch.optim.SGD(self.AC_params, lr=opt.lr, momentum=0.9)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_AC)
            self.optimizers.append(self.optimizer_Re)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, opt, s_img, s_label, t_img):
        
        if len(self.gpu_ids) > 0:
            s_img = s_img.cuda(self.gpu_ids[0], async=True)
            s_label = s_label.cuda(self.gpu_ids[0], async=True).view(-1)
            t_img = t_img.cuda(self.gpu_ids[0], async=True)

            
        self.s_img = Variable(s_img)
        self.t_img = Variable(t_img)        
        self.s_label = Variable(s_label)
        self.img_G = torch.cat((self.s_img, self.t_img), 0)
        t_e = self.netSE(self.t_img)
        self.t_label = torch.max(self.netC(t_e), 1)[1]
        self.label_s_t = torch.cat((self.s_label, self.t_label), 0)

    def reset_grad(self):
        for net in self.nets:
            net.zero_grad()


    def forward(self):
        self.code_s = self.netSE(self.img_G)
        self.code_z = self.netZE(self.img_G)
        code = torch.cat((self.code_s, self.code_z), 1)
        self.Reimg = self.netDe(code).view(-1, 3, 28, 28)
        self.pred_label = self.netAC(self.code_z)


    def backward_Re(self):
        AC_loss = torch.nn.functional.cross_entropy(self.pred_label, self.label_s_t)
        criterionD = nn.L1Loss()
        Re_loss = criterionD(self.Reimg, self.img_G)
        loss_t = Re_loss - 7*AC_loss
        loss_t.backward()
        return AC_loss.data[0], Re_loss.data[0], loss_t.data[0]

    def backward_AC(self):
        AC_loss = torch.nn.functional.cross_entropy(self.pred_label, self.label_s_t)
        AC_loss.backward()
        return AC_loss.data[0]

    def optimize_parameters(self, step):

        if step % 4 == 0:
            self.forward()
            self.AC_loss, self.Re_loss, self.loss_t = self.backward_Re()
            self.optimizer_Re.step()
            self.reset_grad()
        else:
            self.forward()
            self.AC_loss = self.backward_AC()
            self.optimizer_AC.step()
            self.reset_grad()
        return self.Reimg, self.AC_loss, self.Re_loss, self.loss_t


    def combine(self, opt):

        code_s_source, code_s_target = torch.split(self.code_s, opt.batchSize/2, 0)
        code_z_source, code_z_target = torch.split(self.code_z, opt.batchSize/2, 0)
        code_s1_tar, code_s2_tar = torch.split(code_s_target, opt.batchSize / 4, 0)
        code_snew_tar = torch.cat((code_s2_tar, code_s1_tar), 0)
        code_mid = torch.cat((code_s_source, code_z_target), 1)#.contiguous()
        codenew = torch.cat((code_snew_tar, code_z_target), 1)#.contiguous()
        Comimg = self.netDe(codenew).view(-1, 3, 28, 28)
        Commid = self.netDe(code_mid).view(-1, 3, 28, 28)

        return Comimg, Commid
        
