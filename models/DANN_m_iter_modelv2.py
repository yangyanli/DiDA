import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class DANN_m_iter_Modelv2(BaseModel):
    def name(self):
        return 'DANN_m_iter_Modelv2'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Domain', 'C']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['SE', 'C', 'DP']
        self.load_model_names_DA = ['SE', 'C']
        self.load_model_names_Di = ['ZE', 'De']
        self.load_model_names_Pre = ['SE']

        # load/define networks
        self.netSEPre = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netSE = networks.define_SE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC = networks.define_C(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm, not opt.no_dropout,
                                      opt.init_type, self.gpu_ids)
        self.netZE = networks.define_ZE(opt.input_nc, opt.output_nc, opt.which_model_netSE, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netAC = networks.define_AC(opt.output_nc, opt.input_nc, opt.which_model_netC, opt.norm, not opt.no_dropout,
                                        opt.init_type, self.gpu_ids)
        self.netDe = networks.define_De(opt.output_nc, opt.input_nc, opt.which_model_netDe, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netDP = networks.define_DP(opt.output_nc, opt.input_nc, opt.which_model_netDP, opt.norm,
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_networks_DA(opt.which_usename_DA, opt.which_epochs_DA)
        self.load_networks_Di(opt.which_usename_Di, opt.which_epochs_Di)
        self.load_networks_Pre(opt.which_usename_DA, opt.which_epochs_DA)
        self.nets = [self.netSE, self.netDP, self.netC]
        



        if self.isTrain:
                                                
            # initialize optimizers
            self.C_params = list(self.netSE.parameters()) + list(self.netC.parameters())
            self.Domain_params = list(self.netSE.parameters()) + list(self.netDP.parameters())
            self.optimizer_C = torch.optim.SGD(self.C_params, lr=opt.lr/opt.lr_de, momentum = 0.9)#10
            self.optimizer_Domain = torch.optim.SGD(self.Domain_params, lr=opt.lr, momentum = 0.9)
            #self.optimizer_C = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netC.parameters()), lr=opt.lr/10, momentum = 0.9)
            #self.optimizer_Domain = torch.optim.SGD(itertools.chain(self.netSE.parameters(), self.netDP.parameters()), lr=opt.lr, momentum = 0.9)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_C)
            self.optimizers.append(self.optimizer_Domain)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, opt, s_img, s_label, t_img):

        domain_t_label = torch.LongTensor(opt.batchSize/2).zero_() + 1
        domain_s_label = torch.LongTensor(opt.batchSize/2*2).zero_()
        domain_label = torch.cat((domain_s_label, domain_t_label), 0)
        
        if len(self.gpu_ids) > 0:
            s_img = s_img.cuda(self.gpu_ids[0], async=True)
            s_label = s_label.cuda(self.gpu_ids[0], async=True).view(-1)
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            domain_label = domain_label.cuda(self.gpu_ids[0], async=True)
            
        self.s_img = Variable(s_img)
        self.t_img = Variable(t_img)        
        self.s_label = Variable(s_label)
        self.domain_label = Variable(domain_label)

        self.mid_s = self.netSEPre(self.s_img)
        self.mid_z = self.netZE(self.t_img)

        self.mid_code = torch.cat((self.mid_s, self.mid_z), 1)#.contiguous()
        self.mid_img = Variable(self.netDe(self.mid_code).view(-1, 3, 28, 28).data)
        self.mid_label = self.s_label

        self.s_mid_label = torch.cat((self.s_label, self.mid_label), 0)
        self.s_mid_img = torch.cat((self.s_img, self.mid_img), 0)

        self.img_G = torch.cat((self.s_img, self.mid_img, self.t_img), 0)

        return self.img_G

    def forward_C(self):
        self.E_s_mid = self.netSE(self.s_mid_img)
        self.pred_label = self.netC(self.E_s_mid)

    def forward_Domain(self):
        self.E_G = self.netSE(self.img_G)
        self.domain_pred = self.netDP(self.E_G)

    def backward_C(self):
        m = nn.Softmax()
        loss = nn.NLLLoss()
        C_loss = loss(m(self.pred_label), self.s_mid_label)
        #C_loss = torch.nn.functional.cross_entropy(self.pred_label, self.s_mid_label)
        C_loss.backward()
        return C_loss.data[0]

    def backward_Domain(self):
        Domain_loss = torch.nn.functional.cross_entropy(self.domain_pred, self.domain_label)
        Domain_loss.backward()
        return Domain_loss.data[0]

    
    def optimize_parameters(self):
        
        self.forward_C()                
        C_loss = self.backward_C()        
        self.optimizer_C.step()
        self.optimizer_C.zero_grad()

        self.forward_Domain()                
        Domain_loss = self.backward_Domain()        
        self.optimizer_Domain.step()
        self.optimizer_Domain.zero_grad()

        return C_loss, Domain_loss
        
    def test(self, t_img, t_label):

        if len(self.gpu_ids) > 0:
            t_img = t_img.cuda(self.gpu_ids[0], async=True)
            t_label = t_label.cuda(self.gpu_ids[0], async=True).view(-1)
                                           
        t_img = Variable(t_img)        
        t_label = Variable(t_label)

        E_t = self.netSE(t_img)
        pred_label_t = self.netC(E_t)
        
        pred = pred_label_t.data.max(1)[1]
        correct_t = pred.eq(t_label.data).cpu().sum()

        return correct_t
        
