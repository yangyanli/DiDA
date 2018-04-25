from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable, Function
import torchvision.models as models
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
#import sklearn
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from scipy.misc import imread, imresize, imsave
from skimage import io
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
import dataloader
import dataloaderm
#import dataloader_test
import dataloaderm_eval
import functools
import os


opt_gpu = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu




learning_rate = 0.001
num_epochs = 5000

mb_size = 144
input_nc = 1
output_nc = 2
heat_h = 28
heat_w = 28
threshold = 0.5

"""==========net E========="""
class NetE(nn.Module):
   
    def __init__(self, input_nc, output_nc):
        super(NetE, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(48)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2) 

        #self.conv3 = nn.Conv2d(48, 16, kernel_size = 3, stride = 1, padding = 0)
        #self.norm_layer3 = nn.BatchNorm2d(16)        
        #self.relu3 = nn.ReLU(True)
	#self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1200, 48)
        #self.lrelu1 = nn.LeakyReLU(0.2, True)


        #self.fc2 = nn.Linear(48, 48)
        #self.lrelu1 = nn.LeakyReLU(0.2, True)

    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):

        out = self.conv1(x)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

        #out = self.conv3(out)
        #print(out.size())
        #out = self.norm_layer3(out)
        #out = self.relu3(out)

        #print(out.size())
        out = out.view(-1, 1200)

        out = self.fc1(out)
       # out = self.lrelu1(out)

        #out = self.fc2(out)
        



        #for n in range(out.shape[0]):
            #sum = 0
            
            #for sum in range(out.shape[1]):
               # sum = sum + out[n][sum].data *  out[n][sum].data
            #sum = math.sqrt(sum)
            #out[n] = out[n]/sum


       
        return out


"""==============Net Domain_Ped========="""
class Domain_Pred(nn.Module):

	def __init__(self, input_nc, output_nc):
		super(Domain_Pred, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc

		self.fc1 = nn.Linear(48, 100)
                self.relu1 = nn.LeakyReLU(0.2, True)

  

                self.fc2 = nn.Linear(100, 2)
  
	def save_network(self, network, network_label, epoch_label, save_dir ):
		save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
    	        save_path = os.path.join(save_dir, save_filename)
    	        torch.save(network.state_dict(), save_path)

        def forward(self, x):

            x = flip_gradient(x)

    	    out = self.fc1(x)
    	   
      	    out = self.relu1(out)

    	    out = self.fc2(out)

    	    return out

class FlipGrad(Function):
       def forward(self, x):
              
              return x.view_as(x)
       def backward(self, grad_output):

             
          
              return grad_output.neg()
def flip_gradient(x):
       return FlipGrad()(x)









"""==========net G========="""
class NetG(nn.Module):
   
    def __init__(self, input_nc, output_nc):
        super(NetG, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        

        self.fc1 = nn.Linear(48, 100)
        self.relu1 = nn.LeakyReLU(0.2, True)

  

        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.LeakyReLU(0.2, True)

  

        self.fc3 = nn.Linear(100, 10)
 


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):

    	#out = self.fc1(x)
    	    
        x = x.contiguous()

    	out = self.fc1(x)
    	   
      	out = self.relu1(out)

    	out = self.fc2(out)
      	out = self.relu2(out)

    	out = self.fc3(out)


       
        return out


"""==========net ZEn========="""
class NetZEn(nn.Module):
   
    def __init__(self, input_nc, output_nc):
        super(NetZEn, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
        #28,28
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.norm_layer1 = nn.InstanceNorm2d(32)       
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 0)
        self.norm_layer2 = nn.InstanceNorm2d(64)        
        self.relu2 = nn.ReLU(True)
	self.pool2 = nn.MaxPool2d(2) 

        self.conv3 = nn.Conv2d(64, 16, kernel_size = 3, stride = 1, padding = 0)
        self.norm_layer3 = nn.InstanceNorm2d(16)        
        self.relu3 = nn.ReLU(True)
	#self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(144, 144)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(144, 144)
        #self.lrelu1 = nn.LeakyReLU(0.2, True)
        


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):

        out = self.conv1(x)
        #print(out.size())
        out = self.norm_layer1(out)
        out = self.relu1(out)
    	out = self.pool1(out)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.norm_layer2(out)
        out = self.relu2(out)
    	out = self.pool2(out)

        out = self.conv3(out)
        #print(out.size())
        out = self.norm_layer3(out)
        out = self.relu3(out)

        #print(out.size())
        out = out.view(-1, 144)

        out = self.fc1(out)
        out = self.lrelu1(out)

        out = self.fc2(out)
        

       
        return out


"""==========net De========="""
class NetDe(nn.Module):
   
    def __init__(self, input_nc, output_nc):
        super(NetDe, self).__init__() 
        self.input_nc = input_nc
        self.output_nc = output_nc
   
       
        self.fc1 = nn.Linear(192, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2, True)


        self.fc2 = nn.Linear(1024, 1024)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

        self.fc3 = nn.Linear(1024, 2048)
        self.lrelu3 = nn.LeakyReLU(0.2, True)
        

        self.fc4 = nn.Linear(2048, 2352)
        self.Ta = nn.Tanh()
        


    #def save_network(self, network, network_label, epoch_label, gpu_ids):
    def save_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])

   
    #def forward(self, input):
    def forward(self, x):



        out = self.fc1(x)
        out = self.lrelu1(out)

        out = self.fc2(out)
        out = self.lrelu2(out)

        out = self.fc3(out)
        out = self.lrelu3(out)

        out = self.fc4(out)
        out = self.Ta(out)

       
        return out





"""==============Net AClassifier========="""
class AClassifier(nn.Module):

	def __init__(self, input_nc, output_nc):
		super(AClassifier, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc

		self.fc1 = nn.Linear(144, 48)
                self.relu1 = nn.LeakyReLU(0.2, True)

		self.fc2 = nn.Linear(48, 48)
                self.relu2 = nn.LeakyReLU(0.2, True)


  

                self.fc3 = nn.Linear(48, 10)
  
	def save_network(self, network, network_label, epoch_label, save_dir ):
		save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
    	        save_path = os.path.join(save_dir, save_filename)
    	        torch.save(network.state_dict(), save_path)

        def forward(self, x):

            

    	    out = self.fc1(x)
    	   
      	    out = self.relu1(out)

    	    out = self.fc2(out)
    	   
      	    out = self.relu2(out)

    	    out = self.fc3(out)

    	    return out






"""===============TRAINING=================="""
netE = NetE(input_nc, output_nc).train()
netE.load_state_dict(torch.load('modelrgm/3197_net_netE2.pkl'))
netE.cuda()
print(netE)

netSEnPre = NetE(input_nc, output_nc).eval()
netSEnPre.load_state_dict(torch.load('modelrgm/3197_net_netE2.pkl'))
netSEnPre.cuda()
print(netSEnPre)

netG = NetG(input_nc, output_nc).train()
netG.load_state_dict(torch.load('modelrgm/3197_net_netG2.pkl')) 
netG.cuda()
print(netG)


netGP = NetG(input_nc, output_nc).eval()
netGP.load_state_dict(torch.load('modelrgm/3197_net_netG2.pkl')) 
netGP.cuda()
print(netGP)

netZEn = NetZEn(input_nc, output_nc).eval()
netZEn.load_state_dict(torch.load('modelgod/1170_net_netZEn1v2new.pkl')) 
netZEn.cuda()
print(netZEn)

netDe = NetDe(input_nc, output_nc).eval()
netDe.load_state_dict(torch.load('modelgod/1170_net_netDe1v2new.pkl')) 
netDe.cuda()
print(netDe)

netD_Pred = Domain_Pred(input_nc, output_nc).train()
netD_Pred.cuda()
print(netD_Pred)





E_params = list(netE.parameters()) + list(netD_Pred.parameters())

G_params = list(netG.parameters()) + list(netE.parameters())

nets = [netE, netD_Pred, netG]


root_path = "traindata/"
train_set_source = dataloader.get_training_set(root_path)
training_data_loader_s = dataloader.DataLoader(dataset=train_set_source, num_workers=2, batch_size = mb_size/3, shuffle= True)


train_set_target = dataloaderm.get_training_set(root_path)
training_data_loader_t = dataloaderm.DataLoader(dataset=train_set_target, num_workers=2, batch_size = mb_size/3, shuffle= True)


#test_set_source = dataloader_test.get_training_set(root_path)
#testing_data_loader_s = dataloader_test.DataLoader(dataset = test_set_source, num_workers = 2, batch_size = mb_size/2, shuffle = True)

test_set_target = dataloaderm_eval.get_training_set(root_path)
eval_data_loader_t = dataloaderm_eval.DataLoader(dataset = test_set_target, num_workers = 2, batch_size = mb_size/2, shuffle = True)



def reset_grad():
    for net in nets:
        net.zero_grad()


G_solver = optim.SGD(G_params, lr = learning_rate/10, momentum = 0.9)
E_solver = optim.SGD(E_params, lr = learning_rate, momentum = 0.9)



criterionD = nn.L1Loss()
criterionM = nn.MSELoss()

reset_grad()

m = nn.Softmax()
writer = SummaryWriter()




def eval(epoch):
        netE.eval()
        netG.eval()


        test_t_loss = 0
        correct_t = 0
       
	for step, (t_img, t_label, t_view) in enumerate(eval_data_loader_t):
        

			t_img = Variable(t_img.cuda())
			t_label = Variable(t_label.cuda()).view(-1)


                        E_t = netE(t_img)
                        #E_t_G, E_t_C = torch.split(E_t, 200, 1)
                        pred_label_t = netG(E_t)

                        loss = nn.NLLLoss()
    			G_loss = loss(m(pred_label_t), t_label)
                        
                        
                        test_t_loss += G_loss

                        pred = pred_label_t.data.max(1)[1]
                        correct_t += pred.eq(t_label.data).cpu().sum()

        test_t_loss /= len(training_data_loader_t)    
        Accuracy_t = float(correct_t)/float(len(eval_data_loader_t.dataset))             
        print('Epoch [%d/%d], test_t_loss: %.4f, Accuracy_t: %d/%d(%.4f)' %(epoch+1, num_epochs, test_t_loss.data[0], correct_t, len(eval_data_loader_t.dataset), (float(correct_t)/float(len(eval_data_loader_t.dataset)))))         
        writer.add_scalar('data/test_t_loss', test_t_loss.data[0], epoch)
        writer.add_scalar('data/correct_t', correct_t, epoch)
        writer.add_scalar('data/Accuracy_t', float(correct_t)/float(len(eval_data_loader_t.dataset)), epoch)
        return Accuracy_t


lr_ch = 1000
best_acc = 0
#thre = 45

"""===============start epoch==============="""
for epoch in range(num_epochs):


  acc = eval(epoch)
  if acc > best_acc:
    best_acc = acc
  
    netE.save_network(netE, 'netE1', epoch, 'modeliter/')
    
    netG.save_network(netG, 'netG1', epoch, 'modeliter/')
    
    


  netE.train()
  netG.train()

  j = 0
  
  t_dataset = iter(training_data_loader_t)
	
  num_batch = len(training_data_loader_s)
  total_iter = num_epochs * num_batch
 
  for step, (s_img, s_label, s_name) in enumerate(training_data_loader_s):

   i = epoch*num_batch + step
   p = float(i) / float(total_iter)
   lr_val = 0.00001 / (1. + 10 * p) ** 0.45

   def lr_scheduler(optimizer, lr):
        for param_group in optimizer.param_groups:
              param_group['lr'] = lr
              #print('lr now is :', param_group['lr'])
        return optimizer

   if j < len(training_data_loader_s)-1:
    
    j = j +1
    s_img = Variable(s_img.cuda())
    #s_label = Variable(s_label.type(torch.LongTensor).view(-1).cuda())
    s_label = Variable(s_label.view(-1).cuda())


    t_img, _, t_name = t_dataset.next()
    t_img = Variable(t_img.cuda())
    #t_label = Variable(t_label.type(torch.LongTensor).view(-1).cuda())
    #t_label = Variable(t_label.view(-1).cuda())
    

    mid_s = netSEnPre(s_img)
    mid_z = netZEn(t_img)

    mid_code = torch.cat((mid_s, mid_z), 1).contiguous()
    mid_img = Variable(netDe(mid_code).view(-1, 3, 28, 28).data)
    mid_label = s_label

    
    """============init============="""
    s_mid_label = torch.cat((s_label, mid_label), 0)
    s_mid_img = torch.cat((s_img, mid_img), 0)

    
    img_G = torch.cat((s_img, mid_img, t_img), 0)
   

    domain_t_label = torch.LongTensor(mb_size/3).zero_() + 1
    domain_t_label.cuda()

    domain_s_label = torch.LongTensor(mb_size*2/3).zero_()
    domain_s_label.cuda()

   

    domain_label = Variable(torch.cat((domain_s_label, domain_t_label), 0)).cuda()



    

    """=========G======="""
    E = netE(img_G)

    
    E_s_mid = netE(s_mid_img)
    
    pred_label_s_mid = netG(E_s_mid)

   

    """=========loss of G======="""

   
    loss = nn.NLLLoss()
    G_loss_anchor = loss(m(pred_label_s_mid), s_mid_label)
    

    G_loss =  G_loss_anchor

  
    writer.add_scalar('data/G_loss', G_loss.data[0], i)

    

    G_loss.backward()
    G_solver.step()
    reset_grad()

    if epoch>lr_ch:
       lr_scheduler(G_solver, lr_val)


    """=======E======="""
    E_t_s = netE(img_G)
    
    domain_pred = netD_Pred(E_t_s)

    """=====loss for E====="""
    E_loss =  nn.functional.cross_entropy(domain_pred, domain_label)			
    writer.add_scalar('data/E_loss', E_loss.data[0], i)
    E_loss.backward()
    E_solver.step()
    reset_grad()
    if epoch>lr_ch:
       lr_scheduler(E_solver, lr_val)

 

  
    if (i) % 100 == 0:
      
        imgvis = img_G.view(-1, 3, 28, 28)
        img = utils.make_grid(imgvis.data, nrow = 12, padding = 2, normalize=True, scale_each=False, pad_value=1)   
        writer.add_image('img/img_G', img, global_step = i)
       
        

       




        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(E.data)
        
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X = (X_tsne - x_min) / (x_max - x_min)
        
        for index in range(X.shape[0]):
          if index < X.shape[0]/3:
           y = 1
          if (index > X.shape[0]/3) & (index < X.shape[0] * 2/3):
           y = 2
          if index > X.shape[0] * 2/3:
           y = 3
          plt.scatter(X[index, 0], X[index, 1], marker = 'o',
                 color=plt.cm.Set1(y / 10.),
                 label = y, s = 20)           
          
        plt.savefig('emgoditer1.png')
        p1 = imread('emgoditer1.png')
        writer.add_image('embedding/embedding', p1, i)
        plt.close()    


      

   
    

    if (step) % 10 == 0:
        print ('iter1:Epoch [%d/%d],Step [%d/%d], G_Loss: %.4f' %(epoch+1, num_epochs, step, len(training_data_loader_s)-1, G_loss.data[0] ))
        
  print ('Epoch [%d/%d], G_Loss: %.4f' %(epoch+1, num_epochs, G_loss.data[0]))
  
  




writer.close()       
