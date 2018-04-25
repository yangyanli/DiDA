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
#import dataloader3_eval
import functools
import os


opt_gpu = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu




learning_rate = 0.00001
num_epochs = 5000

mb_size = 64
input_nc = 1
output_nc = 2
heat_h = 28
heat_w = 28
threshold = 0.5

"""==========net SEn========="""
class NetSEn(nn.Module):
   
    def __init__(self, input_nc, output_nc):
        super(NetSEn, self).__init__() 
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
netSEn = NetSEn(input_nc, output_nc).eval()
netSEn.load_state_dict(torch.load('modelrgm/3197_net_netE2.pkl'))
netSEn.cuda()
print(netSEn)

netG = NetG(input_nc, output_nc).eval()
netG.load_state_dict(torch.load('modelrgm/3197_net_netG2.pkl')) 
netG.cuda()
print(netG)

netZEn = NetZEn(input_nc, output_nc).train()
netZEn.cuda()
print(netZEn)

netDe = NetDe(input_nc, output_nc).train()
netDe.cuda()
print(netDe)

Aclassifier = AClassifier(input_nc, output_nc).train()
Aclassifier.cuda()
print(Aclassifier)



Ad_params = list(Aclassifier.parameters())
Re_params = list(netZEn.parameters()) + list(netDe.parameters()) + list(Aclassifier.parameters())


nets = [netZEn, netDe, Aclassifier]


root_path = "traindata/"
train_set_source = dataloader.get_training_set(root_path)
training_data_loader_s = dataloader.DataLoader(dataset=train_set_source, num_workers=2, batch_size = mb_size/2, shuffle= True)


#eval_set_source = dataloader_eval.get_training_set(root_path)
#eval_data_loader_s = dataloader_eval.DataLoader(dataset = eval_set_source, num_workers = 2, batch_size = mb_size, shuffle = True)

train_set_target = dataloaderm.get_training_set(root_path)
training_data_loader_t = dataloaderm.DataLoader(dataset = train_set_target, num_workers = 2, batch_size = mb_size/2, shuffle = True)


def reset_grad():
    for net in nets:
        net.zero_grad()


Re_solver = optim.Adam(Re_params, lr = learning_rate)
Ad_solver = optim.SGD(Ad_params, lr = learning_rate, momentum = 0.9)

criterionD = nn.L1Loss()
criterionM = nn.MSELoss()

reset_grad()

m = nn.LogSoftmax()
writer = SummaryWriter()




def eval(epoch):
        netSEn.eval()
        classifier.eval()


        eval_loss = 0
        correct_eval = 0
       
	for step, (eval_img, eval_label, eval_view) in enumerate(eval_data_loader_s):
        

			eval_img = Variable(eval_img.cuda())
			eval_label = Variable(eval_label.cuda()).view(-1)


                        code_s = netSEn(eval_img)
                       
                        pred_label_eval = classifier(code_s)

                        loss = nn.NLLLoss()
    			C_loss = loss(m(pred_label_eval), eval_label)
                        
                        
                        eval_loss += C_loss

                        pred = pred_label_eval.data.max(1)[1]
                        correct_eval += pred.eq(eval_label.data).cpu().sum()

        eval_loss /= len(eval_data_loader_s)    
        Accuracy_eval = float(correct_eval)/float(len(eval_data_loader_s.dataset))             
        print('Epoch [%d/%d], eval_loss: %.4f, Accuracy_eval: %d/%d(%.4f)' %(epoch+1, num_epochs, eval_loss.data[0], correct_eval, len(eval_data_loader_s.dataset), (float(correct_eval)/float(len(eval_data_loader_s.dataset)))))         
        writer.add_scalar('data/eval_loss', eval_loss.data[0], epoch)
        writer.add_scalar('data/correct_eval', correct_eval, epoch)
        writer.add_scalar('data/Accuracy_eval', float(correct_eval)/float(len(eval_data_loader_s.dataset)), epoch)
        return Accuracy_eval


lr_ch = 10000
best_acc = 0
"""===============start epoch==============="""
for epoch in range(num_epochs):

  netZEn.train()
  netDe.train()
  Aclassifier.train()

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
              print('lr now is :', param_group['lr'])
        return optimizer

   if j < len(training_data_loader_s)-1:
    
    j = j +1
    s_img = Variable(s_img.cuda())
    
    s_label = Variable(s_label.view(-1).cuda())


    t_img, _, t_name = t_dataset.next()

    t_img = Variable(t_img.cuda())

    t_e = netSEn(t_img)
    
    t_label = torch.max(netG(t_e), 1)[1]

    img_G = torch.cat((t_img, s_img), 0)

    label_t_s = torch.cat((t_label, s_label), 0)

    if step%4 ==0:

    	"""=======S======="""
    	code_s = netSEn(img_G)

    	"""=======Z======="""
    	code_z = netZEn(img_G)



    	"""=========Re======="""
        code = torch.cat((code_s, code_z), 1)
    	Reimg = netDe(code).view(-1, 3, 28, 28)

    	"""=========lossRe======="""
        Re_loss = criterionD(Reimg, img_G)
        
    	writer.add_scalar('data/Re_loss', Re_loss.data[0], i)



        """=========AClassifier======="""
    
        pred_label = Aclassifier(code_z)

        """=========lossAc======="""

   
        loss = nn.NLLLoss()
        Ac_loss = loss(m(pred_label), label_t_s)
    


        """=========loss======="""
        loss_t = Re_loss - 7*Ac_loss
    
        loss_t.backward()
        Re_solver.step()
        reset_grad()
        if epoch>lr_ch:
               lr_scheduler(Re_solver, lr_val)






 



    else:
   	
    	"""=======Ac======="""
    	code_za = netZEn(img_G)
    
        pred_label = Aclassifier(code_za)

        """=========lossAc======="""

   
        loss = nn.NLLLoss()
        Ad_loss = loss(m(pred_label), label_t_s)

        writer.add_scalar('data/Ad_loss', Ad_loss.data[0], i)
    
        Ad_loss.backward()
        Ad_solver.step()
        reset_grad()
        if epoch>lr_ch:
               lr_scheduler(Ad_solver, lr_val)

    """=======combine======="""
    code_s_target, code_s_source = torch.split(code_s, mb_size/2, 0)
    code_z_target, code_z_source = torch.split(code_z, mb_size/2, 0)
    code_s1_tar, code_s2_tar = torch.split(code_s_target, mb_size/4, 0)
    code_snew_tar = torch.cat((code_s2_tar, code_s1_tar), 0)

    
    
   
    code_mid = torch.cat((code_s_source, code_z_target), 1).contiguous()
 	
    codenew = torch.cat((code_snew_tar, code_z_target), 1).contiguous()
    Comimg = netDe(codenew).view(-1, 3, 28, 28)
    Commid = netDe(code_mid).view(-1, 3, 28, 28)


    if (i) % 100 == 0:
      
        imgvis = img_G.view(-1, 3, 28, 28)
        img = utils.make_grid(imgvis.data, nrow = 8, padding = 2, normalize=True, scale_each=False, pad_value=1)   
        writer.add_image('img/img_G', img, global_step = i)
       
        
        cycle_imgvis = Reimg.view(-1, 3, 28, 28)    
        cycle_img = utils.make_grid(cycle_imgvis.data, nrow = 8, padding = 2, normalize=True, scale_each=False, pad_value=1)   
        writer.add_image('img/cycle_img', cycle_img, i)  


        combine_imgvis = Comimg.view(-1, 3, 28, 28)    
        cycle_img = utils.make_grid(combine_imgvis.data, nrow = 8, padding = 2, normalize=True, scale_each=False, pad_value=1)   
        writer.add_image('img/combine_img', cycle_img, i) 

        combine_imgvis = Commid.view(-1, 3, 28, 28)    
        cycle_img = utils.make_grid(combine_imgvis.data, nrow = 8, padding = 2, normalize=True, scale_each=False, pad_value=1)   
        writer.add_image('img/mid_img', cycle_img, i) 



 

        #writer.add_text('Text/input', str(label_t_s.data), i)    
        
       
        #writer.add_text('Text/output_t', str(pred_label_t.data), i)





        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(code_s.data)
        #plot_embedding(X_tsne, "t-SNE embedding(%d)" %(step)
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X = (X_tsne - x_min) / (x_max - x_min)
        #plt.figure()   
        for index in range(X.shape[0]):
          if index < X.shape[0]/2:
           y = 1
          else:
           y = 2
          plt.scatter(X[index, 0], X[index, 1], marker = 'o',
                 color=plt.cm.Set1(y / 10.),
                 label = y, s = 20)           
          #p1.xticks([]), p1.yticks([])
        plt.savefig('ems1.png')
        p1 = imread('ems1.png')
        writer.add_image('embedding/embeddings', p1, i)
        plt.close()    


        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(code_z.data)
        #plot_embedding(X_tsne, "t-SNE embedding(%d)" %(step)
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X = (X_tsne - x_min) / (x_max - x_min)
        #plt.figure()   
        for index in range(X.shape[0]):
          if index < X.shape[0]/2:
           y = 1
          else:
           y = 2
          plt.scatter(X[index, 0], X[index, 1], marker = 'o',
                 color=plt.cm.Set1(y / 10.),
                 label = y, s = 20)           
          #p1.xticks([]), p1.yticks([])
        plt.savefig('emz1.png')
        p1 = imread('emz1.png')
        writer.add_image('embedding/embeddingz', p1, i)    
        plt.close()


   
    

    if (step) % 10 == 0:
        print ('steptwo1:Epoch [%d/%d],Step [%d/%d], Re_Loss: %.4f' %(epoch+1, num_epochs, step, len(training_data_loader_s)-1, Re_loss.data[0]))
        
  print ('Epoch [%d/%d], Re_Loss: %.4f, Ad_Loss: %.4f' %(epoch+1, num_epochs, Re_loss.data[0], Ad_loss.data[0]))
  
  
  #acc = eval(epoch)
  if (epoch) % 15 == 0:
    #best_acc = acc
  
    netZEn.save_network(netZEn, 'netZEn1', epoch, 'modelsteptwo/')
    
    netDe.save_network(netDe, 'netDe1', epoch, 'modelsteptwo/')
    Aclassifier.save_network(Aclassifier, 'Aclassifier1', epoch, 'modelsteptwo/')
    
    




writer.close()       
