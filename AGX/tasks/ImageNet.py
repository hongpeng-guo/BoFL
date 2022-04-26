##################################################################
# TinyImageNet classification task with ResNet50 model
##################################################################
 
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
 
DATASETS_ROOT = '/home/sl29/data/energyFLDatasets/tiny-imagenet-200/train'

class ImageNet_ResNet50:
 
   ''' Apply VisionTransformer on FashionMNIST dataset for classification training'''
 
   def __init__(self, trainset_size, batch_size):
 
      self.num_classes = 200

      self.transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.ToTensor()])
      self.trainset = datasets.ImageFolder(DATASETS_ROOT, transform=self.transform)
      self.trainset = torch.utils.data.Subset(self.trainset, list(range(trainset_size)))

      self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=1)

      self.model = models.resnet50(pretrained=False)
      self.model.fc = nn.Linear(2048, self.num_classes)

      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters())

 
if __name__ == "__main__":
 
   task_model = ImageNet_ResNet50(240, 8)

   device = device = torch.device("cuda")
   model = task_model.model.to(device)

   # warmup
   print('start warmup')
   for idx, (img, label) in enumerate(task_model.trainloader):
      img, label = img.to(device), label.to(device)
      predicted_label = model(img)
      loss = task_model.criterion(predicted_label, label)
      loss.backward()
      task_model.optimizer.step()

   import time, os, sys, inspect

   currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
   parentdir = os.path.dirname(currentdir)
   sys.path.insert(0, parentdir)
   import powerLog as pl
   from powerLog import PowerLogger

   logger = PowerLogger(interval=0.1, nodes=list(filter(lambda n: n[0].startswith('module/'), pl.getNodes())))

   print('start train')
   logger.start()
   for _ in range(3):
      for idx, (img, label) in enumerate(task_model.trainloader):
         img, label = img.to(device), label.to(device)
         print(img.size(), label.size())
         t0 = time.time()
         predicted_label = model(img)
         loss = task_model.criterion(predicted_label, label)
         loss.backward()
         task_model.optimizer.step()
         torch.cuda.synchronize()
         t1 = time.time()
         print(predicted_label.size(), t1 - t0)
      time.sleep(10)
   logger.stop()

   logger.showDataTraces()