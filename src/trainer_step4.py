import os
import glob
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from models import Dignet
from sys import argv
import tool

loop_id = argv[1]
loop_id = int(loop_id)
data_dir = './tmp_data/'    
model_save_path='./saved_models'+str(int(loop_id))

#%%
use_pretrain = False
batch_size = 32
accumulate_grad = 1
epochs = 10000
gpu_num=1
num_worker=5
lr_rate = 1e-4
background_weight = 0.00001

#%%
if use_pretrain == True and int(loop_id)==0:
#    "start downloading pretrain .ckpt"
#    file_id = '1XeVu57ZAlGpUK9sZzi42Ex8swl-ea5rr'
    model_path = './pretrain.ckpt'
#elif int(loop_id)==1:    
#    model_path = './round0.ckpt'
#elif int(loop_id)==2:    
#    model_path = './round1.ckpt'
#elif int(loop_id)==3:    
#    model_path = './round2.ckpt'
#elif int(loop_id)==4:    
#    model_path = './round3.ckpt'
elif int(loop_id)==7:    
    file_id = '1VJ1uCrph1Xw9_FkU8G0pB86r14VUBcRV'
    model_path = './round7.ckpt'
    tool.download_file_from_google_drive(file_id, model_path)    
    
#%%
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = transforms.ToTensor()
    
    def _convert_raw_label(self, label):
        """Convert raw label id to ready-to-train id"""
        label[label == 128] = 1
        label[label == 0] = 2
        label[label == 255] = 0
        return label

    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(Image.open(image))
        label = np.load(label)
        label = self._convert_raw_label(label)
        label = label.transpose((2, 0, 1))
        return self.transform(image), torch.from_numpy(label).long()

class FCNDataset(Dataset):
    def __init__(self, data_dir='./fcn_sampledata', transform=None):
        """Custom dataset class
        
        Args
        :data_dir: Path to data directory
        :transform: Transform 
        
        Attributes
        :data_dir:
        :transform:
        :image_filenames:
        :label_filenames:
        """
        self.data_dir = data_dir
        self.transform = transform
        
        self.input1 = glob.glob(os.path.join(data_dir, 'input', '*'))
        self.label_filenames = []
        for path in self.input1:
             self.label_filenames.append((path[:-4] + '.npy').replace('input', 'label'))
    
    def _convert_raw_label(self, label):
        """Convert raw label id to ready-to-train id"""
        label[label == 128] = 1
        label[label == 0] = 2
        label[label == 255] = 0
        return label
    
    def __len__(self):
        return len(self.input1)
    
    def __getitem__(self, idx):
        input1 = self.input1[idx]
        label_filename = self.label_filenames[idx]
        
#        image = np.array(Image.open(input1))
#        label = np.load(label_filename)
#        label = np.expand_dims(label, axis=-1)  # (H, W, 1)
        if self.transform:
            image, label = self.transform((input1, label_filename))
            #image, label = self.transform((image, label))
            #label = self._convert_raw_label(label)
        
        # print(image.dtype, input2.dtype, label.dtype)
        return image, label


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '', batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (3, 224, 224)
        # self.num_classes = 3

    def setup(self, stage=None):
        transform = transforms.Compose([
            ToTensor(),
        ])
        
        data_dir = self.data_dir
        train_and_val = FCNDataset(data_dir=data_dir, transform=transform)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            num_train = max(1, int(len(train_and_val) * 0.8))
            num_val = len(train_and_val) - num_train
            
            self.train, self.val = random_split(train_and_val, [num_train, num_val])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_dir = os.path.join(self.data_dir, 'test')
            self.test = FCNDataset(data_dir=test_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)



"""## Train"""

class LitAutoEncoder(pl.LightningModule):

    def __init__(self,learning_rate):
        super().__init__()
        self.auto_encoder = Dignet(num_input_channels=3)
        # print(self.auto_encoder)
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x_hat = self.auto_encoder(x)
        
        y = y.squeeze(dim=1)
        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('train_loss', loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x_hat = self.auto_encoder(x)
        y = y.squeeze(dim=1)
        
#        print(x_hat.shape, y.shape)
        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('val_loss', loss)
        # --------------------------

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x_hat = self.auto_encoder(x)
        y = y.squeeze(dim=1)
        
        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
    
if __name__ == '__main__':
    "class weight"
    transform = transforms.Compose([
        ToTensor(),
    ])
    train_and_val = FCNDataset(data_dir=data_dir, transform=transform)
    dl = DataLoader(train_and_val, batch_size=batch_size, num_workers=0)
     
    good_cnt = 0
    bad_cnt = 0
    breakpoint = 0
    for x, y in dl:
        good_cnt += (y.numpy() == 1).sum()
        bad_cnt += (y.numpy() == 2).sum()
        breakpoint+=1
        print(breakpoint)
        #if breakpoint ==500:
            #break
    
    total = good_cnt + bad_cnt
    a = good_cnt/total*100
    b = bad_cnt/total*100
    nSamples = [a,b]
    normedWeights= [1 - (rt / sum(nSamples)) for rt in nSamples]
    weights = [background_weight, normedWeights[0], normedWeights[1]]
    class_weights = torch.FloatTensor(weights)
    
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
    print('Class weights:', class_weights)
    print('Class weights:',good_cnt,bad_cnt,total)
    
    
    "start training"
    # init model
    model = LitAutoEncoder(learning_rate=lr_rate)
    if use_pretrain == False and int(loop_id)==0:
        print('not use pretrain')
    else:
        print('load pretrain model true')
        model.load_from_checkpoint(model_path,learning_rate=lr_rate)
#    print('ae.learning_rate,model_path')
    # Initialize a trainer
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    #!rm -rf ./saved_models
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_save_path,
        filename='fcn-resnet-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(gpus=gpu_num, max_epochs=epochs, callbacks=[checkpoint_callback],accumulate_grad_batches=accumulate_grad,progress_bar_refresh_rate=1)
    
    # Train the model
    data_loader = MyDataModule(data_dir, batch_size=batch_size,num_workers=num_worker)
    trainer.fit(model, data_loader)
