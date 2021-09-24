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

loop_id = argv[1]
#loop_id = 0
loop_id = int(loop_id)
data_dir = './pre_data'+str(loop_id)
model_save_path='./pre_saved_models'+str(int(loop_id))


#%%
use_pretrain = False
batch_size = 128
accumulate_grad = 1
epochs = 20
gpu_num=1
num_worker=5
#%%

if use_pretrain == True and int(loop_id)==0:
    model_path = './saved_models/fcn-resnet-epoch=98-val_loss=0.52.ckpt'

elif int(loop_id)>0:
    model_path = glob.glob('./saved_models'+str(int(loop_id)-1)+'/*.ckpt')[0]
#
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

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
        # self.input2 = glob.glob(os.path.join(data_dir, 'sec_input', '*'))
        # self.input2 = [path.replace('input/', 'sec_input/').replace('.png', '.npy') for path in self.input1]
        # self.label_filenames = glob.glob(os.path.join(data_dir, 'label', '*'))
        self.input2 = []
        for path in self.input1:
            self.input2.append((path[:-4] + '.npy').replace('input', 'sec_input'))
        self.label_filenames = [path.replace('input', 'label') for path in self.input1]

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
        input2_file = self.input2[idx]
        label_filename = self.label_filenames[idx]

        image = np.array(Image.open(input1))
        input2 = np.load(input2_file).astype('float32')
        label = np.array(Image.open(label_filename))
        label = np.expand_dims(label, axis=-1)  # (H, W, 1)
        if self.transform:
            image, label = self.transform((image, label))
            label = self._convert_raw_label(label)

        input2 = torch.from_numpy(input2)
        # print(image.dtype, input2.dtype, label.dtype)
        return (image, input2), label


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

        data_dir = os.path.join(self.data_dir, 'train')
        train_and_val = FCNDataset(data_dir=data_dir, transform=transform)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            num_train = max(1, int(len(train_and_val) * 0.5))
            #print(num_train)
            num_val = len(train_and_val) - num_train
            #print(num_val)
            self.train, self.val = random_split(train_and_val, [num_train, num_val])
            print(len(self.train))

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
        (x1, x2), y = batch
        x_hat = self.auto_encoder(x1, x2)

        y = y.squeeze(dim=1)
        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('train_loss', loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        (x1, x2), y = batch
        x_hat = self.auto_encoder(x1, x2)
        y = y.squeeze(dim=1)

#        print(x_hat.shape, y.shape)
        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('val_loss', loss)
        # --------------------------

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        (x1, x2), y = batch
        x_hat = self.auto_encoder(x1, x2)
        y = y.squeeze(dim=1)

        loss = F.cross_entropy(x_hat, y, weight=class_weights)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == '__main__':
    "class weight"
    weights = [0.001, 0.8, 1.2]
    class_weights = torch.FloatTensor(weights)

    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
    print('Class weights:', class_weights)


    "start training"
    # init model
    model = LitAutoEncoder(learning_rate=1e-4)
    if use_pretrain == False and int(loop_id)==0:
        print('')
    else:
        model.load_from_checkpoint(model_path,learning_rate=1e-5)
#    print('ae.learning_rate,model_path')
#    print(ae.learning_rate,model_path)
    # Initialize a trainer
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    #!rm -rf ./saved_models
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_save_path,
        filename='fcn-resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(gpus=gpu_num, max_epochs=epochs, callbacks=[checkpoint_callback],accumulate_grad_batches=accumulate_grad,progress_bar_refresh_rate=1)
    # Train the model
    data_loader = MyDataModule(data_dir, batch_size=batch_size,num_workers=num_worker)
    trainer.fit(model, data_loader)
