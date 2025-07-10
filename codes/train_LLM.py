import scipy.io as sio
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from data_preprocess import *
from models import *
from training_funcs import *

# Dataset and Model Initialization
subject_name = 'EC183'
fname_hg_lat = './cache_on_subjects/%s/Concat_SAE_supervised_all_features' % subject_name
fname_timepoint = './cache/timepoints_EC183'
fname_electrodes = './cache/electrodes.mat'

dir_savemodel = './cache_on_subjects/%s/savemodel_LSM/' % subject_name
os.makedirs(dir_savemodel, exist_ok=True)


# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 500# in training, this should be 500
LEARNING_RATE = 0.0001

# Initialize model
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqTransformer().to(device)
criterion = DynamicSequenceLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
length_weights = [0.01, 0.1, 1, 10, 100]
# Create dataloader


TrainSet = Dataset_LSM_resynthesis(
    fname_hg_lat=fname_hg_lat, 
    fname_timepoint=fname_timepoint, 
    subject_name=subject_name,
    fname_electrodes=fname_electrodes,
    Train=True,
    N=450
)
TestSet = Dataset_LSM_resynthesis(
    fname_hg_lat=fname_hg_lat, 
    fname_timepoint=fname_timepoint, 
    subject_name=subject_name,
    fname_electrodes=fname_electrodes,
    Train=False,
    N=50
)
dataloader = DataLoader(TrainSet, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#testloader = ... # test not mentioned here
#'''
print('Beginning training...')
for length_weight in length_weights:
    print(length_weight)
    criterion = DynamicSequenceLoss(length_weight) 
    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        # 每隔10个epoch保存一次模型
        if (epoch + 1) % 50 == 0 or epoch == EPOCHS - 1:
            save_path = os.path.join(dir_savemodel, f'ckpt_epoch{epoch+1}_length_weight={length_weight}loss={avg_loss:.4f}.pkl')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Model saved to {save_path}")
#'''
