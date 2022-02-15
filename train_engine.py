import os
import colossalai
from colossalai.utils import get_dataloader
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from model_zoo.data2vec_vision.D2VModel_c import d2v_small_patch16_224
from utils.data import build_beit_pretraining_dataset
import torch
import torch.nn as nn
from torch.autograd import Variable

DATASET_PATH = str(os.environ['DATA'])  # The directory of your dataset



def create_model(ema=False):
    model = d2v_small_patch16_224(init_method='jax').cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def train_d2v():
    args = colossalai.get_default_parser().parse_args()
    device = torch.device("cuda")
    colossalai.launch_from_torch(config=args.config)
    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)
    
    
    model = create_model(ema=False)
    ema_model = create_model(ema=True)
    
    
    dataset_train = build_beit_pretraining_dataset()
    dataloader_train = get_dataloader(dataset=dataset_train,
                                      shuffle=True,
                                      batch_size=(gpc.config.BATCH_SIZE // gpc.data_parallel_size),
                                      num_workers=4,
                                      pin_memory=True)
    
    criterion = nn.MSELoss(reduction="mean")
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model=model,
                                                                      optimizer=optimizer,
                                                                      criterion=criterion,
                                                                      train_dataloader=dataloader_train,
                                                                      test_dataloader=None,
                                                                      lr_scheduler=lr_scheduler)
                                                                      
    
    
    for epoch in range(gpc.config.NUM_EPOCHS):
        engine.train()
        engine.zero_grad()
        for step, (batch, _) in enumerate(train_dataloader):
            samples, bool_masked_pos = batch
            samples = samples.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

            outputs = engine(samples, bool_masked_pos)
            
            labels = ema_model(samples, bool_masked_pos=None)
            labels = Variable(labels.detach().data, requires_grad=False)
                
            train_loss = engine.criterion(outputs, labels)
            
            #total_loss = torch.mean(train_loss)
            #total_loss = total_loss.mean()
            
            engine.backward(train_loss)
            
           
            engine.step()
            
        #lr_scheduler.step()


        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])


if __name__ == '__main__':
    train_d2v()
