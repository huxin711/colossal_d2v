import os
from pathlib import Path
import torch.nn as nn
import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from data import build_beit_pretraining_dataset
from model_zoo.data2vec_vision.D2VModel_c import d2v_small_patch16_224
from tqdm import tqdm
from colossalai.nn.lr_scheduler import LinearWarmupLR, CosineAnnealingWarmupLR


def create_model(ema=False):
    model = d2v_small_patch16_224(init_method='jax').cuda()
    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def main():
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
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model=model,
                                                                      optimizer=optimizer,
                                                                      criterion=criterion,
                                                                      train_dataloader=dataloader_train,
                                                                      test_dataloader=None,
                                                                      lr_scheduler=lr_scheduler)
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=None,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    main()
