
from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, lr_decay


def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True,pad=13):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        if pad>0: 
            rows=torch.sum(inputs_2d==0,dim=(-1,-2,-3,-4))<(2*pad+1)*16*2
            inputs_2d=inputs_2d[rows]
            targets_3d=targets_3d[rows]
        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)
        
        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        targets_3d=targets_3d[:,0,pad,:]
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        
        optimizer.zero_grad()


        outputs_3d = model_pos(inputs_2d)
        loss_3d_pos = criterion(outputs_3d, targets_3d) 
        
        
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


