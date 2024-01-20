# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from math import sqrt
import math
import sys
from typing import Iterable, Optional

from einops.einops import rearrange

import torch
from torchvision import transforms

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torch.serialization import save

import utils
from PIL import Image
import os
from modeling_finetune import Sobel_conv
from skimage.measure import compare_ssim
import lpips


def train_class_batch_norearange(model, samples, target, cap, criterion, occm_gt , sobel_op=None, occm_loss_w = 0.,):        
    outputs, occm_pred = model(samples,cap)
    # print("outputs.shape",outputs.shape)
    # print("targets.shape",target.shape)
    loss_dict = {}
    loss_l1= criterion(outputs, target)
    loss_total = loss_l1
    loss_dict['l1'] = loss_l1.item()

    if occm_loss_w != 0 and occm_pred is not None:
        weight_occm = occm_gt*100 + 1
        # print('occm_pred',occm_pred.squeeze(-1))
        fn_occm_loss = torch.nn.BCEWithLogitsLoss(weight=weight_occm)
        # print(occm_pred.size())
        # print(occm_gt.size())
        loss_occm =  fn_occm_loss(occm_pred.squeeze(-1),occm_gt)
        loss_total += occm_loss_w *loss_occm
        loss_dict['occm'] = loss_occm.item()
    else:
        loss_dict['occm'] = 0
    return loss_total, outputs,loss_dict

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,patch_size=16):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, cap, keys,occm_mats) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(keys)
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc 
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        occm_mats = occm_mats.to(device, non_blocking=True)

        color_data = utils.get_colorization_data(samples)

        img_l = color_data['A']
        img_ab = color_data['B']
        # print("img_l.shape",img_l.shape)
        
        sobel_op = Sobel_conv().to(device)
        if loss_scaler is None:
            img_l.half()
            loss, output, loss_dict = train_class_batch_norearange(
                model, img_l.repeat(1,3,1,1), img_ab, cap, criterion, occm_mats, sobel_op=sobel_op) #
        else:
            with torch.cuda.amp.autocast():
                loss, output, loss_dict = train_class_batch_norearange(
                    model, img_l.repeat(1,3,1,1), img_ab, cap, criterion,occm_mats,sobel_op=sobel_op)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss) # BP
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize() # Waits for all kernels in all streams on a CUDA device to complete.

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_l1=loss_dict['l1'])
        metric_logger.update(loss_edge=loss_dict['edge'])
        metric_logger.update(loss_occm=loss_dict['occm'])
        # metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        # metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, epoch=10000,patch_size=16,save_img_dir=None, istest = False):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # psnrs_raw = np.zeros(len(data_loader))
    # psnrs_real = []
    lpips_fn_vgg = lpips.LPIPS(net='vgg').to(device, non_blocking=True)

    for step,(samples, cap, keys, occm_mat) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = samples
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        color_data = utils.get_colorization_data(images)
        img_l = color_data['A'] # [-1,1]
        img_ab = color_data['B'] # [-1,1]

        # compute output
        with torch.cuda.amp.autocast():
            output, occm_pred = model(img_l.repeat(1,3,1,1),cap)
        img_ab_fake = output
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        fake_rgb_tensors = utils.lab2rgb(torch.cat((img_l, img_ab_fake), dim=1))
        real_rgb_tensors = utils.lab2rgb(torch.cat((img_l, img_ab), dim=1))

        fake_rgbs = utils.tensor2im(fake_rgb_tensors)
        real_rgbs = utils.tensor2im(real_rgb_tensors)
        
        assert save_img_dir != None, "save_img_dir == None"

        for i in range(len(fake_rgbs)):
            psnr=utils.calculate_psnr_np(fake_rgbs[i],real_rgbs[i])
            # psnrs_real.append(psnr) 
            ssim = compare_ssim(fake_rgbs[i],real_rgbs[i],multichannel=True)

            metric_logger.update(psnr=psnr)
            metric_logger.update(ssim=ssim)

            if epoch%10 == 0:
                output_path = os.path.join(save_img_dir,'image','epoch_%d'%epoch)
                if not os.path.exists(output_path):
                    try:     
                        os.makedirs(output_path)
                    except:
                        pass
                if istest:
                    output_path_fake = os.path.join(output_path,keys[i].split('.')[0]+ "_" + cap[i] + '.png')
                    print("output_path_fake",output_path_fake)
                    save_img_fake = Image.fromarray(fake_rgbs[i])
                    save_img_fake.save(output_path_fake)
                
                else:
                    output_path_fake = os.path.join(output_path,keys[i].replace('jpg','png'))
                    # print(output_path)
                    save_img_fake = Image.fromarray(fake_rgbs[i])
                    save_img_fake.save(output_path_fake)
                    
                
        
        fn_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        lpips_score = lpips_fn_vgg(fn_norm(fake_rgb_tensors),fn_norm(real_rgb_tensors)).mean()
        metric_logger.update(lpips=lpips_score)
    
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
