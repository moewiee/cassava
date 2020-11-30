import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from cvcore.data import cutmix_data, mixup_data
from cvcore.utils import AverageMeter
from cvcore.solver import WarmupCyclicalLR, WarmupMultiStepLR
from cvcore.losses import kd_loss_function
import random


def train_loop(_print, cfg, model, model_swa, train_loader,
               criterion, optimizer, scheduler, epoch, scaler):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    for i, (image, target) in enumerate(tbar):
        image = image.cuda()
        target = target.cuda()

        # mixup/ cutmix
        if cfg.DATA.MIXUP.ENABLED:
            image = mixup_data(image, alpha=cfg.DATA.MIXUP.ALPHA)
        elif cfg.DATA.CUTMIX.ENABLED:
            image = cutmix_data(image, alpha=cfg.DATA.CUTMIX.ALPHA)
        if cfg.MODEL.SELF_DISTILL:
            output, output_aux1, output_aux2 = model(image)
        else:
            output = model(image)
        with autocast():
            if cfg.MODEL.SELF_DISTILL:
                loss = criterion(output, target) + criterion(output_aux1, target) + criterion(output_aux2, target)
                # output_smtemp = torch.softmax(output / 4, 1)
            else:
                loss = criterion(output, target)
            # gradient accumulation
            loss = loss / cfg.SOLVER.GD_STEPS
        scaler.scale(loss).backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.SOLVER.GD_STEPS == 0:
            if isinstance(scheduler, WarmupCyclicalLR):
                scheduler(optimizer, i, epoch)
            elif isinstance(scheduler, WarmupMultiStepLR):
                scheduler.step()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        # record loss
        losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

        if model_swa is not None:
            if i % cfg.SOLVER.SWA.FREQ == 0:
                moving_average(model_swa, model, cfg.SOLVER.SWA.DECAY)

    if model_swa is not None:
        with torch.no_grad():
            bn_update(train_loader, model_swa, False)

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


def moving_average(net1, net2, decay):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data = param1.data * decay + param2.data * (1-decay)

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, half):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _) in enumerate(tbar):
        if half: input = input.cuda(non_blocking=True).half()
        else: input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
