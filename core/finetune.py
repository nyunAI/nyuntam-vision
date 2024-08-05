from trailmet.utils import accuracy, AverageMeter
import time
import torch
import logging
from datetime import datetime
import os


def initialize_args(model, kwargs):
    cr = kwargs.get("CRITERION", "CrossEntropyLoss")
    criterion = eval(f"torch.nn.{cr}()")
    opt = kwargs.get("OPTIMIZER", "Adam")
    lr = kwargs.get("LEARNING_RATE", 0.0001)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    epochs = kwargs.get("FINETUNE_EPOCHS", 10)
    save_model = kwargs.get("SAVE_MODEL", False)
    save_path = kwargs.get("SAVE_PATH", "")
    device = kwargs.get("DEVICE", "cuda:0")
    validation = kwargs.get("VALIDATE", True)
    validation_interval = kwargs.get("VALIDATION_INTERVAL", 1)
    qat_step = kwargs.get("QAT_STEP", False)
    compression_scheduler = kwargs.get("QAT_SCHEDULER", False)

    return (
        criterion,
        device,
        optimizer,
        epochs,
        validation,
        validation_interval,
        save_model,
        save_path,
        qat_step,
        compression_scheduler,
    )


def validate(val_loader, model, name, args):
    logger = logging.getLogger(name)
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    criterion, device, _, _, _, _, _, _, _, _ = initialize_args(model, args)
    # switch to evaluate mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images.to(device))
            if type(output) != torch.Tensor:
                if "logits" in dir(output):
                    output = output.logits
            loss = criterion(output, target.to(device))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.to("cpu"), target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info(
                " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                    top1=top1, top5=top5
                )
            )

        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg, top5.avg


def train(train_loader, val_loader, model, name, args):
    logger = logging.getLogger(name)
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    (
        criterion,
        device,
        optimizer,
        epochs,
        validation,
        validation_interval,
        save_model,
        save_model_path,
        qat_step,
        compression_scheduler,
    ) = initialize_args(model, args)
    # switch to evaluate mode
    model = model.to(device)
    model.train()
    best_acc = 0
    for epoch in range(epochs):
        if qat_step == True:
            compression_scheduler.scheduler.epoch_step()
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            if qat_step == True:
                compression_scheduler.scheduler.step()
            # compute output
            optimizer.zero_grad()
            output = model(images.to(device))
            if type(output) != torch.Tensor:
                if "logits" in dir(output):
                    output = output.logits
            loss = criterion(output, target.to(device))
            if qat_step == True:
                compression_loss = compression_scheduler.loss()
                loss += compression_loss
            loss.backward()
            optimizer.step()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.to("cpu"), target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info(
                " *  Epoch {epoch} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                    epoch=epoch, top1=top1, top5=top5
                )
            )
        if validation == True and (epoch % validation_interval == 0):
            val_acc1, val_acc5 = validate(val_loader, model, name, args)
            if val_acc1 > best_acc:
                best_acc = val_acc1
                if save_model:
                    torch.save(model, os.path.join(save_model_path, "best_model.pt"))
        logger.info(
            " * Final Acc@1 {top1.avg:.3f}  Final Acc@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

    return model, top1.avg, top5.avg
