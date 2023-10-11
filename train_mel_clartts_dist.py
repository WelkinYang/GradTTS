import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist

from accelerate import Accelerator, DistributedDataParallelKwargs

from data_utils import TextMelCollate, TextMelAraLoader
# from dataloader import SingleDataset, SingleCollate
import models
import commons
import utils
from text.symbols import symbols
# import horovod.torch as hvd

# hvd.init()
# torch.cuda.set_device(hvd.local_rank())

global_step = 0


def main():
    # """Assume Single Node Multi GPUs Training Only"""
    # assert torch.cuda.is_available(), "CPU training is not allowed."

    # n_gpus = torch.cuda.device_count()
    hps = utils.get_hparams()
    # train_and_eval(n_gpus, hps)
    train_and_eval(hps)


def train_and_eval(hps):
    global global_step
    # if hvd.local_rank() == 0:
    #     logger = utils.get_logger(hps.model_dir)
    #     logger.info(hps)
    #     utils.check_git_hash(hps.model_dir)
    #     writer = SummaryWriter(log_dir=hps.model_dir)
    #     writer_eval = SummaryWriter(
    #         log_dir=os.path.join(hps.model_dir, "eval"))
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(
        log_dir=os.path.join(hps.model_dir, "eval")
    )
    torch.manual_seed(hps.train.seed)

    # train_dataset = TextMelAraHfLoader(hps.data.training_files, hps.data)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # train_dataset = SingleDataset(hps.data.training_files, hps.data)
    # train_collate_fn = SingleCollate(hps.data.setting)
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=hps.train.batch_size,
    #     collate_fn=train_collate_fn, drop_last=True, num_workers=2, shuffle=False)
    
    # val_dataset = SingleDataset(hps.data.validation_files, hps.data)
    # val_collate_fn = SingleCollate(hps.data.setting)
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=hps.train.batch_size,
    #     collate_fn=val_collate_fn, drop_last=True, num_workers=2, shuffle=False)
    
    collate_fn = TextMelCollate(1)
    
    train_dataset = TextMelAraLoader(hps.data.training_files, hps.data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size,
        collate_fn=collate_fn, drop_last=True, num_workers=2, shuffle=True, pin_memory=True)
    
    val_dataset = TextMelAraLoader(hps.data.validation_files, hps.data)
    val_dataloader = DataLoader(
        val_dataset, batch_size=hps.train.batch_size,
        collate_fn=collate_fn, drop_last=True, num_workers=2, shuffle=False, pin_memory=True)

    generator = models.DiffusionGenerator(
        n_vocab=len(symbols) + getattr(hps.data, "add_blank", False),
        enc_out_channels=hps.data.n_mel_channels,
        **hps.model).cuda()

    optimizer_g = commons.Adam(scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, lr=hps.train.learning_rate)
    t_optimizer = torch.optim.Adam(generator.parameters(
    ), lr=optimizer_g.get_lr(), betas=hps.train.betas, eps=hps.train.eps)
    # t_optimizer = hvd.DistributedOptimizer(
    #     t_optimizer, named_parameters=generator.named_parameters())
    # hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
    optimizer_g.set_optimizer(t_optimizer)

    epoch_str = 1
    global_step = 0
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(
            hps.model_dir, "G_*.pth"), generator, optimizer_g)
        epoch_str += 1
        optimizer_g.step_num = (epoch_str - 1) * len(train_dataloader)
        optimizer_g._update_learning_rate()
        global_step = (epoch_str - 1) * len(train_dataloader)
    except:
        if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
            _ = utils.load_checkpoint(os.path.join(
                hps.model_dir, "ddi_G.pth"), generator, optimizer_g)

    train_dataloader, val_dataloader, generator, optimizer_g = accelerator.prepare(
        train_dataloader, val_dataloader, generator, optimizer_g
    )
    
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # if hvd.local_rank() == 0:
        #     train(hvd.local_rank(), epoch, hps, generator,
        #           optimizer_g, train_loader, logger, writer)
        #     evaluate(hvd.local_rank(), epoch, hps, generator,
        #              optimizer_g, val_loader, logger, writer_eval)
        #     utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(
        #         hps.model_dir, "G_{}.pth".format(epoch)))
        # else:
        #     train(hvd.local_rank(), epoch, hps, generator,
        #           optimizer_g, train_loader, None, None)
        train(accelerator, epoch, 
              hps, generator, 
              optimizer_g, train_dataloader, 
              logger, writer)
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            evaluate(epoch, 
                 hps, generator, 
                 val_dataloader, logger, writer_eval)
            
            utils.save_checkpoint(
                generator,
                optimizer_g,
                hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "G_latest.pth")
            )


def train(accelerator, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
    # train_loader.sampler.set_epoch(epoch)
    global global_step

    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        # x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
        #     rank, non_blocking=True)
        # y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
        #     rank, non_blocking=True)
        # x, x_lengths = x.to("cuda"), x_lengths.to("cuda")
        # y, y_lengths = y.to("cuda"), y_lengths.to("cuda")

        # Train Generator
        optimizer_g.zero_grad()

        grad_loss, (z_m, z_logs, z_mask), (attn, logw, logw_) = generator(
            x, x_lengths, y, y_lengths, gen=False)
        # z_logs is not used because we use N(mu, I) as the X_t
        l_mle = commons.mle_loss(y, z_m, z_logs, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [grad_loss, l_mle, l_length]
        # loss_gs = [grad_loss, l_length]
        loss_g = sum(loss_gs)

        # if hps.train.fp16_run:
        #     with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        #         scaled_loss.backward()
        #     grad_norm = commons.clip_grad_value_(
        #         amp.master_params(optimizer_g._optim), 5)
        # else:
        #     loss_g.backward()
        #     grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        
        accelerator.backward(loss_g)
        # loss_g.backward()
        # grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        grad_norm = accelerator.clip_grad_norm_(generator.parameters(), 5.0)
        optimizer_g.step()

        if accelerator.is_main_process:
            if batch_idx % hps.train.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs] +
                            [global_step, optimizer_g.get_lr()])

                if batch_idx % (hps.train.log_interval * 1000) == 0:
                    (y_gen, *_), *_ = generator(x[:1], x_lengths[:1], gen=True)
                    scalar_dict = {
                        "loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                                "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                                "attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                                },
                        scalars=scalar_dict)

    global_step += 1

    if accelerator.is_main_process:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(epoch, hps, generator, val_loader, logger, writer_eval):
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
        for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
            
            x, x_lengths = x.to("cuda"), x_lengths.to("cuda")
            y, y_lengths = y.to("cuda"), y_lengths.to("cuda")

            grad_loss, (z_m, z_logs, z_mask), (attn, logw, logw_) = generator(
                x, x_lengths, y, y_lengths, gen=False)
            # z_logs is not used because we use N(mu, I) as the X_t
            l_mle = commons.mle_loss(y, z_m, torch.ones_like(z_m), z_mask)
            l_length = commons.duration_loss(logw, logw_, x_lengths)

            loss_gs = [grad_loss, l_mle, l_length]
            # loss_gs = [grad_loss, l_length]
            loss_g = sum(loss_gs)

            if batch_idx == 0:
                losses_tot = loss_gs
            else:
                losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

            if batch_idx % hps.train.log_interval == 0:
                logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(val_loader.dataset),
                    100. * batch_idx / len(val_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs])

    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i,
                        v in enumerate(losses_tot)})
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
    main()
