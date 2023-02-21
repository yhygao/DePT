import os
import logging
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet, ImageFolder
import wandb

from model.dept_dino import DePTDino, DataAugmentationDINO
from losses import DINOLoss, smoothed_cross_entropy
from classifier import ViT_DePT
from image_list import ImageList
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    ProgressMeter,
)
import pdb


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
    'snow', 'frost', 'fog', 'brightness', 
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


def get_source_optimizer(model, args):
    if args.distributed:
        model = model.module
    if args.model.tune_type == 'full':
        backbone_params, extra_params = model.get_full_params()
    elif args.model.tune_type == 'prompt':
        backbone_params, extra_params = model.get_prompt_params()
    elif args.model.tune_type == 'ln':
        backbone_params, extra_params = model.get_ln_params()

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    elif args.optim.name == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "betas": (0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "betas": (0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
            ]
        )

    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]

    return optimizer


def train_source_domain(args):
    logging.info(f"Start source training on {args.data.src_domain}...")

    if 'vit' in args.model.arch:
        src_model = ViT_DePT(args.model).to('cuda')
        momentum_model = ViT_DePT(args.model).to('cuda')
    else:
        raise NotImplementedError("We currently only support ViT backbone")

    model = DePTDino(
       src_model,
       momentum_model,
       m=args.model.m,
       nlayer_head=args.model.nlayer_head,
       dino_out_dim=args.model.out_dim,
       norm_last_layer=True,
       consistency_type=args.model.consistency_type,
       hierarchy=args.model.hierarchy,
    ).cuda()


    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
    logging.info(f"1 - Created source model")

    # transforms
    #train_transform = get_augmentation("plain")
    train_transform = DataAugmentationDINO(
        (0.6, 1.),
        (0.05, 0.4),
        0
    )

    val_transform = get_augmentation("test")

    # datasets
    if args.data.dataset == "imagenet-c":
        train_dataset = ImageNet(args.data.image_root, transform=train_transform)
        val_dataset = ImageNet(
            args.data.image_root, split="val", transform=val_transform
        )
    else:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.src_domain}_list.txt"
        )
        train_dataset = ImageList(
            args.data.image_root, label_file, transform=train_transform
        )
        val_dataset = ImageList(
            args.data.image_root, label_file, transform=val_transform
        )
        assert len(train_dataset) == len(val_dataset)

        # split the dataset with indices
        indices = np.random.permutation(len(train_dataset))
        num_train = int(len(train_dataset) * args.data.train_ratio)
        train_dataset = Subset(train_dataset, indices[:num_train])
        val_dataset = Subset(val_dataset, indices[num_train:])
    logging.info(
        f"Loaded {len(train_dataset)} samples for training "
        + f"and {len(val_dataset)} samples for validation",
    )

    # data loaders
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.data.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    logging.info(f"2 - Created data loaders")

    optimizer = get_source_optimizer(model, args)
    args.learn.full_progress = args.learn.epochs * len(train_loader)
    
    DINO_loss = DINOLoss(args.model.out_dim, teacher_temp=args.model.teacher_temp, student_temp=args.model.student_temp, layerwise_weight=True, prompt_div=args.learn.prompt_div, args=args).cuda()

    logging.info(f"3 - Created optimizer")

    logging.info(f"Start training...")
    best_acc = 0.0
    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, DINO_loss, optimizer, epoch, args)

        # evaluate
        accuracy = evaluate(val_loader, model, domain=args.data.src_domain, args=args)
        if accuracy > best_acc and is_master(args):
            best_acc = accuracy
            filename = f"best_{args.data.src_domain}_{args.seed}.pth.tar"
            save_path = os.path.join(args.log_dir, filename)
            save_checkpoint(model, optimizer, epoch, save_path=save_path)
        
        if (epoch+1) % args.learn.eval_freq == 0:
            # evaluate on target before any adaptation
            
            if args.data.dataset == "imagenet-c":
                acc_list = []
                for c, corrupt_domain in enumerate(CORRUPTIONS):
                    print(corrupt_domain)
                    test_dir = os.path.join(args.data.image_root, corrupt_domain, '5')
                    test_c_dataset = ImageFolder(test_dir, val_transform)
                    sampler = DistributedSampler(test_c_dataset) if args.distributed else None
                    tgt_loader = DataLoader(
                        test_c_dataset,
                        batch_size=args.data.batch_size,
                        sampler=sampler,
                        pin_memory=True,
                        num_workers=args.data.workers,
                    )

                    logging.info(f"Evaluate {args.data.src_domain} model on {corrupt_domain}")
                    acc = evaluate(
                        tgt_loader,
                        model,
                        domain=f"{args.data.src_domain}-{corrupt_domain}",
                        args=args,
                        st_type='student',
                        wandb_commit=False,
                    )
                    acc_list.append(acc)

                avg_acc = np.array(acc_list).mean()
                if use_wandb(args):
                    wandb.log({"c_avg": avg_acc}, commit=True)

            else:
                for t, tgt_domain in enumerate(args.data.target_domains):
                    if tgt_domain == args.data.src_domain:
                        continue
                    label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
                    tgt_dataset = ImageList(args.data.image_root, label_file, val_transform)
                    sampler = DistributedSampler(tgt_dataset) if args.distributed else None
                    tgt_loader = DataLoader(
                        tgt_dataset,
                        batch_size=args.data.batch_size,
                        sampler=sampler,
                        pin_memory=True,
                        num_workers=args.data.workers,
                    )

                    logging.info(f"Evaluate {args.data.src_domain} model on {tgt_domain}")
                    
                    evaluate(
                        tgt_loader,
                        model,
                        domain=f"{args.data.src_domain}-{tgt_domain}",
                        args=args,
                        st_type='student',
                        wandb_commit=(t == len(args.data.target_domains) - 1),
                    )


def train_epoch(train_loader, model, DINO_loss, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    dino_cls_loss_meter = AverageMeter("Dino_cls_loss", ":.4f")
    dino_prompt_loss_meter = AverageMeter("Dino_prompt_loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, loss_meter, dino_cls_loss_meter, dino_prompt_loss_meter, top1], prefix="Epoch: [{}]".format(epoch),
    )

    # make sure to switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        images, labels, _ = data
        
        images = [im.cuda(args.gpu, non_blocking=True) for im in images]
        labels = data[1].cuda(args.gpu, non_blocking=True)

        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        teacher_output_cls, student_output_cls, student_logits, teacher_output_prompt, student_output_prompt = model(images[1:])

        loss_ce = smoothed_cross_entropy(
            student_logits,
            #torch.cat([labels, labels], dim=0),
            labels,
            num_classes=args.model.num_classes,
            epsilon=args.learn.epsilon,
        )
        loss_dino_cls, loss_dino_prompt, loss_prompt_div = DINO_loss(student_output_cls, teacher_output_cls, student_output_prompt, teacher_output_prompt, epoch=epoch)
        
        loss = (
            args.learn.alpha * loss_ce +
            args.learn.beta * loss_dino_cls +
            args.learn.beta2 * loss_dino_prompt +
            args.learn.lam * loss_prompt_div
        )

        # train acc measure (on one GPU only)
        preds = student_logits.argmax(dim=1)
        #acc = (preds == torch.cat([labels, labels], dim=0)).float().mean().detach() * 100.0
        acc = (preds == labels).float().mean().detach() * 100.0
        
        loss_meter.update(loss_ce.item(), images[1].size(0))
        dino_cls_loss_meter.update(loss_dino_cls.item(), images[1].size(0))
        if 'prompt' in args.model.consistency_type:
            dino_prompt_loss_meter.update(loss_dino_prompt.item(), images[1].size(0))
        top1.update(acc.item(), images[1].size(0))

        if use_wandb(args):
            wandb.log({"Loss_dino_cls": loss_dino_cls.item()}, commit=False)
            if 'prompt' in args.model.consistency_type:
                wandb.log({"Loss_dino_prompt": loss_dino_prompt.item()}, commit=False)

            if args.learn.prompt_div:
                wandb.log({"Loss_prompt_div": loss_prompt_div.item()}, commit=False)
            wandb.log({"Loss_ce": loss_ce.item() * args.learn.alpha}, commit=(i != len(train_loader)))

        # perform one gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)


def evaluate(val_loader, model, domain, args, wandb_commit=True, st_type='student'):
    model.eval()

    logging.info(f"Evaluating...")
    gt_labels, all_preds = [], []
    with torch.no_grad():
        iterator = tqdm(val_loader) if is_master(args) else val_loader
        for data in iterator:
            images = data[0].cuda(args.gpu, non_blocking=True)
            labels = data[1]

            _, logits = model(images, st_type=st_type, cls_only=True)
            preds = logits.argmax(dim=1).cpu()

            gt_labels.append(labels)
            all_preds.append(preds)

    gt_labels = torch.cat(gt_labels)
    all_preds = torch.cat(all_preds)

    if args.distributed:
        gt_labels = concat_all_gather(gt_labels.cuda())
        all_preds = concat_all_gather(all_preds.cuda())

        ranks = len(val_loader.dataset) % dist.get_world_size()
        gt_labels = remove_wrap_arounds(gt_labels, ranks).cpu()
        all_preds = remove_wrap_arounds(all_preds, ranks).cpu()

    accuracy = (all_preds == gt_labels).float().mean() * 100.0
    wandb_dict = {f"{domain} Acc": accuracy}

    logging.info(f"Accuracy: {accuracy:.2f}")
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.numpy(), y_pred=all_preds.numpy()
        )
        wandb_dict[f"{domain} Avg"] = acc_per_class.mean()
        wandb_dict[f"{domain} Per-class"] = acc_per_class

    if use_wandb(args):
        wandb.log(wandb_dict, commit=wandb_commit)

    return accuracy



