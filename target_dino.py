from copy import deepcopy
import logging
import os
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import wandb

from classifier import ViT_DePT
from image_list import ImageList, ImageList_Percent
from model.dept_dino import DePTDino, DataAugmentationDINO
from losses import DINOLoss, classification_loss, div
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
    clip_gradients,
)

import pdb

@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, args, mode='Val', st_type='student'):
    wandb_dict = dict()

    # make sure to switch to eval mode
    model.eval()

    # run inference
    logits, gt_labels, indices = [], [], []
    features = []
    banks = None
    torch.cuda.empty_cache()
    logging.info(f"Eval and labeling... {st_type}")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for imgs, labels, idxs in iterator:
    
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        feats, logits_cls = model(imgs, cls_only=True, st_type=st_type)
        
        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)
        torch.cuda.empty_cache()
   
    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")
    
    if args.distributed:
        # gather results from all ranks
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        # remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        features = remove_wrap_arounds(features, ranks)
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict[f"{mode} {st_type} Acc"] = accuracy
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict[f"{mode} {st_type} Avg"] = acc_per_class.mean()
        for idx in range(len(acc_per_class)):
            wandb_dict[f"{mode} {st_type} Class {idx}"] = acc_per_class[idx]

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
        features, probs, banks, args=args, gt_labels=gt_labels
    )
    wandb_dict[f"{mode} {st_type} Post Acc"] = acc
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict[f"{mode} {st_type} Post Avg"] = acc_per_class.mean()
        for idx in range(len(acc_per_class)):
            wandb_dict[f"{mode} {st_type} Post Class {idx}"] = acc_per_class[idx]

    pseudo_item_list = []
    for pred_label, idx in zip(pred_labels, indices):
        img_path, _ = dataloader.dataset.item_list[idx]
        pseudo_item_list.append((img_path, int(pred_label)))
    logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

    if use_wandb(args):
        wandb.log(wandb_dict)
    torch.cuda.empty_cache()

    return pseudo_item_list, banks


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):

        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


@torch.no_grad()
def update_labels(banks, idxs, features, logits, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features.contiguous())
        logits = concat_all_gather(logits)

    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])


@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]

        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy



def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    if args.model.tune_type == 'full':
        backbone_params, extra_params = model.get_full_params(fix_ss_head=False)
    elif args.model.tune_type == 'prompt':
        backbone_params, extra_params = model.get_prompt_params(fix_ss_head=False)
    elif args.model.tune_type == 'ln':
        backbone_params, extra_params = model.get_ln_params(fix_ss_head=False)
    else:
        raise NameError


    trainable_param_num = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    logging.info(f"{trainable_param_num} parameters need to be tuned.")
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
                    "lr": args.optim.lr,
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
                    "lr": args.optim.lr,# * 10,
                    "betas": (0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
            ]
        )

    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )
    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        if args.data.dataset == 'imagenet-c':
            target_dir = os.path.join(args.data.image_root, args.data.tgt_domain, '5')
            dummy_dataset = MyFolder(target_dir)
        else:
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_list.txt"
            )
            dummy_dataset = ImageList_Percent(args.data.image_root, label_file, percent=args.data.percent)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset
    

    if 'vit' in args.model.arch:
        src_model = ViT_DePT(args.model)
        momentum_model = ViT_DePT(args.model)
    else:
        raise NotImplementedError("We currently only support ResNet and ViT backbone.")

    model = DePTDino(
        src_model,
        momentum_model,
        m=args.model.m,
        nlayer_head=args.model.nlayer_head,
        dino_out_dim=args.model.out_dim,
        norm_last_layer=True,
        consistency_type=args.model.consistency_type,
        hierarchy=args.model.hierarchy
    )

    if args.model.src_log_dir is not None:
        checkpoint_path = os.path.join(
            args.model.src_log_dir,
        )

    model.load_from_checkpoint(checkpoint_path, same_student_teacher=True) # direct load source trained student and teacher
    #model.student.load_from_checkpoint(checkpoint_path)
    #model.teacher.load_from_checkpoint(checkpoint_path)

    '''
    # only load source trained student model for both student and teacher
    temp_model = copy.deepcopy(model.student)
    temp_head = copy.deepcopu(model.student_dinohead)
    model.teacher = temp_model
    model.teacher_dinohead = temp_head

    '''

    model.teacher.requires_grad_(False)
    model.teacher_dinohead.requires_grad_(False)

    model = model.cuda()



    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    logging.info(f"1 - Created target model")
    
    # define loss function (criterion) and optimizer
    ###############################################################################
    # add dino head to the optimizer
    optimizer = get_target_optimizer(model, args)
    logging.info("2 - Created optimizer")


    test_transform = get_augmentation("test")
    if args.data.dataset == 'imagenet-c':
        test_dir = os.path.join(args.data.image_root, args.data.tgt_domain, '5')
        test_dataset = MyFolder(test_dir, test_transform)

        val_dataset = MyFolder(test_dir, test_transform)
    else:
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
        test_dataset = ImageList(
            image_root=args.data.image_root,
            label_file=label_file,
            transform=test_transform,
        )

        val_transform = get_augmentation("test")
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
        val_dataset = ImageList_Percent(
            image_root=args.data.image_root,
            label_file=label_file,
            transform=val_transform,
            percent=args.data.percent,
            random_seed=args.data.random_seed,
        )
    test_sampler = (
        DistributedSampler(test_dataset, shuffle=False) if args.distributed else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, sampler=test_sampler, num_workers=2
    )

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, sampler=val_sampler, num_workers=2
    )
    
    if args.data.percent != 1.0:
        pseudo_item_list, banks = eval_and_label_dataset(
            val_loader, model, banks=None, args=args, mode='Val'
        )
    else:
        pseudo_item_list, banks = eval_and_label_dataset(
            val_loader, model, banks=None, args=args, mode='Test'
        )
    logging.info("3 - Computed initial pseudo labels")

    # Training data
    #train_transform = get_augmentation_versions(args)
    train_transform = DataAugmentationDINO(
        (0.2, 1.),
        (0.05, 0.4),
        0
    )
    
    if args.data.dataset == 'imagenet-c':
        train_dataset = MyFolder(test_dir, train_transform)
    else:
        train_dataset = ImageList(
                image_root=args.data.image_root,
                label_file=None,  # uses pseudo labels
                transform=train_transform,
                pseudo_item_list=pseudo_item_list,
        )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.data.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    DINO_loss = DINOLoss(args.model.out_dim, teacher_temp=args.model.teacher_temp, layerwise_weight=True, prompt_div=args.learn.prompt_div, args=args).cuda()


    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("4 - Created train/val loader")

    logging.info("Start training...")
    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, banks, DINO_loss, optimizer, epoch, args)
        
        if args.data.percent != 1.0:
            if (epoch+1) % args.learn.eval_freq == 0:
                eval_and_label_dataset(test_loader, model, None, args, mode='Test', st_type='student')
                eval_and_label_dataset(test_loader, model, None, args, mode='Test', st_type='teacher')

                _, banks = eval_and_label_dataset(val_loader, model, banks, args, mode='Val', st_type='student')
                _, banks = eval_and_label_dataset(val_loader, model, banks, args, mode='Val', st_type='teacher')

        else:
            if (epoch+1) % args.learn.eval_freq == 0:
                _, banks = eval_and_label_dataset(val_loader, model, banks, args, mode='Test', st_type='student')
                _, banks = eval_and_label_dataset(val_loader, model, banks, args, mode='Test', st_type='teacher')
       

    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")


def train_epoch(train_loader, model, banks, DINO_loss, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    dino_cls_loss_meter = AverageMeter("Dino_cls_loss", ":.4f")
    dino_prompt_loss_meter = AverageMeter("Dino_prompt_loss", ":.4f")
    top1_psd = AverageMeter("CLS-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, dino_cls_loss_meter, dino_prompt_loss_meter, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    # make sure to switch to train mode
    model.train()
    
    #m_schedule = [0.99]*3 + [args.model.m] * (args.learn.epochs - 3)
    #model.set_m(m_schedule[epoch])

    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images = [im.cuda(non_blocking=True) for im in images]

        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        
        feats_w, logits_w = model(images[0], cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args
            )
        teacher_output_cls, student_output_cls, student_logits, teacher_output_prompt, student_output_prompt = model(images[1:])
        

        #student_logits_1, student_logits_2 = student_logits.chunk(2)
          
        
        # classification
        loss_cls, accuracy_psd = classification_loss(
            student_logits, pseudo_labels_w, args
        )
        top1_psd.update(accuracy_psd.item(), len(logits_w))

        # DINO loss
        loss_dino_cls, loss_dino_prompt, loss_prompt_div = DINO_loss(student_output_cls, teacher_output_cls, student_output_prompt, teacher_output_prompt, epoch=epoch)
        if 'cls' in args.model.consistency_type:
            dino_cls_loss_meter.update(loss_dino_cls.item())
        if 'prompt' in args.model.consistency_type:
            dino_prompt_loss_meter.update(loss_dino_prompt.item())
        
        # diversification
        #loss_div = (div(student_logits_1) + div(student_logits_2)) / 2
        loss_div = div(student_logits)

        loss = (
            args.learn.alpha * loss_cls
            + args.learn.beta * loss_dino_cls
            + args.learn.beta2 * loss_dino_prompt
            + args.learn.eta * loss_div
            + args.learn.lam * loss_prompt_div
        )
        loss_meter.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        if args.learn.clip_grad:
            clip_gradients(model.student, args.learn.clip_grad)
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w, _ = model.teacher(images[0], return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "acc_psd": accuracy_psd.item(),
            }
            if 'cls' in args.model.consistency_type:
                wandb.log({"loss_dino_cls": loss_dino_cls.item()}, commit=False) 
            if 'prompt' in args.model.consistency_type:
                wandb.log({"loss_dino_prompt": loss_dino_prompt.item()}, commit=False)
            if args.learn.prompt_div:
                wandb.log({"Loss_prompt_div": loss_prompt_div.item()}, commit=False)
            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)


@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

