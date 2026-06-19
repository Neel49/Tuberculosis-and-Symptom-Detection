import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler  # type: ignore[attr-defined]
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vit_o
from vision_transformer import DINOHead, CLS_head, SymptomHead

import CXR_dataset

import main_run

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

# -----------------------------------------------------------------------------
# GPU / distributed environment diagnostics & sane defaults (copied from Covid)
# -----------------------------------------------------------------------------
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
print("torch.cuda.current_device():", torch.cuda.current_device() if torch.cuda.is_available() else None)

# Force NCCL to use the main ethernet interface and avoid docker / Infiniband
if 'NCCL_SOCKET_IFNAME' not in os.environ:
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp67s0'
if 'NCCL_IB_DISABLE' not in os.environ:
    os.environ['NCCL_IB_DISABLE'] = '1'
if 'NCCL_P2P_DISABLE' not in os.environ:
    os.environ['NCCL_P2P_DISABLE'] = '1'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

# -----------------------------------------------------------------------------
# Main training routine (now supports single-GPU mode similarly to Covid script)
# -----------------------------------------------------------------------------
def train_dino(args):
    # ----------------------------- init mode ---------------------------------
    if hasattr(args, 'single_gpu') and args.single_gpu:
        # -------- single GPU (no torch.distributed) --------
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using single GPU (GPU 0) for training")

        # Mock distributed env so downstream utils work transparently
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'

        # Do NOT call utils.init_distributed_mode here
        utils.fix_random_seeds(args.seed)
        print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        cudnn.benchmark = True
    else:
        # -------- distributed multi-GPU (torchrun spawns multiple ranks) ------
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # default to two A6000s
            print("Using GPUs 0,1 for distributed training")

        utils.init_distributed_mode(args)
        utils.fix_random_seeds(args.seed)
        print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = CXR_dataset.CXR_Dataset(
        args.data_path,
        csv_file='datafile.csv',
        transforms=transform,
        mode='train',
        labeled=True
    )

    if hasattr(args, 'single_gpu') and args.single_gpu:
        # Single GPU: regular sampler, larger batch to match effective size
        data_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=16,   # match 8×2 in distributed mode
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Created single-GPU dataloader. Dataset size: {len(dataset)} images.")
    else:
        # Distributed: DistributedSampler for shuffling across GPUs
        sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=8,    # per-GPU
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Created distributed dataloader. Dataset size: {len(dataset)} images.")


    student = vit_o.vit_small(
        patch_size=8,
        drop_path_rate=0.1,  # stochastic depth
    )
    teacher = vit_o.vit_small(patch_size=args.patch_size)
    embed_dim = student.embed_dim

    state_dict_full = torch.load(args.pretrained_dir, map_location="cpu")['state_dict']
    # Remove old symptom head params (14 classes) to avoid shape mismatch
    state_dict = {k: v for k, v in state_dict_full.items() if 'symptom_head' not in k}
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # remove `dino.`
    state_dict = {k.replace("dino.", ""): v for k, v in state_dict.items()}
    msg_t = teacher.load_state_dict(state_dict, strict=False)
    print('(Teacher) Pretrained weights found at {} and loaded with msg: {}'.format('CheXpert', msg_t))
    msg_s = student.load_state_dict(state_dict, strict=False)
    print('(Student) Pretrained weights found at {} and loaded with msg: {}'.format('CheXpert', msg_s))

    inter_dim = 384

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        384,
        65536,
        False,
        norm_last_layer=True,
    ), CLS_head(inter_dim, 256, 3), SymptomHead(inter_dim, 256, CXR_dataset.NUM_SYMPTOMS), args)
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(384, 65536, False), CLS_head(inter_dim, 256, 3), SymptomHead(inter_dim, 256, CXR_dataset.NUM_SYMPTOMS), args
    )
    
    # def print_model_details(model):
    #     print("Model Layers and their dimensions:")
    #     for name, param in model.state_dict().items():
    #         if isinstance(param, torch.Tensor):
    #             print(f"{name}: {param.size()}")
    #         else:
    #             print(f"{name} is not a Tensor.")
    # print_model_details(student)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    if hasattr(args, 'single_gpu') and args.single_gpu:
        # ----- single GPU: no DDP wrapper needed -----
        teacher_without_ddp = teacher
        teacher_without_ddp.load_state_dict(student.state_dict())
    else:
        # ----- distributed: convert to SyncBN + DDP (student only) -----
        if utils.has_batchnorms(student):
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        student = nn.parallel.DistributedDataParallel(
            student,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        teacher_without_ddp = teacher  # keep as raw nn.Module
        teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    # ------- class-imbalance weighting for 3-way disease classification -------
    disease_counts = np.zeros(3)
    for meta in dataset.total_images.values():
        label = meta["disease"]
        if 0 <= label < 3:
            disease_counts[label] += 1
    # Avoid division by zero and convert to inverse-frequency weights
    disease_counts[disease_counts == 0] = 1
    class_weights = disease_counts.sum() / disease_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    # Compute per-label pos_weight for symptom loss to handle imbalance
    pos_counts = np.zeros(CXR_dataset.NUM_SYMPTOMS)
    neg_counts = np.zeros(CXR_dataset.NUM_SYMPTOMS)
    for meta in dataset.total_images.values():
        sym = meta["symptoms"]
        pos_counts += sym
        neg_counts += 1 - sym

    # Print symptom distribution
    print("\nSymptom distribution in training data:")
    if hasattr(CXR_dataset, 'SYMPTOMS_V1') and CXR_dataset.NUM_SYMPTOMS == len(CXR_dataset.SYMPTOMS_V1):
        symptoms = CXR_dataset.SYMPTOMS_V1
    else:
        symptoms = [f"Symptom_{i}" for i in range(CXR_dataset.NUM_SYMPTOMS)]
    
    for i, symptom in enumerate(symptoms):
        if i < len(pos_counts):
            print(f"  {symptom}: {int(pos_counts[i])} positive, {int(neg_counts[i])} negative (ratio: {neg_counts[i]/max(pos_counts[i], 1):.1f})")
    
    # Avoid division by zero
    pos_counts[pos_counts == 0] = 1
    neg_counts[neg_counts == 0] = 1

    # Ratio can explode when a label is extremely rare; clip to avoid unstable gradients
    ratio = neg_counts / pos_counts
    ratio = np.clip(ratio, 1.0, 50.0)  # Increased from 20.0 to 50.0 for rare symptoms
    
    # Special boost for rare symptoms (Pneumothorax and Consolidation)
    print(f"Symptom weights before boost: {ratio}")
    
    # Boost Pneumothorax (index 5) and Consolidation (index 6) if using SYMPTOMS_V1
    if hasattr(CXR_dataset, 'SYMPTOMS_V1') and CXR_dataset.NUM_SYMPTOMS == 7:
        # Pneumothorax boost
        ratio[5] = min(ratio[5] * 1.5, 75.0)  # 50% boost, cap at 75
        # Consolidation boost  
        ratio[6] = min(ratio[6] * 1.5, 75.0)  # 50% boost, cap at 75
        print(f"Symptom weights after boost: {ratio}")
    
    pos_weight_tensor = torch.tensor(ratio, dtype=torch.float32).cuda()
    symptom_loss = FocalBCE(gamma=2.0, pos_weight=pos_weight_tensor)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    # Learning-rate scaling: only scale when truly distributed
    if hasattr(args, 'single_gpu') and args.single_gpu:
        base_lr = args.lr
    else:
        base_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 16.

    lr_schedule = utils.cosine_scheduler(
        base_lr, args.min_lr, args.epochs,
        len(data_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting pretraining !")
    for epoch in range(start_epoch, args.epochs):
        if not (hasattr(args, 'single_gpu') and args.single_gpu):
            data_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, dino_loss, ce_loss, symptom_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, args.ssl_epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        if hasattr(args, 'single_gpu') and args.single_gpu:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
        else:
            save_dict = {
                'student': student.module.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, dino_loss, ce_loss, symptom_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, ssl_epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    count = 0
    for it, (images, labels, symptom_labels) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        labels = labels.long().cuda(non_blocking=True)
        symptom_labels = symptom_labels.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            student_cls, student_symptoms = student(images[1])   # logits of 5 symptoms
            
            loss = ce_loss(student_cls, labels.view(-1))
            symptoms_loss = symptom_loss(student_symptoms, symptom_labels)
   

            if count %400 ==0:
                print("Disease Labels in Code")
                print(labels)
                print("Symptoms Labels in Code")
                print(symptom_labels)
                print("Symptoms Labels in Code")
                print(symptom_labels)
            if count %100 ==0:
                print("Disease loss")
                print(loss)
                print("Symptom loss")
                print(symptoms_loss)
            

            # Scale symptom loss down to avoid dominating gradients
            loss = loss * 0.25 + symptoms_loss * 0.75
            if count %100 ==0:
                print("total loss")
                print(loss)

            count = count + 1
            # print("loss")
            # print(loss)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

     

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
        world_sz = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        batch_center = batch_center / (len(teacher_output) * world_sz)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5)]
        )

        # Standard ImageNet normalisation (CheXpert checkpoints expect this)
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # first global crop (original image)
        self.global_transfo1 = transforms.Compose([
            normalize,
        ])

        # second global crop (little augmentation)
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=global_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.3),
            utils.GaussianBlur(0.3),
            normalize,
        ])

        # transformation for the local small crops (multiple augmentations with cropping)
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=local_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.5),
            utils.GaussianBlur(0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction='none',
                                          pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).detach()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t).pow(self.gamma) * bce
        return focal.mean()


# -----------------------------------------------------------------------------
# Entry-point: auto-launch torchrun (for multi-GPU) or run directly (single-GPU)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = main_run.get_args_parser()

    args = parser.parse_args()

    # Are we already inside a torchrun distributed environment?
    if 'RANK' not in os.environ:
        if args.single_gpu:
            # ---------------- single-GPU: run directly -----------------
            print("Running single GPU training...")
            args.option_dir = './'
            os.makedirs(args.option_dir, exist_ok=True)
            with open(os.path.join(args.option_dir, args.name + '_argv.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(vars(args).items())
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            train_dino(args)
        else:
            # -------------- multi-GPU: spawn with torchrun --------------
            import subprocess
            import random
            import socket

            print("Launching distributed training with torchrun...")

            # Find a free port in 29500-29999
            def find_free_port():
                start_port, end_port = 29500, 29999
                for _ in range(100):
                    port = random.randint(start_port, end_port)
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('localhost', port))
                            return port
                    except OSError:
                        continue
                # Fallback
                return random.randint(start_port, end_port)

            free_port = find_free_port()
            print(f"Auto-selected free port: {free_port}")

            env = os.environ.copy()
            env.update({
                'NCCL_SOCKET_IFNAME': 'enp67s0',
                'NCCL_IB_DISABLE': '1',
                'NCCL_P2P_DISABLE': '1',
                'OMP_NUM_THREADS': '1',
                'MASTER_PORT': str(free_port),
            })

            cmd = [
                'torchrun',
                '--nproc_per_node=2',
                f'--master_port={free_port}',
                __file__,
            ] + sys.argv[1:]

            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed with exit code {e.returncode}")
                sys.exit(e.returncode)
            except KeyboardInterrupt:
                print("Training interrupted by user")
                sys.exit(1)
    else:
        # Already inside torchrun
        args.option_dir = './'
        os.makedirs(args.option_dir, exist_ok=True)
        with open(os.path.join(args.option_dir, args.name + '_argv.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(vars(args).items())
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        train_dino(args)
