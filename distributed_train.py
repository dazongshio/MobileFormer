#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Plant Disease Classification with MobileFormer
This script trains a MobileFormer model on a plant disease dataset.
Use
torchrun --nproc_per_node=3 distributed_train.py --distributed --model_type 508m --batch_size 192 --pretrained_path /home/niu/Program/MobileFormer/mobile-former-508m.pth.tar --finetune
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import argparse
from datetime import datetime

# Import the MobileFormer components
from mobile_former import MergeClassifier

# Ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom dataset for plant disease images
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return the first image as a fallback
            img_path, label = self.samples[0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label


# Define augmentation and normalization transformations
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


# Prepare data loaders
# Prepare data loaders
def prepare_data(data_dir, batch_size=32, val_split=0.2, num_workers=4, distributed=False, world_size=1, rank=0):
    transforms_dict = get_transforms()

    # Create full dataset
    full_dataset = PlantDiseaseDataset(
        root_dir=data_dir,
    )

    # Split into training and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Generate indices for consistent train/val split across processes
    indices = list(range(len(full_dataset)))
    # Use a fixed seed for the shuffle to ensure all processes have the same split
    generator = torch.Generator().manual_seed(42)
    torch.randperm(len(indices), generator=generator)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Create train and validation datasets with proper subsetting
    train_dataset = torch.utils.data.Subset(
        PlantDiseaseDataset(root_dir=data_dir, transform=transforms_dict['train']),
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        PlantDiseaseDataset(root_dir=data_dir, transform=transforms_dict['val']),
        val_indices
    )

    if distributed:
        # Create sampler for distributed training (only for training set)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        # Validation doesn't need to be shuffled, but we need to ensure proper distributed evaluation
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create data loaders with samplers
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # Use the distributed sampler
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,  # Use the distributed sampler for validation too
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # For non-distributed training
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # For accuracy calculation in distributed training
    if distributed:
        # Each process only sees a portion of the data, so we need to adjust the dataset sizes
        # The actual count will be aggregated across processes during training
        train_samples_per_process = len(train_loader.dataset)
        val_samples_per_process = len(val_loader.dataset)
        dataset_sizes = {'train': train_samples_per_process, 'val': val_samples_per_process}
    else:
        dataset_sizes = {'train': train_size, 'val': val_size}

    dataloaders = {'train': train_loader, 'val': val_loader}

    return dataloaders, dataset_sizes, full_dataset.classes


# Training function with fixed accuracy calculation for distributed training
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu',
                save_dir='./checkpoints', distributed=False, rank=0):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # Wrap model with DDP for distributed training
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = datetime.now()

    # Only print from rank 0 in distributed mode
    is_main_process = not distributed or rank == 0
    if is_main_process:
        print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(num_epochs):
        if is_main_process:
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

        # Reset DistributedSampler for each epoch
        if distributed and 'train' in dataloaders and hasattr(dataloaders['train'].sampler, 'set_epoch'):
            dataloaders['train'].sampler.set_epoch(epoch)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0  # Keep track of actual samples processed

            # Iterate over data
            if is_main_process:
                pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}")
            else:
                pbar = dataloaders[phase]

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)  # Get actual batch size

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics - use batch size for actual count
                batch_loss = loss.item() * batch_size
                batch_corrects = torch.sum(preds == labels.data).item()
                running_loss += batch_loss
                running_corrects += batch_corrects
                running_samples += batch_size  # Count actual processed samples

                # Update progress bar if main process
                if is_main_process and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': batch_loss / batch_size,
                        'acc': batch_corrects / batch_size
                    })

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # Synchronize metrics across devices if distributed
            if distributed:
                # Create tensors for metrics
                loss_tensor = torch.tensor([running_loss], device=device)
                corrects_tensor = torch.tensor([running_corrects], device=device)
                samples_tensor = torch.tensor([running_samples], device=device)

                # Sum across all processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(corrects_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

                # Get the global values
                running_loss = loss_tensor.item()
                running_corrects = corrects_tensor.item()
                running_samples = samples_tensor.item()

                # Now calculate global metrics using the summed values
                epoch_loss = running_loss / running_samples
                epoch_acc = running_corrects / running_samples
            else:
                # For non-distributed case, use dataset size as before
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            # Print metrics from main process only
            if is_main_process:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Record history
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)

                # Save model if validation accuracy improved
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    model_path = os.path.join(save_dir, f'best_mobile_former_model_acc_{best_acc:.4f}.pth')

                    # Save model state
                    if distributed:
                        # Save the module without DDP wrapper
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': best_acc,
                            'loss': epoch_loss,
                        }, model_path)
                    else:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': best_acc,
                            'loss': epoch_loss,
                        }, model_path)
                    print(f"Saved new best model with accuracy: {best_acc:.4f}")

            # Save intermediate checkpoint every 5 epochs (from main process only)
            if is_main_process and (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'mobile_former_epoch_{epoch + 1}.pth')
                if distributed:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1}")

        if is_main_process:
            print()

    # Final statistics from main process only
    if is_main_process:
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"Training completed in {training_time}")
        print(f'Best val Acc: {best_acc:.4f}')

    # Wait for all processes to complete
    if distributed:
        dist.barrier()

    return model, history


# Plot training history
def plot_training_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


# Initialize distributed training
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a MobileFormer model for plant disease classification')
    parser.add_argument('--data_dir', type=str, default='diseases', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--model_type', type=str, default='96m',
                        choices=['26m', '52m', '96m', '151m', '214m', '294m', '508m'],
                        help='MobileFormer model size (26m, 52m, 96m, 151m, 214m, 294m, or 508m)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model weights (.pth or .pth.tar file)')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune from pretrained weights')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=-1, help='Number of processes for distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    args = parser.parse_args()

    # Initialize distributed training if requested
    if args.distributed:
        init_distributed_mode(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directories
    if not args.distributed or args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # Determine device
    if args.distributed:
        device = torch.device(f'cuda:{args.gpu}')
        if args.rank == 0:
            print(f"Using distributed training on {args.world_size} GPUs")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Prepare data
    dataloaders, dataset_sizes, class_names = prepare_data(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        distributed=args.distributed,
        world_size=args.world_size if args.distributed else 1,
        rank=args.rank if args.distributed else 0
    )

    num_classes = len(class_names)
    if not args.distributed or args.rank == 0:
        print(f"Classes: {class_names}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {dataset_sizes['train']}")
        print(f"Validation samples: {dataset_sizes['val']}")

    # Load model based on selected size
    if not args.distributed or args.rank == 0:
        print(f"Initializing MobileFormer {args.model_type} model...")
    if args.model_type == '26m':
        from mobile_former import mobile_former_26m
        model = mobile_former_26m(pretrained=False)
        token_dim = 128
    elif args.model_type == '52m':
        from mobile_former import mobile_former_52m
        model = mobile_former_52m(pretrained=False)
        token_dim = 128
    elif args.model_type == '96m':
        from mobile_former import mobile_former_96m
        model = mobile_former_96m(pretrained=False)
        token_dim = 128
    elif args.model_type == '151m':
        from mobile_former import mobile_former_151m
        model = mobile_former_151m(pretrained=False)
        token_dim = 192
    elif args.model_type == '214m':
        from mobile_former import mobile_former_214m
        model = mobile_former_214m(pretrained=False)
        token_dim = 192
    elif args.model_type == '294m':
        from mobile_former import mobile_former_294m
        model = mobile_former_294m(pretrained=False)
        token_dim = 192
    elif args.model_type == '508m':
        from mobile_former import mobile_former_508m
        model = mobile_former_508m(pretrained=False)
        token_dim = 192
    else:
        print(f"Model type {args.model_type} not recognized, defaulting to 96m")
        from mobile_former import mobile_former_96m
        model = mobile_former_96m(pretrained=False)
        token_dim = 128

    # Load pretrained weights if specified
    if args.pretrained_path:
        if not args.distributed or args.rank == 0:
            print(f"Loading pretrained weights from: {args.pretrained_path}")
        try:
            # Try to load the pretrained weights
            checkpoint = torch.load(args.pretrained_path, map_location=device)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if it exists (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            # Filter out classifier weights that won't match our target classes
            filtered_state_dict = {k: v for k, v in new_state_dict.items()
                                   if not k.startswith('classifier')}

            # Load the filtered weights
            model.load_state_dict(filtered_state_dict, strict=False)
            if not args.distributed or args.rank == 0:
                print("Successfully loaded pretrained weights (except classifier)")

        except Exception as e:
            if not args.distributed or args.rank == 0:
                print(f"Error loading pretrained weights: {e}")
                print("Continuing with randomly initialized weights")

    # Modify the classifier for our task
    model.classifier = MergeClassifier(
        inp=model.classifier.conv[1].in_channels,  # Get input channels from original classifier
        oup=1280,
        ch_exp=6,
        num_classes=num_classes,
        drop_rate=0.2,
        drop_branch=[0.0, 0.0],
        token_dim=token_dim,
        cls_token_num=1,
        last_act='relu'
    )

    # Loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer with different learning rates if fine-tuning
    if args.finetune and args.pretrained_path:
        if not args.distributed or args.rank == 0:
            print("Setting up fine-tuning optimization strategy")
        # Feature extractor layers get a smaller learning rate
        feature_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if name.startswith('classifier'):
                classifier_params.append(param)
            else:
                feature_params.append(param)

        # Use different learning rates
        optimizer = optim.AdamW([
            {'params': feature_params, 'lr': args.learning_rate * 0.1},  # 10x smaller LR for pretrained parts
            {'params': classifier_params, 'lr': args.learning_rate}
        ], weight_decay=args.weight_decay)

        if not args.distributed or args.rank == 0:
            print(
                f"Using different learning rates: {args.learning_rate * 0.1} for features, {args.learning_rate} for classifier")
    else:
        # Standard optimizer with uniform learning rate
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Train the model
    if not args.distributed or args.rank == 0:
        print("Starting training...")
    model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
        distributed=args.distributed,
        rank=args.rank if args.distributed else 0
    )

    # Plot training history (only from rank 0 in distributed mode)
    if not args.distributed or args.rank == 0:
        plot_training_history(history)
        print("Training complete!")


if __name__ == "__main__":
    main()