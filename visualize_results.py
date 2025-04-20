#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MobileFormer 
@PRODUCT ：PyCharm
@Author  ：Dazis
@Date    ：4/20/25 12:11 PM 
'''
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Visualization Script for MobileFormer Plant Disease Classification Results
This script visualizes training results from checkpoints and logs generated during training.
'''

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
colors = sns.color_palette("muted")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize training results for plant disease classification')
    parser.add_argument('--log_file', type=str, default=None, help='Path to the training log file')
    parser.add_argument('--history_file', type=str, default=None,
                        help='Path to the training history JSON file (if available)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                        help='Directory to save visualization results')
    parser.add_argument('--compare_models', action='store_true', help='Compare multiple model runs')
    parser.add_argument('--model_dirs', nargs='+', default=[], help='Directories containing model results to compare')
    parser.add_argument('--model_names', nargs='+', default=[], help='Names for the models being compared')
    return parser.parse_args()


def extract_metrics_from_log(log_file):
    """Extract training and validation metrics from a log file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None

    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }

    with open(log_file, 'r') as f:
        log_content = f.read()

    # Extract epochs
    epoch_matches = re.finditer(r'Epoch (\d+)/\d+', log_content)
    for match in epoch_matches:
        metrics['epochs'].append(int(match.group(1)))

    # Extract training metrics
    train_matches = re.finditer(r'Train Loss: ([\d\.]+) Acc: ([\d\.]+)', log_content)
    for match in train_matches:
        metrics['train_loss'].append(float(match.group(1)))
        metrics['train_acc'].append(float(match.group(2)))

    # Extract validation metrics
    val_matches = re.finditer(r'Val Loss: ([\d\.]+) Acc: ([\d\.]+)', log_content)
    for match in val_matches:
        metrics['val_loss'].append(float(match.group(1)))
        metrics['val_acc'].append(float(match.group(2)))

    # Handle potential mismatch in number of entries
    min_len = min(len(metrics['train_loss']), len(metrics['train_acc']),
                  len(metrics['val_loss']), len(metrics['val_acc']))

    metrics['train_loss'] = metrics['train_loss'][:min_len]
    metrics['train_acc'] = metrics['train_acc'][:min_len]
    metrics['val_loss'] = metrics['val_loss'][:min_len]
    metrics['val_acc'] = metrics['val_acc'][:min_len]
    metrics['epochs'] = metrics['epochs'][:min_len]

    return metrics


def load_history_from_json(history_file):
    """Load training history from a JSON file"""
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return None

    with open(history_file, 'r') as f:
        history = json.load(f)

    return history


def analyze_checkpoints(checkpoints_dir):
    """Analyze model checkpoints in the given directory"""
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return None

    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoints_dir}")
        return None

    # Extract checkpoint information
    checkpoint_info = []
    best_acc = 0
    best_model = None

    for filename in checkpoint_files:
        filepath = os.path.join(checkpoints_dir, filename)

        # Extract accuracy from filename if available
        acc_match = re.search(r'acc_([\d\.]+)', filename)
        epoch_match = re.search(r'epoch_(\d+)', filename)

        info = {
            'filename': filename,
            'filepath': filepath,
            'size': os.path.getsize(filepath) / (1024 * 1024),  # Size in MB
            'date': datetime.fromtimestamp(os.path.getmtime(filepath))
        }

        if acc_match:
            acc = float(acc_match.group(1))
            info['accuracy'] = acc
            if acc > best_acc:
                best_acc = acc
                best_model = filename

        if epoch_match:
            info['epoch'] = int(epoch_match.group(1))

        checkpoint_info.append(info)

    # Sort by modified date
    checkpoint_info.sort(key=lambda x: x['date'])

    return {
        'checkpoints': checkpoint_info,
        'best_model': best_model,
        'best_accuracy': best_acc
    }


def plot_training_curves(metrics, title_prefix="Training Results", output_dir="./"):
    """Plot training and validation curves for loss and accuracy"""
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot loss
    ax1.plot(metrics['epochs'], metrics['train_loss'], 'o-', color=colors[0], label='Training Loss')
    ax1.plot(metrics['epochs'], metrics['val_loss'], 'o-', color=colors[1], label='Validation Loss')
    ax1.set_title(f'{title_prefix} - Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Add min val loss marker
    min_val_loss = min(metrics['val_loss'])
    min_val_loss_epoch = metrics['epochs'][metrics['val_loss'].index(min_val_loss)]
    ax1.scatter(min_val_loss_epoch, min_val_loss, color='red', s=100, zorder=5)
    ax1.annotate(f'Min: {min_val_loss:.4f}',
                 (min_val_loss_epoch, min_val_loss),
                 xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Plot accuracy
    ax2.plot(metrics['epochs'], metrics['train_acc'], 'o-', color=colors[0], label='Training Accuracy')
    ax2.plot(metrics['epochs'], metrics['val_acc'], 'o-', color=colors[1], label='Validation Accuracy')
    ax2.set_title(f'{title_prefix} - Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Add max val accuracy marker
    max_val_acc = max(metrics['val_acc'])
    max_val_acc_epoch = metrics['epochs'][metrics['val_acc'].index(max_val_acc)]
    ax2.scatter(max_val_acc_epoch, max_val_acc, color='green', s=100, zorder=5)
    ax2.annotate(f'Max: {max_val_acc:.4f}',
                 (max_val_acc_epoch, max_val_acc),
                 xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))

    # Set integer ticks for epochs
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")

    return output_path


def plot_learning_rate(metrics, output_dir="./"):
    """Plot learning rate over epochs if available"""
    if 'learning_rate' not in metrics or not metrics['learning_rate']:
        return None

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['learning_rate'], 'o-', color=colors[2])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')  # Often useful for learning rate plots

    # Set integer ticks for epochs
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Learning rate plot saved to {output_path}")

    return output_path


def plot_checkpoint_progression(checkpoint_info, output_dir="./"):
    """Plot how model accuracy progresses across checkpoints"""
    if not checkpoint_info or 'checkpoints' not in checkpoint_info:
        return None

    checkpoints = checkpoint_info['checkpoints']

    # Extract accuracy and epoch data if available
    epochs = []
    accuracies = []

    for ckpt in checkpoints:
        if 'epoch' in ckpt and 'accuracy' in ckpt:
            epochs.append(ckpt['epoch'])
            accuracies.append(ckpt['accuracy'])

    if not epochs or not accuracies:
        return None

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, 'o-', color=colors[3])
    plt.title('Model Accuracy Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)

    # Set integer ticks for epochs
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = os.path.join(output_dir, 'checkpoint_progression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Checkpoint progression plot saved to {output_path}")

    return output_path


def compare_models(model_dirs, model_names, output_dir="./"):
    """Compare multiple model runs"""
    if len(model_dirs) != len(model_names):
        print("Error: Number of model directories must match number of model names")
        return None

    if len(model_dirs) < 2:
        print("Error: At least two models are required for comparison")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Collect metrics for each model
    all_metrics = []
    for model_dir, model_name in zip(model_dirs, model_names):
        # Try to load from history file first
        history_file = os.path.join(model_dir, 'training_history.json')
        if os.path.exists(history_file):
            metrics = load_history_from_json(history_file)
        else:
            # Try to load from log file
            log_file = None
            for file in os.listdir(model_dir):
                if file.endswith('.log'):
                    log_file = os.path.join(model_dir, file)
                    break

            if log_file:
                metrics = extract_metrics_from_log(log_file)
            else:
                print(f"No history or log file found for model: {model_name}")
                continue

        if metrics:
            metrics['name'] = model_name
            all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics could be loaded for any model")
        return None

    # Plot validation accuracy comparison
    plt.figure(figsize=(12, 8))

    for i, metrics in enumerate(all_metrics):
        if 'epochs' in metrics and 'val_acc' in metrics:
            plt.plot(metrics['epochs'], metrics['val_acc'], 'o-', label=metrics['name'], color=colors[i % len(colors)])

    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Set integer ticks for epochs
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = os.path.join(output_dir, 'model_comparison_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison (accuracy) saved to {output_path}")

    # Plot validation loss comparison
    plt.figure(figsize=(12, 8))

    for i, metrics in enumerate(all_metrics):
        if 'epochs' in metrics and 'val_loss' in metrics:
            plt.plot(metrics['epochs'], metrics['val_loss'], 'o-', label=metrics['name'], color=colors[i % len(colors)])

    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Set integer ticks for epochs
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = os.path.join(output_dir, 'model_comparison_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison (loss) saved to {output_path}")

    # Create a bar chart comparing best validation accuracies
    best_accuracies = []
    model_labels = []

    for metrics in all_metrics:
        if 'val_acc' in metrics:
            best_acc = max(metrics['val_acc'])
            best_accuracies.append(best_acc)
            model_labels.append(metrics['name'])

    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_labels, best_accuracies, color=colors[:len(model_labels)])

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, best_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

    plt.title('Best Validation Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(best_accuracies) + 0.05)  # Add some space for the text
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = os.path.join(output_dir, 'model_comparison_best_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Best accuracy comparison saved to {output_path}")

    return True


def create_summary_report(metrics, checkpoint_info, output_dir="./"):
    """Create a summary report with key metrics and findings"""
    if not metrics or not checkpoint_info:
        return None

    os.makedirs(output_dir, exist_ok=True)

    report = [
        "# MobileFormer Plant Disease Classification - Training Summary Report",
        f"## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Key Performance Metrics",
        f"- **Best Validation Accuracy**: {max(metrics['val_acc']):.4f} (Epoch {metrics['epochs'][metrics['val_acc'].index(max(metrics['val_acc']))]})",
        f"- **Lowest Validation Loss**: {min(metrics['val_loss']):.4f} (Epoch {metrics['epochs'][metrics['val_loss'].index(min(metrics['val_loss']))]})",
        f"- **Final Training Accuracy**: {metrics['train_acc'][-1]:.4f}",
        f"- **Final Validation Accuracy**: {metrics['val_acc'][-1]:.4f}",
        f"- **Final Training Loss**: {metrics['train_loss'][-1]:.4f}",
        f"- **Final Validation Loss**: {metrics['val_loss'][-1]:.4f}",
        f"- **Total Training Epochs**: {len(metrics['epochs'])}",
        "",
        "## Model Checkpoints",
        f"- **Best Model**: {checkpoint_info['best_model']} (Accuracy: {checkpoint_info['best_accuracy']:.4f})",
        f"- **Total Checkpoints Saved**: {len(checkpoint_info['checkpoints'])}",
        "",
        "## Training Observations",
        f"- The model {'improved consistently' if metrics['val_acc'][-1] >= metrics['val_acc'][0] else 'did not show consistent improvement'} throughout training.",
    ]

    # Check for overfitting
    train_acc_final = metrics['train_acc'][-1]
    val_acc_final = metrics['val_acc'][-1]
    acc_gap = train_acc_final - val_acc_final

    if acc_gap > 0.1:  # More than 10% gap between train and val accuracy
        report.append(
            f"- **Potential Overfitting Detected**: There is a significant gap between training accuracy ({train_acc_final:.4f}) and validation accuracy ({val_acc_final:.4f}).")
        report.append("  - Consider using stronger regularization techniques or early stopping.")

    # Check for underfitting
    if val_acc_final < 0.7:  # Less than 70% validation accuracy
        report.append(
            f"- **Potential Underfitting Detected**: The model achieved relatively low validation accuracy ({val_acc_final:.4f}).")
        report.append("  - Consider using a more complex model or training for more epochs.")

    # Check for convergence
    if len(metrics['val_acc']) > 5:
        recent_val_accs = metrics['val_acc'][-5:]
        val_acc_std = np.std(recent_val_accs)
        if val_acc_std < 0.005:  # Very small standard deviation in recent epochs
            report.append(
                f"- **Model Convergence**: The model appears to have converged in the final epochs (std: {val_acc_std:.6f}).")

    # Write report to file
    output_path = os.path.join(output_dir, 'training_summary_report.md')
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Summary report saved to {output_path}")

    return output_path


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_models:
        if not args.model_dirs or not args.model_names:
            print("Error: --model_dirs and --model_names must be provided when using --compare_models")
            return

        print("\n=== Comparing Multiple Models ===\n")
        compare_models(args.model_dirs, args.model_names, args.output_dir)
        return

    # Get metrics from history file or log file
    metrics = None
    if args.history_file:
        print(f"\n=== Loading Metrics from History File: {args.history_file} ===\n")
        metrics = load_history_from_json(args.history_file)
    elif args.log_file:
        print(f"\n=== Extracting Metrics from Log File: {args.log_file} ===\n")
        metrics = extract_metrics_from_log(args.log_file)
    else:
        # Try to find a history file in the checkpoints directory
        possible_history = os.path.join(args.checkpoints_dir, 'training_history.json')
        if os.path.exists(possible_history):
            print(f"\n=== Found and Loading History File: {possible_history} ===\n")
            metrics = load_history_from_json(possible_history)
        else:
            # Look for any log files
            log_files = [f for f in os.listdir(args.checkpoints_dir) if f.endswith('.log')]
            if log_files:
                log_file = os.path.join(args.checkpoints_dir, log_files[0])
                print(f"\n=== Found and Extracting Metrics from Log File: {log_file} ===\n")
                metrics = extract_metrics_from_log(log_file)
            else:
                print("No history or log files found. Please provide --history_file or --log_file.")

    if not metrics:
        print("Failed to load or extract metrics.")
        return

    # Analyze checkpoints
    print(f"\n=== Analyzing Checkpoints in: {args.checkpoints_dir} ===\n")
    checkpoint_info = analyze_checkpoints(args.checkpoints_dir)

    if not checkpoint_info:
        print("Failed to analyze checkpoints.")
        return

    # Print best model information
    print(f"Best model: {checkpoint_info['best_model']} with accuracy: {checkpoint_info['best_accuracy']:.4f}")

    # Calculate epochs if not present
    if 'epochs' not in metrics or not metrics['epochs']:
        metrics['epochs'] = list(range(1, len(metrics['train_loss']) + 1))

    # Plot training curves
    print("\n=== Generating Visualization Plots ===\n")
    plot_training_curves(metrics, title_prefix="MobileFormer Training", output_dir=args.output_dir)

    # Plot learning rate if available
    if 'learning_rate' in metrics and metrics['learning_rate']:
        plot_learning_rate(metrics, output_dir=args.output_dir)

    # Plot checkpoint progression
    plot_checkpoint_progression(checkpoint_info, output_dir=args.output_dir)

    # Create summary report
    print("\n=== Generating Summary Report ===\n")
    create_summary_report(metrics, checkpoint_info, output_dir=args.output_dir)

    print(f"\nAll visualization results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()