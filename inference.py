#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MobileFormer 
@PRODUCT ：PyCharm
@Author  ：Dazis
@Date    ：4/20/25 3:22 AM 
'''
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Inference Script for MobileFormer Plant Disease Classification
This script performs inference using a trained MobileFormer model on:
1. A single image
2. All images in a directory

Usage:
    python inference.py --model_path /path/to/best_model.pth --image_path /path/to/image.jpg
    python inference.py --model_path /path/to/best_model.pth --dir_path /path/to/image_directory
'''

import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import MobileFormer components (ensure this matches your training code)
from mobile_former import MergeClassifier

# Define the supported model types
MODEL_TYPES = ['26m', '52m', '96m', '151m', '214m', '294m', '508m']


def parse_arguments():
    parser = argparse.ArgumentParser(description='MobileFormer Inference for Plant Disease Classification')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image for inference')
    parser.add_argument('--dir_path', type=str, default=None, help='Path to a directory of images for inference')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--model_type', type=str, default='508m', choices=MODEL_TYPES, help='MobileFormer model type')
    parser.add_argument('--visualize', action='store_true', help='Visualize inference results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for directory inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--class_map_path', type=str, default=None, help='Path to class mapping JSON file')
    return parser.parse_args()


class InferenceDataset(Dataset):
    """Dataset for inference on multiple images"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), img_path
            else:
                return torch.zeros((3, 224, 224)), img_path


def get_transform():
    """Get transform for inference (should match validation transform from training)"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_path, model_type, num_classes, device):
    """Load a trained MobileFormer model"""
    print(f"Loading model from: {model_path}")

    # Import the appropriate model architecture
    if model_type == '26m':
        from mobile_former import mobile_former_26m
        model = mobile_former_26m(pretrained=False)
        token_dim = 128
    elif model_type == '52m':
        from mobile_former import mobile_former_52m
        model = mobile_former_52m(pretrained=False)
        token_dim = 128
    elif model_type == '96m':
        from mobile_former import mobile_former_96m
        model = mobile_former_96m(pretrained=False)
        token_dim = 128
    elif model_type == '151m':
        from mobile_former import mobile_former_151m
        model = mobile_former_151m(pretrained=False)
        token_dim = 192
    elif model_type == '214m':
        from mobile_former import mobile_former_214m
        model = mobile_former_214m(pretrained=False)
        token_dim = 192
    elif model_type == '294m':
        from mobile_former import mobile_former_294m
        model = mobile_former_294m(pretrained=False)
        token_dim = 192
    elif model_type == '508m':
        from mobile_former import mobile_former_508m
        model = mobile_former_508m(pretrained=False)
        token_dim = 192
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Modify the classifier to match the number of classes
    model.classifier = MergeClassifier(
        inp=model.classifier.conv[1].in_channels,
        oup=1280,
        ch_exp=6,
        num_classes=num_classes,
        drop_rate=0.2,
        drop_branch=[0.0, 0.0],
        token_dim=token_dim,
        cls_token_num=1,
        last_act='relu'
    )

    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if it exists (from DataParallel/DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model


def load_class_mapping(class_map_path=None):
    """Load class mapping from a JSON file or return a default mapping"""
    if class_map_path and os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            class_mapping = json.load(f)
        print(f"Loaded class mapping from {class_map_path}")
        return class_mapping
    else:
        print("No class mapping file provided or file not found. Using index-based labels.")
        return None


def predict_single_image(model, image_path, transform, device, class_mapping=None):
    """Perform inference on a single image"""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            start_time = time.time()
            output = model(input_tensor)
            inference_time = time.time() - start_time

        # Get prediction
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[predicted_class].item()

        # Get class label if mapping is available
        if class_mapping:
            if isinstance(class_mapping, dict):
                if str(predicted_class) in class_mapping:
                    predicted_label = class_mapping[str(predicted_class)]
                else:
                    predicted_label = f"Class {predicted_class}"
            elif isinstance(class_mapping, list) and predicted_class < len(class_mapping):
                predicted_label = class_mapping[predicted_class]
            else:
                predicted_label = f"Class {predicted_class}"
        else:
            predicted_label = f"Class {predicted_class}"

        # Get top-3 predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
        top_predictions = []

        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            prob = top_probs[i].item()

            if class_mapping:
                if isinstance(class_mapping, dict):
                    if str(idx) in class_mapping:
                        label = class_mapping[str(idx)]
                    else:
                        label = f"Class {idx}"
                elif isinstance(class_mapping, list) and idx < len(class_mapping):
                    label = class_mapping[idx]
                else:
                    label = f"Class {idx}"
            else:
                label = f"Class {idx}"

            top_predictions.append({"class_id": idx, "label": label, "probability": prob})

        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "inference_time": inference_time
        }

        return result

    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e)
        }


def predict_directory(model, dir_path, transform, device, batch_size=16, num_workers=4, class_mapping=None):
    """Perform inference on all images in a directory"""
    # Get all image file paths
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                   if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(valid_extensions)]

    if not image_paths:
        print(f"No valid images found in directory: {dir_path}")
        return []

    print(f"Found {len(image_paths)} images for inference")

    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    results = []
    total_time = 0
    total_images = 0

    # Process batches
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, desc="Processing images"):
            batch_images = batch_images.to(device)
            start_time = time.time()
            outputs = model(batch_images)
            batch_time = time.time() - start_time

            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)

            # Process each image in the batch
            for i in range(len(batch_paths)):
                img_path = batch_paths[i]
                predicted_class = predicted_classes[i].item()
                confidence = probabilities[i][predicted_class].item()

                # Get class label if mapping is available
                if class_mapping:
                    if isinstance(class_mapping, dict):
                        if str(predicted_class) in class_mapping:
                            predicted_label = class_mapping[str(predicted_class)]
                        else:
                            predicted_label = f"Class {predicted_class}"
                    elif isinstance(class_mapping, list) and predicted_class < len(class_mapping):
                        predicted_label = class_mapping[predicted_class]
                    else:
                        predicted_label = f"Class {predicted_class}"
                else:
                    predicted_label = f"Class {predicted_class}"

                # Get top-3 predictions
                top_k = 3
                top_probs, top_indices = torch.topk(probabilities[i], min(top_k, len(probabilities[i])))
                top_predictions = []

                for j in range(len(top_indices)):
                    idx = top_indices[j].item()
                    prob = top_probs[j].item()

                    if class_mapping:
                        if isinstance(class_mapping, dict):
                            if str(idx) in class_mapping:
                                label = class_mapping[str(idx)]
                            else:
                                label = f"Class {idx}"
                        elif isinstance(class_mapping, list) and idx < len(class_mapping):
                            label = class_mapping[idx]
                        else:
                            label = f"Class {idx}"
                    else:
                        label = f"Class {idx}"

                    top_predictions.append({"class_id": idx, "label": label, "probability": prob})

                # Add individual image time (approximation from batch)
                img_time = batch_time / len(batch_paths)
                total_time += img_time
                total_images += 1

                result = {
                    "image_path": img_path,
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                    "inference_time": img_time
                }

                results.append(result)

    # Calculate average inference time
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"Average inference time per image: {avg_time:.4f} seconds")

    return results


def visualize_results(results, output_dir, class_mapping=None):
    """Visualize inference results"""
    os.makedirs(output_dir, exist_ok=True)

    # Visualization for single image
    if len(results) == 1:
        result = results[0]
        if "error" in result:
            print(f"Cannot visualize result due to error: {result['error']}")
            return

        image_path = result["image_path"]
        image = Image.open(image_path).convert('RGB')

        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')

        # Create prediction text
        pred_text = f"Prediction: {result['predicted_label']}\n"
        pred_text += f"Confidence: {result['confidence']:.2%}\n"
        pred_text += f"Inference Time: {result['inference_time']:.4f}s\n\n"

        pred_text += "Top Predictions:\n"
        for pred in result["top_predictions"]:
            pred_text += f"- {pred['label']}: {pred['probability']:.2%}\n"

        plt.title(pred_text, loc='left')
        plt.tight_layout()

        # Save visualization
        output_path = os.path.join(output_dir, 'single_result.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")

    # Visualization for multiple images
    else:
        # Create grid visualization for the first 20 images (or less if fewer)
        num_images = min(20, len(results))
        rows = (num_images + 3) // 4  # Ceiling division
        cols = min(4, num_images)

        plt.figure(figsize=(16, 4 * rows))

        for i in range(num_images):
            result = results[i]
            if "error" in result:
                continue

            image_path = result["image_path"]
            try:
                image = Image.open(image_path).convert('RGB')

                plt.subplot(rows, cols, i + 1)
                plt.imshow(image)
                plt.axis('off')

                # Create prediction text
                pred_text = f"{os.path.basename(image_path)}\n"
                pred_text += f"Prediction: {result['predicted_label']}\n"
                pred_text += f"Confidence: {result['confidence']:.2%}"

                plt.title(pred_text, fontsize=9)
            except Exception as e:
                print(f"Error visualizing {image_path}: {e}")

        plt.tight_layout()

        # Save visualization
        output_path = os.path.join(output_dir, 'batch_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Batch visualization saved to {output_path}")

        # Create a bar chart of class distribution
        class_counts = {}
        for result in results:
            if "error" in result:
                continue

            label = result["predicted_label"]
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_classes]
        counts = [item[1] for item in sorted_classes]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Predicted Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()

        # Add count labels above bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        # Save chart
        output_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Class distribution chart saved to {output_path}")


def save_results(results, output_dir):
    """Save inference results to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # For single image result
    if len(results) == 1:
        output_path = os.path.join(output_dir, 'single_result.json')
    else:
        output_path = os.path.join(output_dir, 'batch_results.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # For multiple images, also create a CSV summary
    if len(results) > 1:
        import csv
        csv_path = os.path.join(output_dir, 'results_summary.csv')

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'predicted_class', 'predicted_label', 'confidence', 'inference_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                if "error" in result:
                    row = {'image_path': result['image_path'], 'predicted_class': 'ERROR',
                           'predicted_label': result['error'], 'confidence': 0, 'inference_time': 0}
                else:
                    row = {k: result[k] for k in fieldnames}
                writer.writerow(row)

        print(f"Summary CSV saved to {csv_path}")


def main():
    args = parse_arguments()

    # Check if either image_path or dir_path is provided
    if not args.image_path and not args.dir_path:
        print("Error: Either --image_path or --dir_path must be provided")
        return

    # Detect device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping if provided
    class_mapping = load_class_mapping(args.class_map_path)

    # Determine number of classes based on class mapping
    if class_mapping:
        if isinstance(class_mapping, dict):
            num_classes = max(map(int, class_mapping.keys())) + 1
        elif isinstance(class_mapping, list):
            num_classes = len(class_mapping)
        else:
            num_classes = 10  # Default fallback
        print(f"Detected {num_classes} classes from class mapping")
    else:
        # Default number of classes if no mapping provided
        num_classes = 10
        print(f"Using default number of classes: {num_classes}")

    # Load model
    model = load_model(args.model_path, args.model_type, num_classes, device)

    # Get transform for preprocessing
    transform = get_transform()

    # Perform inference
    if args.image_path:
        print(f"Performing inference on single image: {args.image_path}")
        results = [predict_single_image(model, args.image_path, transform, device, class_mapping)]
    else:
        print(f"Performing inference on images in directory: {args.dir_path}")
        results = predict_directory(model, args.dir_path, transform, device,
                                    args.batch_size, args.num_workers, class_mapping)

    # Save results
    save_results(results, args.output_dir)

    # Visualize results if requested
    if args.visualize:
        visualize_results(results, args.output_dir, class_mapping)

    print("Inference completed successfully")


if __name__ == "__main__":
    main()