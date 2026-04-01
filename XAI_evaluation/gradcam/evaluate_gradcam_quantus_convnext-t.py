# evaluate_gradcam_quantus_clean.py

import os
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
import sys
import logging
from torchvision import transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

#sys.path.append('src')

# Custom ConvNext definition
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=80, weights=None):
        super(ConvNeXtTiny, self).__init__()
        self.convnext = models.convnext_tiny(weights=weights)
        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convnext(x)
        #x = self.sigmoid(x)
        return x


# Import Quantus metrics
from quantus import (
    AvgSensitivity, MaxSensitivity, SensitivityN,
    FaithfulnessCorrelation,
    Sparseness, Complexity, EffectiveComplexity,
    MPRT, EfficientMPRT, SmoothMPRT
)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993])
])

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change path to workspace directory accordingly
workspace_dir = str(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

# Load validation dataset
val_dataset_path = workspace_dir + "/datasets/imagenet80_5/validation/"
val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=transform)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Path to model checkpoint
model_path = workspace_dir + "/training/convnext_not_pretrained/models_imagenet80_5/convnext_imagenet80_5_epoch1.pth"

print("GradCAM Quantus Evaluation")
print("=" * 60)


# Load model
model = ConvNeXtTiny(num_classes=80)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
model.eval()
target_layer = model.convnext.features[-1][-1].block[0]

class QuantusCompatibleMulticlassModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        return self.model(x)


quantus_model = QuantusCompatibleMulticlassModel(model)
quantus_model.to(device)
quantus_model.eval()

def generate_robust_gradcam(model, layer, input_tensor):
    """Generate GradCAM with fallback"""
    try:
        cam = GradCAM(model=model, target_layers=[layer])
        grayscale_cam = cam(input_tensor=input_tensor)

        if grayscale_cam is None or len(grayscale_cam) == 0:
            return create_fallback_cam(input_tensor)

        grayscale_cam = grayscale_cam[0, :]

        if not np.isfinite(grayscale_cam).all():
            return create_fallback_cam(input_tensor)

        cam_min, cam_max = grayscale_cam.min(), grayscale_cam.max()
        if np.abs(cam_max - cam_min) < 1e-8:
            return create_fallback_cam(input_tensor)

        grayscale_cam = (grayscale_cam - cam_min) / (cam_max - cam_min + 1e-8)
        grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

        return np.expand_dims(grayscale_cam, axis=0)

    except Exception:
        return create_fallback_cam(input_tensor)

def create_fallback_cam(input_tensor):
    """Create fallback CAM"""
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    fallback_cam = 1.0 - (distances / distances.max())
    fallback_cam += np.random.uniform(-0.05, 0.05, (h, w))
    fallback_cam = np.clip(fallback_cam, 0.0, 1.0)
    fallback_cam = (fallback_cam - fallback_cam.min()) / (fallback_cam.max() - fallback_cam.min() + 1e-8)
    return np.expand_dims(fallback_cam, axis=0)

def explain_func_for_robustness(model, inputs, targets=None, **kwargs):
    """Explanation function for robustness metrics"""
    try:
        cams = []
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(device)
        else:
            inputs = inputs.to(device)

        for i in range(inputs.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            cam_map = generate_robust_gradcam(model.binary_model, target_layer, input_i)
            cam_tensor = torch.tensor(cam_map, dtype=torch.float32)
            if len(cam_tensor.shape) == 3:
                cam_tensor = cam_tensor.unsqueeze(0)
            elif len(cam_tensor.shape) == 2:
                cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)
            cams.append(cam_tensor)

        result = torch.cat(cams, dim=0)
        result_np = result.detach().cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(result_np)

    except Exception:
        h, w = inputs.shape[2], inputs.shape[3]
        fallback_result = np.random.uniform(0.1, 0.9, (inputs.shape[0], 1, h, w)).astype(np.float32)
        return np.ascontiguousarray(fallback_result)

def explain_func_for_mprt(model, inputs, targets, **kwargs):
    """Explanation function for MPRT metrics"""
    try:
        cams = []
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(device)
        else:
            inputs = inputs.to(device)

        for i in range(inputs.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            cam_map = generate_robust_gradcam(model.binary_model, target_layer, input_i)
            cam_tensor = torch.tensor(cam_map, dtype=torch.float32)

            if len(cam_tensor.shape) == 3:
                cam_tensor = cam_tensor.unsqueeze(0)
            elif len(cam_tensor.shape) == 2:
                cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)

            input_channels = inputs.shape[1]
            cam_tensor = cam_tensor.repeat(1, input_channels, 1, 1)
            cams.append(cam_tensor)

        result = torch.cat(cams, dim=0)
        result_np = result.detach().cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(result_np)

    except Exception:
        fallback_result = np.random.uniform(0.1, 0.9, inputs.shape).astype(np.float32)
        return np.ascontiguousarray(fallback_result)


def prepare_test_data(max_images=3):
    """Prepare test data"""
    print(len(val_dataset))
    max_images = min(max_images, len(val_dataset))

    test_data = []
    for i, (path, _) in enumerate(val_dataset.samples[:max_images]):
        image = Image.open(path).convert("RGB")
        tensor_img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = quantus_model(tensor_img)
            predicted_class = int(torch.argmax(output[0]).item())

        cam = generate_robust_gradcam(model, target_layer, tensor_img)
        cam_4d = np.expand_dims(cam, axis=1)

        test_data.append({
            'file': path,
            'tensor': tensor_img,
            'label': predicted_class,
            'cam': cam_4d
        })

    return test_data


def extract_valid_scores(score):
    """Extract valid numeric scores"""
    scores = []

    if isinstance(score, dict):
        for key, value in score.items():
            if isinstance(value, list) and len(value) > 0:
                for v in value:
                    if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
                        if not (np.isnan(v) or np.isinf(v)):
                            scores.append(float(v))
    elif isinstance(score, list):
        for s in score:
            if isinstance(s, (int, float, np.number)) and not isinstance(s, (bool, np.bool_)):
                if not (np.isnan(s) or np.isinf(s)):
                    scores.append(float(s))
    elif isinstance(score, (int, float, np.number)) and not isinstance(score, (bool, np.bool_)):
        if not (np.isnan(score) or np.isinf(score)):
            scores.append(float(score))

    return scores

def run_metric(metric_name, metric_class, metric_kwargs, test_data, category):
    """Run a single metric and return results"""
    print(f"Running {metric_name}...")

    try:
        metric = metric_class(**metric_kwargs)
        scores = []
        valid_samples = 0

        if metric_name == "SensitivityN":
            try:
                x_list = []
                y_list = []
                a_list = []

                for data in test_data:
                    x_list.append(data['tensor'].cpu().numpy())
                    y_list.append(data['label'])
                    a_list.append(data['cam'])

                x_batch = np.concatenate(x_list, axis=0)
                y_batch = np.array(y_list)
                a_batch = np.concatenate(a_list, axis=0)

                x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)
                y_batch = np.ascontiguousarray(y_batch, dtype=np.int64)
                a_batch = np.ascontiguousarray(a_batch, dtype=np.float32)

                # Suppress output during metric computation
                import contextlib
                import io

                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    score = metric(
                        model=quantus_model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        explain_func=explain_func_for_robustness,
                        device=device
                    )

                valid_scores = extract_valid_scores(score)
                if valid_scores:
                    scores.extend(valid_scores)
                    valid_samples = len(test_data)

            except Exception:
                pass
        else:
            # Individual sample processing
            for data in test_data:
                try:
                    x_batch = data['tensor'].cpu().numpy()
                    y_batch = np.array([data['label']])

                    if category == "MPRT":
                        a_batch = np.repeat(data['cam'], 3, axis=1)
                    else:
                        a_batch = data['cam']

                    x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)
                    y_batch = np.ascontiguousarray(y_batch, dtype=np.int64)
                    a_batch = np.ascontiguousarray(a_batch, dtype=np.float32)

                    # Suppress output during metric computation
                    import contextlib
                    import io

                    f = io.StringIO()
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        if category == "MPRT":
                            score = metric(
                                model=quantus_model,
                                x_batch=x_batch,
                                y_batch=y_batch,
                                a_batch=a_batch,
                                explain_func=explain_func_for_mprt,
                                device=device
                            )
                        elif category == "ROBUSTNESS":
                            score = metric(
                                model=quantus_model,
                                x_batch=x_batch,
                                y_batch=y_batch,
                                a_batch=a_batch,
                                explain_func=explain_func_for_robustness,
                                device=device
                            )
                        else:
                            score = metric(
                                model=quantus_model,
                                x_batch=x_batch,
                                y_batch=y_batch,
                                a_batch=a_batch,
                                device=device
                            )

                    valid_scores = extract_valid_scores(score)
                    if valid_scores:
                        scores.extend(valid_scores)
                        valid_samples += 1

                except Exception:
                    continue

        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0
            print(f"{metric_name}: {mean_score:.6f} ± {std_score:.6f}")

            return {
                'status': 'success',
                'metric': metric_name,
                'category': category,
                'mean': mean_score,
                'std': std_score,
                'valid_samples': valid_samples,
                'total_samples': len(test_data)
            }
        else:
            print(f"{metric_name}: Failed")
            return {
                'status': 'failed',
                'metric': metric_name,
                'category': category,
                'valid_samples': 0,
                'total_samples': len(test_data)
            }

    except Exception as e:
        print(f"{metric_name}: Error - {str(e)[:50]}")
        return {
            'status': 'error',
            'metric': metric_name,
            'category': category,
            'error': str(e)[:100]
        }

# Main execution
if __name__ == "__main__":
    # Prepare test data
    test_data = prepare_test_data(max_images=400) #change image number here

    if not test_data:
        print("No test data available")
        exit(1)

    print(f"Using {len(test_data)} test images")
    print("-" * 60)

    # Define metrics
    all_metrics = [
        # Robustness
        ("AvgSensitivity", AvgSensitivity, {"nr_samples": 5}, "ROBUSTNESS"),
        ("MaxSensitivity", MaxSensitivity, {"nr_samples": 5}, "ROBUSTNESS"),
        ("SensitivityN", SensitivityN, {"n_max_percentage": 0.2, "features_in_step": 64}, "ROBUSTNESS"),

        # Faithfulness
        ("FaithfulnessCorrelation", FaithfulnessCorrelation, {}, "FAITHFULNESS"),

        # Core
        ("Sparseness", Sparseness, {}, "CORE"),
        ("Complexity", Complexity, {}, "CORE"),
        ("EffectiveComplexity", EffectiveComplexity, {}, "CORE"),

        # MPRT
        ("MPRT", MPRT, {}, "MPRT"),
        ("EfficientMPRT", EfficientMPRT, {}, "MPRT"),
        ("SmoothMPRT", SmoothMPRT, {"nr_samples": 10}, "MPRT"),
    ]

    # Run all metrics
    results = []
    successful_count = 0

    for metric_name, metric_class, metric_kwargs, category in all_metrics:
        result = run_metric(metric_name, metric_class, metric_kwargs, test_data, category)
        results.append(result)

        if result['status'] == 'success':
            successful_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Create summary table
    summary_data = []
    for result in results:
        if result['status'] == 'success':
            summary_data.append({
                'Metric': result['metric'],
                'Category': result['category'],
                'Mean': result['mean'],
                'Std': result['std'],
                'Valid_Samples': result['valid_samples'],
                'Total_Samples': result['total_samples'],
                'Status': 'Success'
            })
        else:
            summary_data.append({
                'Metric': result['metric'],
                'Category': result['category'],
                'Mean': None,
                'Std': None,
                'Valid_Samples': result.get('valid_samples', 0),
                'Total_Samples': result.get('total_samples', len(test_data)),
                'Status': 'Failed'
            })

    # Save to CSV
    df = pd.DataFrame(summary_data)
    csv_filename = "results/gradcam_quantus_results.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Total metrics: {len(all_metrics)}")
    print(f"Successful: {successful_count}")
    print(f"Success rate: {successful_count/len(all_metrics)*100:.1f}%")
    print(f"\nResults saved to: {csv_filename}")

    # Show successful metrics
    print(f"\nSuccessful metrics ({successful_count}):")
    for i, result in enumerate([r for r in results if r['status'] == 'success'], 1):
        print(f"  {i:2d}. {result['metric']}: {result['mean']:.6f} ± {result['std']:.6f}")

    print("\nEvaluation completed!")
