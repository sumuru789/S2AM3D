import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pointnet_classifier import PointNetClassifier


class InferenceDataset(Dataset):

    def __init__(self, data_paths, num_points=10000):
        self.data_paths = data_paths
        self.num_points = num_points

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx], allow_pickle=True).item()
        coords = data['coord'].astype(np.float32)
        point_labels = data['label'].flatten().astype(np.float32)
        if len(coords) > self.num_points:
            indices = np.random.choice(len(coords), self.num_points, replace=False)
            coords = coords[indices]
            point_labels = point_labels[indices]
        elif len(coords) < self.num_points:
            indices = np.random.choice(len(coords), self.num_points, replace=True)
            coords = coords[indices]
            point_labels = point_labels[indices]
        point_features = np.column_stack([coords, point_labels])
        point_features = self._normalize_point_cloud(point_features)
        return torch.FloatTensor(point_features), self.data_paths[idx]

    def _normalize_point_cloud(self, point_features):
        xyz = point_features[:, :3]
        labels = point_features[:, 3:]
        xyz = xyz - np.mean(xyz, axis=0)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        return np.column_stack([xyz, labels])


def load_model(model_path, device, num_points=10000):
    model = PointNetClassifier(num_classes=2, num_points=num_points, feature_dim=4).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model: {model_path}')
    print(f"Val acc: {checkpoint.get('val_acc', 'N/A')}%")
    print(f"Test acc: {checkpoint.get('test_acc', 'N/A')}%")
    print(f"Best test acc: {checkpoint.get('best_test_acc', 'N/A')}%")
    return model


def predict_single_pointcloud(model, data_path, device, num_points=10000):
    model.eval()
    data = np.load(data_path, allow_pickle=True).item()
    coords = data['coord'].astype(np.float32)
    point_labels = data['label'].flatten().astype(np.float32)
    if len(coords) > num_points:
        indices = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[indices]
        point_labels = point_labels[indices]
    elif len(coords) < num_points:
        indices = np.random.choice(len(coords), num_points, replace=True)
        coords = coords[indices]
        point_labels = point_labels[indices]
    point_features = np.column_stack([coords, point_labels])
    xyz = point_features[:, :3]
    labels = point_features[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
    point_features = np.column_stack([xyz, labels])
    point_features = torch.FloatTensor(point_features).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(point_features)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def batch_inference(model, data_paths, device, batch_size=32, num_points=10000):
    model.eval()
    dataset = InferenceDataset(data_paths, num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    results = []
    with torch.no_grad():
        for batch_data, batch_paths in tqdm(dataloader, desc='Inference'):
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            for pred_class, conf, prob, path in zip(
                predicted_classes.cpu().numpy(),
                confidences.cpu().numpy(),
                probabilities.cpu().numpy(),
                batch_paths,
            ):
                results.append({
                    'file_path': path,
                    'file_id': os.path.basename(path).replace('.npy', ''),
                    'predicted_class': int(pred_class),
                    'predicted_label': 'qualified' if pred_class == 1 else 'unqualified',
                    'confidence': float(conf),
                    'prob_qualified': float(prob[1]),
                    'prob_unqualified': float(prob[0]),
                })
    return results


def analyze_results(results, output_dir='inference_results'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    total_count = len(df)
    qualified_count = len(df[df['predicted_class'] == 1])
    unqualified_count = len(df[df['predicted_class'] == 0])

    print('\nInference stats:')
    print(f'Total: {total_count}')
    print(f'Qualified: {qualified_count} ({qualified_count / total_count * 100:.1f}%)')
    print(f'Unqualified: {unqualified_count} ({unqualified_count / total_count * 100:.1f}%)')
    print('\nConfidence stats:')
    print(f"Mean: {df['confidence'].mean():.4f}")
    print(f"Std: {df['confidence'].std():.4f}")
    print(f"Max: {df['confidence'].max():.4f}")
    print(f"Min: {df['confidence'].min():.4f}")

    high_conf = df[df['confidence'] >= 0.8]
    medium_conf = df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)]
    low_conf = df[df['confidence'] < 0.6]
    print('\nBy confidence:')
    print(f'High (>=0.8): {len(high_conf)}')
    print(f'Medium (0.6-0.8): {len(medium_conf)}')
    print(f'Low (<0.6): {len(low_conf)}')

    df.to_csv(os.path.join(output_dir, 'inference_results.csv'), index=False)
    high_conf.to_csv(os.path.join(output_dir, 'high_confidence_results.csv'), index=False)
    low_conf.to_csv(os.path.join(output_dir, 'low_confidence_results.csv'), index=False)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')

    plt.subplot(2, 2, 2)
    plt.pie(
        [unqualified_count, qualified_count],
        labels=['unqualified', 'qualified'],
        autopct='%1.1f%%',
        startangle=90,
    )
    plt.title('Prediction Distribution')

    plt.subplot(2, 2, 3)
    plt.hist(df[df['predicted_class'] == 1]['confidence'], bins=20, alpha=0.7, label='qualified', color='green')
    plt.hist(df[df['predicted_class'] == 0]['confidence'], bins=20, alpha=0.7, label='unqualified', color='red')
    plt.title('Confidence by Class')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(df['prob_qualified'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Qualified Probability')
    plt.xlabel('Probability')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nResults saved to: {output_dir}')
    return df


def main():
    parser = argparse.ArgumentParser(description='PointNet point-cloud inference')
    parser.add_argument('--model', type=str, required=True, help='Trained model path')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with .npy files')
    parser.add_argument('--output-dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--num-points', type=int, default=10000, help='Points per cloud')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--single-file', type=str, default=None, help='Optional single-file inference')
    parser.add_argument('--confidence-threshold', type=float, default=0.8, help='High-confidence threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = load_model(args.model, device, args.num_points)

    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f'File not found: {args.single_file}')
            return
        predicted_class, confidence, probabilities = predict_single_pointcloud(
            model, args.single_file, device, args.num_points
        )
        print('\nSingle-file result:')
        print(f'File: {args.single_file}')
        print(f"Prediction: {'qualified' if predicted_class == 1 else 'unqualified'}")
        print(f'Confidence: {confidence:.4f}')
        print(f'Prob qualified: {probabilities[1]:.4f}')
        print(f'Prob unqualified: {probabilities[0]:.4f}')
        return

    data_paths = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.npy'):
                data_paths.append(os.path.join(root, file))
    if not data_paths:
        print(f'No .npy files found in {args.data_dir}')
        return

    print(f'Found {len(data_paths)} point-cloud files')
    results = batch_inference(model, data_paths, device, args.batch_size, args.num_points)
    df = analyze_results(results, args.output_dir)
    high_conf_results = df[df['confidence'] >= args.confidence_threshold]
    print(f'\nHigh-confidence results (>= {args.confidence_threshold}):')
    print(f'Count: {len(high_conf_results)}')
    if len(high_conf_results) > 0:
        print(f"Qualified: {len(high_conf_results[high_conf_results['predicted_class'] == 1])}")
        print(f"Unqualified: {len(high_conf_results[high_conf_results['predicted_class'] == 0])}")


if __name__ == '__main__':
    main()
