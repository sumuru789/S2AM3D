import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class PointNetClassifier(nn.Module):

    def __init__(self, num_classes=2, num_points=10000, feature_dim=4):
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.input_transform = self._create_transform_net(4, 4)
        self.feature_transform = self._create_transform_net(64, 64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _create_transform_net(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, output_dim * output_dim, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        trans = self.input_transform(x.transpose(2, 1))
        trans = trans.view(batch_size, 4, 4)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.mlp1(x)
        trans = self.feature_transform(x)
        trans = trans.view(batch_size, 64, 64)
        x = torch.bmm(x.transpose(2, 1), trans)
        x = x.transpose(2, 1)
        x = self.mlp2(x)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


class PointCloudDataset(Dataset):

    def __init__(self, selections_file, data_dir=None, num_points=10000, augment=True):
        self.num_points = num_points
        self.augment = augment
        with open(selections_file, 'r') as f:
            self.selections = json.load(f)
        self.data = []
        self.labels = []
        for _, record in self.selections.items():
            if record['selection'] not in ['Y', 'N']:
                continue
            data_path = record['data_path']
            if data_dir:
                data_path = os.path.join(data_dir, os.path.basename(data_path))
            if os.path.exists(data_path):
                self.data.append(data_path)
                self.labels.append(1 if record['selection'] == 'Y' else 0)
        print(f'Loaded {len(self.data)} samples')
        print(f'Qualified: {sum(self.labels)}')
        print(f'Unqualified: {len(self.labels) - sum(self.labels)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.load(self.data[idx], allow_pickle=True).item()
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
        if self.augment:
            point_features = self._augment_point_cloud(point_features)
        point_features = self._normalize_point_cloud(point_features)
        return torch.FloatTensor(point_features), torch.LongTensor([self.labels[idx]])

    def _augment_point_cloud(self, point_features):
        xyz = point_features[:, :3]
        labels = point_features[:, 3:]
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        xyz = xyz @ rotation_matrix.T
        xyz *= np.random.uniform(0.8, 1.2)
        xyz += np.random.normal(0, 0.01, xyz.shape)
        return np.column_stack([xyz, labels])

    def _normalize_point_cloud(self, point_features):
        xyz = point_features[:, :3]
        labels = point_features[:, 3:]
        xyz = xyz - np.mean(xyz, axis=0)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        return np.column_stack([xyz, labels])


class StratifiedPointCloudDataset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def create_stratified_split(dataset, train_split=0.7, val_split=0.15, test_split=0.15, random_state=42):
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, 'Split ratios must sum to 1'
    all_labels = [dataset.labels[i] for i in range(len(dataset))]
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=test_split, stratify=all_labels, random_state=random_state
    )
    val_ratio = val_split / (train_split + val_split)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        stratify=[all_labels[i] for i in train_val_indices],
        random_state=random_state,
    )
    train_dataset = StratifiedPointCloudDataset(dataset, train_indices)
    val_dataset = StratifiedPointCloudDataset(dataset, val_indices)
    test_dataset = StratifiedPointCloudDataset(dataset, test_indices)
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    test_labels = [all_labels[i] for i in test_indices]
    print('Stratified split:')
    print(f'  Train: {len(train_dataset)} (qualified: {sum(train_labels)}, unqualified: {len(train_labels) - sum(train_labels)})')
    print(f'  Val:   {len(val_dataset)} (qualified: {sum(val_labels)}, unqualified: {len(val_labels) - sum(val_labels)})')
    print(f'  Test:  {len(test_dataset)} (qualified: {sum(test_labels)}, unqualified: {len(test_labels) - sum(test_labels)})')
    return train_dataset, val_dataset, test_dataset


def train_model(model, train_loader, val_loader, test_loader, num_epochs, device,
                learning_rate=0.001, save_path='pointnet_classifier.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    writer = SummaryWriter('runs/pointnet_training')
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for data, target in train_pbar:
            data, target = data.to(device), target.squeeze().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * train_correct / train_total:.2f}%',
            })

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct / val_total:.2f}%',
                })

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
            for data, target in test_pbar:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                test_pbar.set_postfix({'Acc': f'{100.0 * test_correct / test_total:.2f}%'})

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        test_acc = 100.0 * test_correct / test_total
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_predictions, average='binary'
        )
        writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Precision/Val', val_precision, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'val_acc': val_acc,
                'val_loss': val_loss / len(val_loader),
                'best_test_acc': best_test_acc,
            }, save_path)
            print(f'Saved best model (Test Acc: {test_acc:.2f}%)')
        scheduler.step()

    writer.close()
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='binary'
    )
    cm = confusion_matrix(all_targets, all_predictions)
    print('\nTest results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Train PointNet point-cloud classifier')
    parser.add_argument('--selections', type=str, default='point_cloud_selections.json',
                        help='Annotation JSON path')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Optional directory to override data_path basenames')
    parser.add_argument('--num-points', type=int, default=10000, help='Points per cloud')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val-split', type=float, default=0.15, help='Val split ratio')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--model-path', type=str, default='pointnet_classifier.pth',
                        help='Checkpoint save/load path')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only, no training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = PointCloudDataset(args.selections, args.data_dir, args.num_points)
    train_dataset, val_dataset, test_dataset = create_stratified_split(
        dataset, args.train_split, args.val_split, args.test_split, args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PointNetClassifier(num_classes=2, num_points=args.num_points, feature_dim=4).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    if args.evaluate:
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model: {args.model_path}')
            print(f"Best test acc: {checkpoint.get('best_test_acc', 'N/A')}%")
            evaluate_model(model, test_loader, device)
        else:
            print(f'Model not found: {args.model_path}')
    else:
        print('Start training...')
        train_model(
            model, train_loader, val_loader, test_loader,
            args.epochs, device, args.lr, args.model_path,
        )
        print('\nEvaluate best model...')
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best test acc: {checkpoint.get('best_test_acc', 'N/A')}%")
        evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()
