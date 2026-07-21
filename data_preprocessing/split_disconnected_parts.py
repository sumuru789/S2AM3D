import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import glob


def load_npy_file(npy_path):
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        if not isinstance(data, dict) or 'coord' not in data or 'label' not in data:
            return None
        return data
    except Exception as e:
        print(f'Failed to load {npy_path}: {e}')
        return None


def save_npy_file(data, output_path):
    try:
        np.save(output_path, data)
        return True
    except Exception as e:
        print(f'Failed to save {output_path}: {e}')
        return False


def analyze_label_clusters(points, labels, label_id, eps_factor=0.3, min_samples=5):
    label_mask = labels.flatten() == label_id
    label_points = points[label_mask]
    label_indices = np.where(label_mask)[0]
    if len(label_points) == 0:
        return {'cluster_count': 0, 'clusters': [], 'is_connected': True}
    if len(label_points) == 1:
        return {'cluster_count': 1, 'clusters': [label_indices.tolist()], 'is_connected': True}
    if len(label_points) < min_samples:
        return {'cluster_count': 1, 'clusters': [label_indices.tolist()], 'is_connected': True}
    min_bound = np.min(label_points, axis=0)
    max_bound = np.max(label_points, axis=0)
    diagonal = np.linalg.norm(max_bound - min_bound)
    eps = max(diagonal * eps_factor, 0.01)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(label_points)
    cluster_labels = clustering.labels_
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    if len(unique_clusters) <= 1:
        return {'cluster_count': 1, 'clusters': [label_indices.tolist()], 'is_connected': True}
    clusters = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_point_indices = label_indices[cluster_mask]
        clusters.append(cluster_point_indices.tolist())
    return {'cluster_count': len(clusters), 'clusters': clusters, 'is_connected': False}


def split_disconnected_parts(input_data, eps_factor=0.15, min_samples=3):
    coords = input_data['coord']
    labels = input_data['label']
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    label_analysis = {}
    total_clusters = 0
    for label_id in unique_labels:
        analysis = analyze_label_clusters(coords, labels, label_id, eps_factor, min_samples)
        label_analysis[int(label_id)] = analysis
        total_clusters += analysis['cluster_count']
    if total_clusters == len(unique_labels):
        return {
            'original_data': input_data,
            'split_data': None,
            'split_info': {
                'needs_split': False,
                'total_labels': len(unique_labels),
                'total_clusters': total_clusters,
                'label_analysis': label_analysis,
            },
        }
    new_coords = []
    new_labels = []
    new_label_mapping = {}
    new_label_id = 0
    for original_label_id in unique_labels:
        analysis = label_analysis[int(original_label_id)]
        if analysis['is_connected']:
            label_mask = labels.flatten() == original_label_id
            new_coords.extend(coords[label_mask])
            new_labels.extend([new_label_id] * int(np.sum(label_mask)))
            new_label_mapping[new_label_id] = int(original_label_id)
            new_label_id += 1
        else:
            for cluster_indices in analysis['clusters']:
                new_coords.extend(coords[cluster_indices])
                new_labels.extend([new_label_id] * len(cluster_indices))
                new_label_mapping[new_label_id] = int(original_label_id)
                new_label_id += 1
    split_data = {
        'coord': np.array(new_coords),
        'label': np.array(new_labels),
        'original_label_mapping': new_label_mapping,
    }
    return {
        'original_data': input_data,
        'split_data': split_data,
        'split_info': {
            'needs_split': True,
            'total_labels': len(unique_labels),
            'total_clusters': total_clusters,
            'new_total_labels': new_label_id,
            'label_analysis': label_analysis,
            'label_mapping': new_label_mapping,
        },
    }


def process_single_file(npy_path, output_dir, eps_factor=0.15, min_samples=3):
    file_id = os.path.basename(npy_path).replace('.npy', '')
    data = load_npy_file(npy_path)
    if data is None:
        return None
    result = split_disconnected_parts(data, eps_factor, min_samples)
    original_output_path = os.path.join(output_dir, f'{file_id}_original.npy')
    save_npy_file(result['original_data'], original_output_path)
    split_output_path = os.path.join(output_dir, f'{file_id}_split.npy')
    if not result['split_info']['needs_split']:
        save_npy_file(result['original_data'], split_output_path)
        print(f'{file_id}: no split needed, all parts are connected')
    else:
        save_npy_file(result['split_data'], split_output_path)
        print(
            f"{file_id}: split {result['split_info']['total_labels']} -> "
            f"{result['split_info']['new_total_labels']} parts"
        )
    info_output_path = os.path.join(output_dir, f'{file_id}_split_info.json')
    with open(info_output_path, 'w') as f:
        json.dump(result['split_info'], f, indent=2)
    return {
        'file_id': file_id,
        'original_path': original_output_path,
        'split_path': split_output_path,
        'info_path': info_output_path,
        'split_info': result['split_info'],
    }


def main():
    parser = argparse.ArgumentParser(description='Split disconnected parts with DBSCAN')
    parser.add_argument('--input', type=str, required=True, help='Input directory with .npy files')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--eps-factor', type=float, default=0.1, help='DBSCAN eps = bbox diagonal * factor')
    parser.add_argument('--min-samples', type=int, default=15, help='DBSCAN min_samples')
    parser.add_argument('--max-files', type=int, default=None, help='Max number of files to process')
    args = parser.parse_args()

    assert os.path.exists(args.input), f'{args.input} does not exist'
    os.makedirs(args.output, exist_ok=True)

    print('=' * 60)
    print('Split disconnected parts')
    print('=' * 60)
    print(f'Input: {args.input}')
    print(f'Output: {args.output}')
    print(f'DBSCAN: eps_factor={args.eps_factor}, min_samples={args.min_samples}')
    print('=' * 60)

    npy_files = glob.glob(os.path.join(args.input, '**', '*.npy'), recursive=True)
    if args.max_files:
        npy_files = npy_files[:args.max_files]
        print(f'Limited to {args.max_files} files')
    print(f'Found {len(npy_files)} .npy files')
    if len(npy_files) == 0:
        print('No .npy files found')
        return

    results = []
    processed_count = 0
    failed_count = 0
    split_count = 0
    for npy_file in tqdm(npy_files, desc='Processing'):
        try:
            result = process_single_file(npy_file, args.output, args.eps_factor, args.min_samples)
            if result is not None:
                results.append(result)
                processed_count += 1
                if result['split_info']['needs_split']:
                    split_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f'Failed to process {npy_file}: {e}')
            failed_count += 1

    report = {
        'summary': {
            'total_files': len(npy_files),
            'processed_files': processed_count,
            'failed_files': failed_count,
            'split_files': split_count,
            'no_split_files': processed_count - split_count,
            'parameters': {
                'eps_factor': args.eps_factor,
                'min_samples': args.min_samples,
            },
        },
        'results': results,
    }
    report_path = os.path.join(args.output, 'split_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 60)
    print('Done')
    print('=' * 60)
    print(f'Total: {len(npy_files)}')
    print(f'Processed: {processed_count}')
    print(f'Failed: {failed_count}')
    print(f'Split: {split_count}')
    print(f'No split: {processed_count - split_count}')
    print(f'Report: {report_path}')
    print(f'Output: {args.output}')
    print('=' * 60)


if __name__ == '__main__':
    main()
