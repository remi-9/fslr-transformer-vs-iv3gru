"""
Sync occlusion status from train/val CSVs to labels.csv.

This script updates the 'occluded' column in labels.csv based on the updated
values in FSL105_train.csv and FSL105_val.csv.

Usage:
    python data/processed/sync_labels.py
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def normalize_file_path(file_path: str) -> str:
    """Normalize file path for matching.
    
    Removes subdirectory prefix (e.g., 'C0/') and .npz extension.
    
    Args:
        file_path: File path from CSV (e.g., 'C0/clip_00001_good_morning_S0.npz')
        
    Returns:
        Normalized filename (e.g., 'clip_00001_good_morning_S0')
    """
    # Remove .npz extension if present
    if file_path.endswith('.npz'):
        file_path = file_path[:-4]
    
    # Remove subdirectory prefix if present (e.g., 'C0/', 'C1/')
    parts = file_path.split('/')
    if len(parts) > 1:
        # Return basename (last part)
        return parts[-1]
    
    return file_path


def load_split_csv(csv_path: Path) -> Dict[str, int]:
    """Load train or val CSV and create lookup dictionary.
    
    Args:
        csv_path: Path to train or val CSV file
        
    Returns:
        Dictionary mapping normalized filename to occluded status
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path.name} not found")
        return {}
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')
    
    # Create dictionary: normalized filename -> occluded status
    lookup = {}
    for _, row in df.iterrows():
        file_path = str(row['file'])
        normalized = normalize_file_path(file_path)
        occluded = int(row['occluded']) if pd.notna(row['occluded']) else 0
        lookup[normalized] = occluded
    
    return lookup


def sync_labels(
    labels_csv_path: Path,
    train_csv_path: Path,
    val_csv_path: Path
) -> bool:
    """Sync occlusion status from train/val CSVs to labels.csv.
    
    Args:
        labels_csv_path: Path to labels.csv
        train_csv_path: Path to FSL105_train.csv
        val_csv_path: Path to FSL105_val.csv
        
    Returns:
        True if successful, False otherwise
    """
    # Load train and val CSVs to get updated occlusion statuses
    print("Loading train and val CSVs...")
    train_lookup = load_split_csv(train_csv_path)
    val_lookup = load_split_csv(val_csv_path)
    
    print(f"  Train entries: {len(train_lookup)}")
    print(f"  Val entries: {len(val_lookup)}")
    
    # Load labels.csv
    print(f"\nLoading {labels_csv_path.name}...")
    try:
        df_labels = pd.read_csv(labels_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_labels = pd.read_csv(labels_csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            df_labels = pd.read_csv(labels_csv_path, encoding='cp1252')
    
    print(f"  Total entries: {len(df_labels)}")
    
    # Statistics
    stats = {
        'total': len(df_labels),
        'matched_train': 0,
        'matched_val': 0,
        'matched_both': 0,
        'not_matched': 0,
        'changed': 0,
        'unchanged': 0,
        'old_occluded': 0,
        'old_not_occluded': 0,
        'new_occluded': 0,
        'new_not_occluded': 0
    }
    
    changes = []
    
    # Process each row in labels.csv
    new_occluded_values = []
    
    for idx, row in df_labels.iterrows():
        file_path = str(row['file'])
        old_occluded = int(row['occluded']) if pd.notna(row['occluded']) else 0
        
        # Normalize file path for matching
        normalized = normalize_file_path(file_path)
        
        # Look up in val first (higher priority), then train
        new_occluded = old_occluded  # Default: keep old value
        matched_in_val = normalized in val_lookup
        matched_in_train = normalized in train_lookup
        
        if matched_in_val:
            new_occluded = val_lookup[normalized]
            if matched_in_train:
                stats['matched_both'] += 1
                # If different, prefer val (higher priority)
                if val_lookup[normalized] != train_lookup[normalized]:
                    print(f"  Warning: {normalized} has different values in train ({train_lookup[normalized]}) and val ({val_lookup[normalized]}). Using val.")
            stats['matched_val'] += 1
        elif matched_in_train:
            new_occluded = train_lookup[normalized]
            stats['matched_train'] += 1
        else:
            stats['not_matched'] += 1
            if idx < 5:  # Show first 5 unmatched files as examples
                print(f"  Warning: {normalized} not found in train or val CSVs")
        
        new_occluded_values.append(new_occluded)
        
        # Track old and new counts
        stats['old_occluded'] += old_occluded
        stats['old_not_occluded'] += (1 - old_occluded)
        stats['new_occluded'] += new_occluded
        stats['new_not_occluded'] += (1 - new_occluded)
        
        # Track changes
        if old_occluded != new_occluded:
            stats['changed'] += 1
            changes.append({
                'file': file_path,
                'normalized': normalized,
                'old': old_occluded,
                'new': new_occluded,
                'source': 'val' if matched_in_val else 'train' if matched_in_train else 'none'
            })
        else:
            stats['unchanged'] += 1
    
    # Update DataFrame
    df_labels['occluded'] = new_occluded_values
    
    # Save updated labels.csv
    print(f"\nSaving updated {labels_csv_path.name}...")
    df_labels.to_csv(labels_csv_path, index=False)
    
    # Print statistics
    print_statistics(stats, changes)
    
    return True


def print_statistics(stats: Dict, changes: list):
    """Print synchronization statistics.
    
    Args:
        stats: Statistics dictionary
        changes: List of changed entries
    """
    print("\n" + "="*60)
    print("SYNCHRONIZATION STATISTICS")
    print("="*60)
    
    print(f"\nTotal entries in labels.csv: {stats['total']}")
    print(f"  Matched in train: {stats['matched_train']}")
    print(f"  Matched in val: {stats['matched_val']}")
    print(f"  Matched in both: {stats['matched_both']}")
    print(f"  Not matched: {stats['not_matched']}")
    
    print(f"\nOLD Counts:")
    print(f"  Occluded:     {stats['old_occluded']:5d}")
    print(f"  Not Occluded: {stats['old_not_occluded']:5d}")
    print(f"  Total:        {stats['old_occluded'] + stats['old_not_occluded']:5d}")
    
    print(f"\nNEW Counts:")
    print(f"  Occluded:     {stats['new_occluded']:5d}")
    print(f"  Not Occluded: {stats['new_not_occluded']:5d}")
    print(f"  Total:        {stats['new_occluded'] + stats['new_not_occluded']:5d}")
    
    print(f"\nOcclusion status updates:")
    print(f"  Changed: {stats['changed']}")
    print(f"  Unchanged: {stats['unchanged']}")
    
    if stats['changed'] > 0 and changes:
        print(f"\nChanged entries (first 20):")
        for change in changes[:20]:
            print(f"  {change['normalized']}: {change['old']} -> {change['new']} (from {change['source']})")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    
    if stats['not_matched'] > 0:
        print(f"\nWarning: {stats['not_matched']} entries were not found in train/val CSVs")
        print("  These entries kept their original occlusion status")
    
    print("="*60 + "\n")


def main():
    """Main function to sync labels.csv with train/val CSVs."""
    # Get script directory (data/processed)
    script_dir = Path(__file__).parent
    
    labels_csv_path = script_dir / 'labels.csv'
    train_csv_path = script_dir / 'FSL105_train.csv'
    val_csv_path = script_dir / 'FSL105_val.csv'
    
    # Validate files exist
    if not labels_csv_path.exists():
        print(f"Error: {labels_csv_path} not found")
        return
    
    if not train_csv_path.exists() and not val_csv_path.exists():
        print(f"Error: Neither {train_csv_path.name} nor {val_csv_path.name} found")
        return
    
    print("Syncing occlusion status from train/val CSVs to labels.csv")
    print(f"Working directory: {script_dir}\n")
    
    # Perform sync
    if sync_labels(labels_csv_path, train_csv_path, val_csv_path):
        print(f"✓ Successfully updated {labels_csv_path.name}")
    else:
        print(f"✗ Failed to update {labels_csv_path.name}")


if __name__ == '__main__':
    main()

