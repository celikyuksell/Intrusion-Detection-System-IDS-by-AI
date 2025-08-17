# merge_kitsune.py
# How to use

# Download dataset from Kaggle https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune/data
# Extract the Kaggle dataset in you prject folder "dataset".

import pandas as pd
import os

# Folder containing your CSVs
folder_path = "dataset"  # change if needed
os.makedirs(folder_path, exist_ok=True)

# File lists
datasets = [
    "ARP_MitM_dataset.csv", "Active_Wiretap_dataset.csv", "Fuzzing_dataset.csv",
    "OS_Scan_dataset.csv", "SSDP_Flood_dataset.csv", "SSL_Renegotiation_dataset.csv",
    "SYN_DoS_dataset.csv", "Video_Injection_dataset.csv"
]

labels = [
    "ARP_MitM_labels.csv", "Active_Wiretap_labels.csv", "Fuzzing_labels.csv",
    "OS_Scan_labels.csv", "SSDP_Flood_labels.csv", "SSL_Renegotiation_labels.csv",
    "SYN_DoS_labels.csv", "Video_Injection_labels.csv"
]

merged_dfs = []

for dataset, label in zip(datasets, labels):
    dataset_path = os.path.join(folder_path, dataset)   # <-- use folder_path
    label_path   = os.path.join(folder_path, label)     # <-- use folder_path

    # Load files
    df_data = pd.read_csv(dataset_path)
    df_labels = pd.read_csv(label_path)

    # Keep it simple: align by row order; trim to shortest just in case
    n = min(len(df_data), len(df_labels))
    df_data = df_data.iloc[:n].reset_index(drop=True)
    df_labels = df_labels.iloc[:n].reset_index(drop=True)

    # If labels file has exactly one column, call it 'label'
    if df_labels.shape[1] == 1 and df_labels.columns[0].lower() != "label":
        df_labels = df_labels.rename(columns={df_labels.columns[0]: "label"})

    # Optional: add a simple 'source' column (comment out if you don't want it)
    source_name = dataset.replace("_dataset.csv", "")
    df_data.insert(0, "source", source_name)

    # Merge side-by-side
    df_merged = pd.concat([df_data, df_labels], axis=1)

    # Save per-pair merged file
    merged_file_path = os.path.join(folder_path, f"merged_{source_name}.csv")
    df_merged.to_csv(merged_file_path, index=False)
    print(f"Saved merged file: {merged_file_path}")

    merged_dfs.append(df_merged)

# Combine all merged parts into a single file
final_merged = pd.concat(merged_dfs, axis=0, ignore_index=True)
final_path = os.path.join(folder_path, "merged_all.csv")
final_merged.to_csv(final_path, index=False)
print(f"\n✓ All datasets merged into: {final_path}")
