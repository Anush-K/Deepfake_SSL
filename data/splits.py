import os
import random

random.seed(42)

def get_splits(dataset_name, raw_path):

    if dataset_name == "FFPP":
        return get_ffpp_splits(raw_path)

    elif dataset_name == "CelebDF":
        return get_celebdf_splits(raw_path)

    elif dataset_name == "DFD":
        return get_dfd_splits(raw_path)

    else:
        raise ValueError("Unknown dataset")
    
def get_ffpp_splits(raw_path):

    splits = {"train": [], "val": [], "test": []}

    original_path = os.path.join(
        raw_path,
        "original_sequences/youtube/c23/videos"
    )

    manip_types = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter"
    ]

    fake_video_paths = []

    for manip in manip_types:
        manip_path = os.path.join(
            raw_path,
            f"manipulated_sequences/{manip}/c23/videos"
        )

        for vid in os.listdir(manip_path):
            if vid.endswith(".mp4"):
                fake_video_paths.append({
                    "path": os.path.join(manip_path, vid),
                    "label": 1,
                    "manipulation": manip,
                    "video_id": vid.replace(".mp4", "")
                })

    real_video_paths = []

    for vid in os.listdir(original_path):
        if vid.endswith(".mp4"):
            real_video_paths.append({
                "path": os.path.join(original_path, vid),
                "label": 0,
                "manipulation": "real",
                "video_id": vid.replace(".mp4", "")
            })

    all_videos = real_video_paths + fake_video_paths
    random.shuffle(all_videos)

    n = len(all_videos)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits["train"] = all_videos[:train_end]
    splits["val"] = all_videos[train_end:val_end]
    splits["test"] = all_videos[val_end:]

    return splits

def get_celebdf_splits(raw_path):

    splits = {"train": [], "val": [], "test": []}

    test_list_path = os.path.join(raw_path, "List_of_testing_videos.txt")

    with open(test_list_path, "r") as f:
        test_videos = set(line.strip() for line in f.readlines())

    categories = {
        "Celeb-real": 0,
        "YouTube-real": 0,
        "Celeb-synthesis": 1
    }

    all_train_videos = []

    for folder, label in categories.items():
        folder_path = os.path.join(raw_path, folder)

        for vid in os.listdir(folder_path):
            if not vid.endswith(".mp4"):
                continue

            full_path = os.path.join(folder_path, vid)

            video_info = {
                "path": full_path,
                "label": label,
                "manipulation": "fake" if label == 1 else "real",
                "video_id": vid.replace(".mp4", "")
            }

            if vid in test_videos:
                splits["test"].append(video_info)
            else:
                all_train_videos.append(video_info)

    random.shuffle(all_train_videos)

    n = len(all_train_videos)
    train_end = int(0.8 * n)

    splits["train"] = all_train_videos[:train_end]
    splits["val"] = all_train_videos[train_end:]

    return splits

def get_dfd_splits(raw_path):

    splits = {"train": [], "val": [], "test": []}

    fake_root = os.path.join(
        raw_path,
        "DFD_manipulated_sequences",
        "DFD_manipulated_sequences"
    )

    real_root = os.path.join(
        raw_path,
        "DFD_original_sequences"
    )

    video_entries = []

    # Load fake videos
    for vid in os.listdir(fake_root):
        if vid.endswith(".mp4"):
            video_entries.append({
                "path": os.path.join(fake_root, vid),
                "label": 1,
                "manipulation": "fake",
                "video_id": vid.replace(".mp4", "")
            })

    # Load real videos
    for vid in os.listdir(real_root):
        if vid.endswith(".mp4"):
            video_entries.append({
                "path": os.path.join(real_root, vid),
                "label": 0,
                "manipulation": "real",
                "video_id": vid.replace(".mp4", "")
            })

    random.shuffle(video_entries)

    n = len(video_entries)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits["train"] = video_entries[:train_end]
    splits["val"] = video_entries[train_end:val_end]
    splits["test"] = video_entries[val_end:]

    print(f"DFD loaded with {n} videos")
    print(f"Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    return splits
