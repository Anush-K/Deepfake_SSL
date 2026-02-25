import os
import cv2
import random
import argparse
from tqdm import tqdm
import numpy as np
from data.face_extract import FaceExtractor
from data.splits import get_splits
from data.metadata import MetadataWriter


def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def process_dataset(dataset_name, raw_path, processed_path, debug=False):
    face_extractor = FaceExtractor(target_size=224)
    splits = get_splits(dataset_name, raw_path)

    metadata_writer = MetadataWriter(
        save_path=os.path.join(processed_path, f"{dataset_name}_metadata.csv")
    )

    for split_name, video_list in splits.items():

        if debug:
            video_list = video_list[:3]

        for video_info in tqdm(video_list):
            video_path = video_info["path"]
            label = video_info["label"]
            manipulation = video_info.get("manipulation", "NA")
            video_id = video_info["video_id"]

            frames = sample_frames(video_path, num_frames=8)

            for i, frame in enumerate(frames):
                face = face_extractor.extract_face(frame)

                if face is None:
                    continue

                save_dir = os.path.join(
                    processed_path, dataset_name, split_name,
                    "real" if label == 0 else "fake"
                )
                os.makedirs(save_dir, exist_ok=True)

                filename = f"{video_id}_{i}.jpg"
                save_path = os.path.join(save_dir, filename)

                cv2.imwrite(save_path, face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                metadata_writer.add_entry(
                    image_path=save_path,
                    label=label,
                    dataset=dataset_name,
                    manipulation=manipulation,
                    video_id=video_id
                )

    metadata_writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    RAW_BASE = "/content/drive/MyDrive/DF_Datasets"
    PROCESSED_BASE = "/content/drive/MyDrive/DF_Datasets/processed"

    raw_path = os.path.join(RAW_BASE, f"{args.dataset}_raw")
    process_dataset(args.dataset, raw_path, PROCESSED_BASE, debug=args.debug)