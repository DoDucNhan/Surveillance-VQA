#!/usr/bin/env bash
for i in 1 2 3 4 5; do
    python -m scripts.extract_resnet152_features_with_bboxes \
    -i data/snn/video_frames \
    -f data/snn/frame_splits/split${i}.pkl \
    -p data/snn/bboxes_splits/split${i}.pt \
    -o data/snn/bbox_features_splits/split${i}layer
done
