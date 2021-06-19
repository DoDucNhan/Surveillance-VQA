#!/usr/bin/env bash
for i in 1 2 3 4 5; do
    python -m scripts.extract_bboxes_with_maskrcnn \
    -f data/snn/frame_splits/split${i}.pkl \
    -o data/snn/bboxes_splits/split${i}.pt \
    -c config/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml
done
