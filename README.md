# L-GCN on Surveillance Video-QA Dataset

Đây là phần source code tham khảo từ bài báo [Location-aware Graph Convolutional Networks for Video Question Answering](https://arxiv.org/abs/2008.09105) để chạy trên tập dữ liệu về giám sát an ninh do nhóm xây dựng nên.

Khuyến khích chạy colab nếu máy không có GPU, xem thêm hướng dẫn chạy trên colab tại [đây](https://colab.research.google.com/drive/1LCggpvJd6jQ9bm2VJ0EaEzplZYelLGYw?usp=sharing)

## Thiết lập
1. Tải code về máy
```
 git clone https://github.com/DoDucNhan/Surveilllance-VQA.git
```

2. Tải tập dữ liệu giám sát của nhóm [SNN-QA](https://drive.google.com/file/d/1MuEtb_FVnJFfZ33gPI0SLMcxUoXf50NF/view?usp=sharing) về máy, sau đó để các video vào thư mục theo đường dẫn `L-GCN/data/snn`. Những file chứa câu hỏi đã có sẵn trong đường dẫn `L-GCN/data/dataset`.

3. Cài đặt Mask R-CNN theo hướng dẫn ở [đây](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)

4. Cài đặt các thư viện cần thiết để chạy mô hình:
```bash
pip install -r requirements.txt
python -m spacy download en
```

## Tiền xử lý dữ liệu
Các lệnh tiền xử lý dữ liệu có thể tham khảo thêm ở [đây](https://github.com/SunDoge/L-GCN). Trước khi chạy tiền xử lý dữ liệu hãy chuyển vào thư mục `L-GCN` để thực hiện các bước tiếp theo.
#### Tiền xử lý đặc trưng câu hỏi
Để trích các đặc trưng câu hỏi, chạy lệnh:
```
python -m scripts.build_tgif_cache -t frameqa -o cache/snn
```
- `cache/snn` thư mục để lưu các đặc trưng câu hỏi trích xuất được.

#### Tiền xử lý đặc trưng hình ảnh
1. Tách frame từ video.
```
bash save-frames.sh video video_frames
```

2. Chia nhỏ những frame tách được thành từng phần để chạy.
```
python -m scripts.split_n_parts -f data/snn/video_frames -o data/snn/frame_splits/
```

3. Lấy các hộp giới hạn, thực hiện tương tự với cái file **split** còn lại.
```
python -m scripts.extract_bboxes_with_maskrcnn \
-f data/snn/frame_splits/split0.pkl \
-o data/snn/bboxes_splits/split0.pt \
-c config/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml
```

4. Kết hợp các file **split** lại với nhau.
```
python -m scripts.merge_box_scores_and_labels \
--bboxes data/snn/bboxes_splits \
-o data/snn/video_bboxes
```

5. Trích xuất đặc trưng từ các hộp giới hạn, thực hiện tương tự với các file **split** còn lại.
```
python -m scripts.extract_resnet152_features_with_bboxes \
-i data/snn/video_frames \
-f data/snn/frame_splits/split0.pkl \
-p data/snn/bboxes_splits/split0.pt \
-o data/snn/bbox_features_splits/split0layer
```

6. Kết hợp các file **split** lại với nhau.
```
python -m scripts.merge_bboxes \
--bboxes data/snn/bbox_features_splits \
-o data/snn/resnet152_bbox_features
```

7. Trích xuất đặc trưng pool5
```
python -m scripts.extract_resnet152_features \
-i data/snn/video_frames \
-o data/snn/resnet152_pool5_features
```

## Huấn luyện mô hình
Sử dụng lệnh sau để huấn luyện mô hình
```
!python train.py -c config/resnet152-bbox/frameqa.conf -e result
```

- `result` là thư mục để lưu kết quả.

## Kiểm tra kết quả huấn luyện trên từng loại câu hỏi.
Tập dữ liệu của nhóm có thể chia làm bốn loại dựa trên nội dung câu hỏi giành cho video, bao gồm `action, human, location, time`. Sử dụng lệnh sau để đánh giá mô hình trên từng loại câu hỏi.
1. Tiền xử lý đặc trưng câu hỏi
```
python -m scripts.build_snn_questionType -t frameqa -o cache/snn -q Action
```

- `-q` chính là loại câu hỏi muốn tiền xử lý, có thể thay bằng `Human` hoặc `Location` hoặc `Time`.

2. Đánh giá trên loại câu hỏi đã tiền xử lý
```
python testQA.py -e result -c config/resnet152-bbox/frameqa.conf -q Action
```

- `-q` chính là loại câu hỏi muốn tiền xử lý, có thể thay bằng `Human` hoặc `Location` hoặc `Time`.
