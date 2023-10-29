# 1. Installation
Pull các docker images bằng các lệnh sau:
```
dokcer pull 21522542/crusader_dbnetpp:v1
docker pull 21522542/crusader_corner:v1
docker pull 21522542/crusader:latest
```

Tải sourcecode từ github:
```
git clone https://github.com/hoho303/BKAI-NAVER-OCR-2023.git
```

Sau khi có sourcecode, chạy 3 containers với các lệnh sau:
### Container PIPELINE
```
docker run -it --name CRUSADER_BKAI_PIPELINE \ 
--gpus all --cpus 20 --shm-size 1gb \
--mount type=bind,source=<path to sourcecode folder>,target=/workspace \
21522542/crusader:latest
```

### Container CORNER_TRANSFORMER
```
docker run -it --name CRUSADER_BKAI_CORNER \ 
--gpus all --cpus 20 --shm-size 1gb \
--mount type=bind,source=<path to sourcecode folder>,target=/workspace \
 21522542/crusader_corner:v1
```

### Container DBNETPP
```
docker run -it --name CRUSADER_BKAI_DBNETPP \ 
--gpus all --cpus 20 --shm-size 1gb \
--mount type=bind,source=<path to sourcecode folder>,target=/workspace \
 21522542/crusader_dbnetpp:v1
```

# 2. Prepare dataset
```
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspae
sh scripts/prepare_data.sh
```

# 3. Training

## Đối với các mô hình mmocr và parseq
```
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspace
sh scripts/train_mmocr_103.sh
```

## Đối với mô hình Corner Transformer, train from scracth:
```
docker exec -it CRUSADER_BKAI_CORNER bash
cd /workspace
sh scripts/train_mmocr_060.sh
```

# 4. Inference với best checkpoints
## 4.1. Download best checkpoints
```
# Tải checkpoints của chúng tôi
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspace
sh scripts/download_ckpts.sh

# Tải checkpoints dbnetpp 
mkdir /workspace/DBNetpp/pretrained
cd /workspace/DBNetpp/pretrained
wget https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth
```
## 4.2. Inference
Quá trình inference được chia làm 4 giai đoạn:
### 4.2.1. Giai đoạn 1:
Tạo dictionary với các từ trong tập train
```
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspace
python3 utils/create_dict.py 
```
Sau khi chạy xong, 1 file dict.json sẽ xuất hiện trong /workspace/data

#### 4.2.2. Giai đoạn 2: 
Infer trực tiếp trên private test
```
# Đối với các mô hình mmocr và parseq
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspace
sh scripts/inference_mmocr_103_test.sh

# Đối với mô hình corner transformer
docker exec -it CRUSADER_BKAI_CORNER bash
cd /workspace
sh scripts/inference_mmocr_060_test.sh

# Tổng hợp dự đoán từ các mô hình và dictionary file
# Sau khi tổng hợp ta sẽ nhận được  file prediction.txt
python3 utils/ensemble.py
```

### 4.2.3. Giai đoạn 3:
Dùng dbnetpp để trích xuất bounding box của các từ nhằm thu  hẹp nhiễu của background cho các recognition model dự đoán. Lọc các bounding box mong muốn có confidence score >= 0.75. Nếu số lượng boudning box mong muốn kkhác 1 thì sẽ giữ nguyên hình ảnh, ngược lại ta sẽ crop dể dự  đoán.
```
## Trích xuất bounding box từ private test.
## Crop những ảnh có số bounding box (có confidence score >= 0.75). 
## Những ảnh được crop sẽ được luư trong folder 
## workspace/DBNetpp/mmocr/infernece-result/crop-image
docker exec -it CRUSADER_BKAI_DBNETPP bash
cd /workspace
sh scripts/inference_mmocr_063_test.sh

## Infernece các mô hình trên dữ liệu đã được cropped
# Đối với corner transformer
docker exec -it CRUSADER_BKAI_CORNER bash
cd /workspace
sh scripts/inference_det_rec_mmocr_060.sh

# Đối với mmocr, parseq
docker exec -it CRUSADER_BKAI_PIPELINE bash
cd /workspace
sh scripts/inference_det_rec_mmocr_103.sh

## Tổng hợp kết quả dự đoán  của các mô hình ngoài trừ corner transformer
## Sau khi tổng hợp ta sẽ được file ensemble_crop.txt
cd /workspace
python3 utils/ensemble_crop.py
```

### 4.2.4. Giai đoạn 4: Final ensemble
Tổng hợp dự đoán từ  prediction.txt (kết quả  tổng hợp ở giai đoạn 2)  và ensemble_crop.txt (kết quả tổng hợp ở giai đoạn 3), corner_crop.txt .
```
docker exec -it CRUSADER_BKAI_PIPELINE
cd /workspace
python3 utils/final_ensemble.py
```