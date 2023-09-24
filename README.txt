# SoICT Hackathon 2023 - Vietnamese Handwritten Text Recognition - Team: Crusader

## Table of Contents

- [Environment Setup:](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Model](#train)
- [Inference On Public Test Data](#inference-on-testing-data)

## Environment Setup:

We recommend to use Docker for building environment.

### 1. Pull Docker Image From Docker Hub:

```
docker pull 21522542/crusader
```

### 2. Run Docker container:

```
docker run --name <YOUR_CONTAINER_NAME> --gpus all -v <ABSOLUTE_PATH_TO_SOURCE_CODE_FOLDER>:/workspace -w /workspace -it --shm-size=1g 21522542/crusader:latest bash
```
Where <ABSOLUTE_PATH_TO_SOURCE_CODE_FOLDER> is path to **src folder in zip file after unzip**.
\
\
Example:
```
docker run --name crusader_bkai2023 --gpus all -v /src:/workspace -w /workspace -it --shm-size=1g 21522542/crusader:latest bash
```

## Data Preparation

You can download and reformat dataset by running:

```
cd /workspace
bash scripts/prepare_data.sh
```
If you can't download data from code, you can directly download it from url in file prepare_data.sh and put it in correct structure then run this code again.

## Training Model

We train each model on 1 GPU RTX 3090 with 24GB VRAM **for about 6 hours**, total approximate about 50 hours. You can train models by running:

```
cd /workspace
bash scripts/train.sh
```

You can read _train.sh_ to see more details about training process. Besides, you can also train each model independently.
\
\
Example:

<details>
    <summary>ABINET</summary>

      cd /workspace/mmocr
      python tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet_v1 \
You can change parameters by editing config file in */mmocr/configs* folder.
</details>

<details>
      <summary>PARSEQ</summary>

      cd /workspace/parseq
      python3.8 train.py

</details>

<details>
      <summary>VIETOCR</summary>
      
      cd /workspace/vietocr
      python3.8 train_vietocr.py
</details>

## Inference On Public Test Data
### Inference after training
After training process, for inference to submit, we provide a file script to run the complete pipeline.

```
bash scripts/inference_train.sh
```

### Inference with pretrained weights
If you want to inference with best pretrained weights to get our score in public board, you can download pretrained weights by:

```
bash scripts/download_ckpts.sh
```

Then, run inference script:

```
bash scripts/inference.sh
```

The file **result.zip** to submit will be in folder **workspace**.
