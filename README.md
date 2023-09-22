# SoICT Hackathon 2023 - Vietnamese Handwritten Text Recognition - Team: Crusader

## Table of Contents

- [Environment Setup:](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Model](#train)
- [Inference On Public Test Data](#inference-on-testing-data)

## Environment Setup:

We recommend to use Docker for building environment.

1. Build Docker Image with Dockerfile:

```
docker build -t soict2023-htr - < Dockerfile
```

2. Run Docker container:

```
bash docker/run.sh
```

## Data Preparation

You can download and reformat dataset by running:

```
bash scripts/prepare_data.sh
```

## Training Model

We train models on 1 GPU RTX 3090 with 24GB VRAM. You can train models by running:

```
bash scripts/train.sh
```

You can read _train.sh_ to see more details about training process. Besides, you can also train each model independently.
\
\
Example:

<details>
    <summary>ABINET</summary>

      python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet_v1 \
You can change parameters by editing config file in */mmocr/configs* folder.
</details>

<details>
      <summary>PARSEQ</summary>
    
      python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet_v1 \
</details>

<details>
      <summary>VIETOCR</summary>
    
      python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet_v1 \
</details>

## Inference On Public Test Data
For inference to submit, we provide a file script to run the complete pipeline. The file prediction.zip to submit will be in folder **output**.

```
bash scripts/inference.sh
```