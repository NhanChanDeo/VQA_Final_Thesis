# TextVQA Dataset & Downloads

> **Official Website:** [https://textvqa.org/dataset/](https://textvqa.org/dataset/)  
> **Challenge / EvalAI:** [TextVQA Challenge on EvalAI](https://evalai.cloudcv.org/web/challenges/challenge-page/244/)  
> **Official Code:** [facebookresearch/mmf](https://github.com/facebookresearch/mmf)  
> **Paper:** ["Towards VQA Models That Can Read" (Singh et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html)  
> **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
> **Contact:** `textvqa@fb.com`

---

## Overview

TextVQA requires models to read and reason about text in images to answer questions. The dataset contains **45,336 questions** defined over **28,408 images** from the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset.

---

## Dataset Version 0.5.1 (Current Version)

> **Note on Version 0.5.1:**  
> Version 0.5.1 contains the exact same questions and image splits as Version 0.5, but **Rosetta OCR tokens have been updated to v0.2** and moved to dedicated standalone JSON files.

### 1. Questions and Annotations

| Split | Questions | File Size | Direct Download URL |
| :--- | :--- | :--- | :--- |
| **Train** | 34,602 | ~103 MB | [TextVQA_0.5.1_train.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json) |
| **Validation** | 5,000 | ~16 MB | [TextVQA_0.5.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) |
| **Test** | 5,734 | ~13 MB | [TextVQA_0.5.1_test.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json) |

### 2. Images Archives

*Images are sourced directly from OpenImages.*

| Split | Images Count | File Size | Direct Download URL |
| :--- | :--- | :--- | :--- |
| **Train + Val Images** | 25,119 (21,953 train + 3,166 val) | ~6.6 GB | [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) |
| **Test Images** | 3,289 | ~926 MB | [test_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip) |

### 3. Official Rosetta OCR Tokens (v0.2)

*Extracted using the Meta AI Rosetta OCR system.*

| Split | OCR Version | Direct Download URL |
| :--- | :--- | :--- |
| **Train OCR** | Rosetta OCR v0.2 | [TextVQA_Rosetta_OCR_v0.2_train.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json) |
| **Val OCR** | Rosetta OCR v0.2 | [TextVQA_Rosetta_OCR_v0.2_val.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json) |
| **Test OCR** | Rosetta OCR v0.2 | [TextVQA_Rosetta_OCR_v0.2_test.json](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json) |

---

## OpenImages Metadata & Image Rotation

Some OpenImages images require rotation adjustments based on EXIF/OpenImages metadata. The rotation CSV files are provided below:

* **Train Images Rotation Metadata:** [train-images-boxable-with-rotation.csv](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv)
* **Test Images Rotation Metadata:** [test-images-with-rotation.csv](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv)

---

## Dataset Version 0.5 (Legacy Version)

| Split | Questions | Images Archive |
| :--- | :--- | :--- |
| **Train** | [TextVQA_0.5_train.json (103 MB)](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_train.json) | [train_val_images.zip (6.6 GB)](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) |
| **Validation** | [TextVQA_0.5_val.json (16 MB)](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_val.json) | *(included in `train_val_images.zip`)* |
| **Test** | [TextVQA_0.5_test.json (13 MB)](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_test.json) | [test_images.zip (926 MB)](https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip) |

---

## Data Structures & Annotation Formats

### 1. Questions and Annotations Schema (`TextVQA_0.5.1_*.json`)

```json
{
  "dataset_name": "textvqa",
  "dataset_type": "train",
  "dataset_version": "0.5.1",
  "data": [
    {
      "question_id": 24135,
      "question": "what is the brand of the camera?",
      "question_tokens": ["what", "is", "the", "brand", "of", "the", "camera"],
      "image_id": "0123456789abcdef",
      "image_classes": ["Camera", "Electronics"],
      "flickr_original_url": "https://...",
      "flickr_300k_url": "https://...",
      "image_width": 1024,
      "image_height": 768,
      "set_name": "train",
      "answers": [
        "canon",
        "canon",
        "canon",
        "canon",
        "canon",
        "canon",
        "canon",
        "canon",
        "canon",
        "canon"
      ]
    }
  ]
}
```

### 2. OCR Annotations Schema (`TextVQA_Rosetta_OCR_v0.2_*.json`)

```json
{
  "dataset_name": "textvqa_ocr",
  "dataset_type": "train",
  "dataset_version": "0.2",
  "data": {
    "0123456789abcdef": {
      "image_id": "0123456789abcdef",
      "ocr_tokens": ["CANON", "EOS", "5D"],
      "ocr_info": [
        {
          "word": "CANON",
          "bounding_box": {
            "top_left_x": 0.42,
            "top_left_y": 0.35,
            "width": 0.12,
            "height": 0.04,
            "rotation": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pitch": 0.0
          }
        }
      ]
    }
  }
}
```

---

## Evaluation Metric

Submissions are evaluated using the standard **VQA Accuracy** metric:

$$\text{Accuracy}(\text{pred}) = \min\left(\frac{\text{number of humans with identical answer}}{3}, 1\right)$$

* Answers are normalized (lower-cased, punctuation stripped, digit/word normalization) before computing accuracy.
