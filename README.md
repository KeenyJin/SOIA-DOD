# SOIA-DOD: Short-term Object Interaction Anticipation with Disentangled Object Detection

This is the github repository of the following technical report, prepared for Ego4D Short-term Object Interaction Anticipation Challenge 2024:

H. Cho, D. U. Kang, S. Y. Chun. Short-term Object Interaction Anticipation with Disentangled Object Detection.

Team ICL@SNU
- Ranked 3rd in Noun+TTC & Overall
- Ranked 1st in Noun & Noun+Verb

[Leaderboard](https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910)

## Installation

To install the necessary dependencies, run the following command:
```
pip install -r requirements.txt
```

## Ego4D Dataset

To train/test the model on the Ego4D dataset, follow the instructions provided here to download the dataset and its annotations for the Short-term Object Interaction Anticipation task:

`https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md`

Only the annotations and pre-extracted high-resolution image frames are required for this project.

## Training

### YOLOv9

You should fine-tune the pre-trained YOLOv9 object detector to predict the next active objects. You can download the fine-tuned weights [here](https://drive.google.com/file/d/1nlGRP-zKhLWj_HK_gNkoULXcdC3phRA2/view?usp=drive_link).

### SOIA-DOD

To train SOIA-DOD on the Ego4D dataset, first fill in the `img_path`, `anno_path`, and `yolo_checkpoint` in `configs/config.yaml`, and then execute the following command:

Single GPU
```
python main.py --output_dir <output_directory>
```

Multiple GPUs
```
torchrun --nproc_per_node=<gpu_number> main.py --output_dir <output_directory> --find_unused_params
```

Checkpoints will be saved in the output directory. Validation mAP results will be saved in `<output_directory>/map.json`.

## Validation

Trained models can be validated using the following command:

Single GPU
```
python main.py --output_dir <output_directory> --eval --resume <checkpoint_file>.pth
```

Multiple GPUs
```
torchrun --nproc_per_node=<gpu_number> main.py --output_dir <output_directory> --eval --resume <checkpoint_file>.pth --find_unused_params
```

Validation mAP results will be saved in `<output_directory>/map.json`.

## Test

To test the trained models, you can use the following command:

Single GPU
```
python main.py --output_dir <output_directory> --test --resume <checkpoint_file>.pth
```

Multiple GPUs
```
torchrun --nproc_per_node=<gpu_number> main.py --output_dir <output_directory> --test --resume <checkpoint_file>.pth --find_unused_params
```

Predictions will be saved in `<output_directory>/results/test_epoch<epoch>.json`. To obtain the mAP results, submit the file to the [challenge](https://eval.ai/web/challenges/challenge-page/1623/overview).

## Visualization

To visualize the predictions of the trained models, execute the following command:

Single GPU
```
python main.py --output_dir <output_directory> --visualize --resume <checkpoint_file>.pth
```

Multiple GPUs
```
torchrun --nproc_per_node=<gpu_number> main.py --output_dir <output_directory> --visualize --resume <checkpoint_file>.pth --find_unused_params
```

You can change the variable `eval_idxs` in function `visualize` in `main.py` to set the indices that you want to visualize.

Ground truth and top-5 prediction results will be saved in `<output_directory>/visualizations/`.

## Citation

## Acknowledgment

We would like to thank [Ego4D](https://github.com/facebookresearch/Ego4d), [StillFast](https://github.com/fpv-iplab/stillfast), [GANOv2](https://github.com/sanketsans/ganov2), [YOLOv9](https://github.com/WongKinYiu/yolov9), [CLIP](https://github.com/openai/CLIP), and [DINO](https://github.com/IDEA-Research/DINO) for their contributions and inspiration. These works have been instrumental in the development of this project.
