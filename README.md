# modelling
This repository comprises works undertaken to evaluate and train AI models used throughout the project

## scope
- performance assessment of YOLOv8-based detection models for solar panels
    - [ariel_ultimate_eval_X_25.ipynb](ariel_ultimate_eval_X_25.ipynb)
    - [finloop_ultimate_eval_X_25.ipynb](finloop_ultimate_eval_X_25.ipynb)
    - [pool_ultimate_eval_X_25.ipynb](pool_ultimate_eval_X_25.ipynb)
- segmentation of solar panels on images (YOLO-segmentation model and YOLO+SAM composition)
    - [ariel_segment_with_sam.ipynb](ariel_segment_with_sam.ipynb)
    - [finloop_segment.ipynb](finloop_segment.ipynb)
    - [finloop_segment_with_sam.ipynb](finloop_segment_with_sam.ipynb)
    - [finloop_segment_comparison.ipynb](finloop_segment_comparison.ipynb)
    - [pool_segment_with_sam.ipynb](pool_segment_with_sam.ipynb)
    - [SAM_segment_comparison.ipynb](SAM_segment_comparison.ipynb)
    - [sam_comparison.csv](sam_comparison.csv) 


## acknowledgements
> Obliczenia wykonano z wykorzystaniem komputerów Centrum Informatycznego Trójmiejskiej Akademickiej Sieci Komputerowej" (Computations were carried out using the computers of Centre of Informatics Tricity Academic Supercomputer & Network).

### PV models:
- [finloop](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- [Ariel Drabkin's](https://huggingface.co/spaces/ArielDrabkin/Solar-Panel-Detector/tree/main)
- [Andrew Gray's](https://huggingface.co/andrewgray11/autotrain-solar-panel-object-detection-50559120777)
