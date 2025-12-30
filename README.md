# modelling
This repository comprises works undertaken to evaluate and train AI models used throughout the project

## scope

### PV detection & segmentation
- performance assessment of YOLOv8-based detection models for solar panels
    - [ariel_ultimate_eval_X_25.ipynb](panels/ariel_ultimate_eval_X_25.ipynb)
    - [finloop_ultimate_eval_X_25.ipynb](panels/finloop_ultimate_eval_X_25.ipynb)
    - [pool_ultimate_eval_X_25.ipynb](panels/pool_ultimate_eval_X_25.ipynb)
- segmentation of solar panels on images (YOLO-segmentation model and YOLO+SAM composition)
    - [ariel_segment_with_sam.ipynb](panels/ariel_segment_with_sam.ipynb)
    - [finloop_segment.ipynb](panels/finloop_segment.ipynb)
    - [finloop_segment_with_sam.ipynb](panels/finloop_segment_with_sam.ipynb)
    - [finloop_segment_comparison.ipynb](panels/finloop_segment_comparison.ipynb)
    - [pool_segment_with_sam.ipynb](panels/pool_segment_with_sam.ipynb)
    - [SAM_segment_comparison.ipynb](panels/SAM_segment_comparison.ipynb)
    - [sam_comparison.csv](panels/sam_comparison.csv) 

### tree coverage detection & segmentation
- performance assessment of tree models (Deepforest+SAM, CLIPSeg, SegFormer)

## acknowledgements
> Obliczenia wykonano z wykorzystaniem komputerów Centrum Informatycznego Trójmiejskiej Akademickiej Sieci Komputerowej" (Computations were carried out using the computers of Centre of Informatics Tricity Academic Supercomputer & Network).

### PV models
- [finloop](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- [Ariel Drabkin's](https://github.com/ArielDrabkin/Solar-Panel-Detector)
- [Andrew Gray's](https://huggingface.co/andrewgray11/autotrain-solar-panel-object-detection-50559120777)