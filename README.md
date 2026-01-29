# modelling
This repository comprises works undertaken to evaluate, fine-tune and employ AI models used throughout the project

## scope

### analysis & conclusion
[correlation.ipynb](correlation_v2.ipynb)

### PV detection & segmentation
- model deployment
    - [inference code](panels/yolo_pv_segment_sum.py)
    - [SLRUM script](panels/job_inf_pv.sh)
- performance assessment of YOLOv8-based detection models for solar panels
    - [baseline metrics](panels/evaluation_results.csv)
    - [ariel_ultimate_eval_X_25.ipynb](panels/scheme_exploration/ariel_ultimate_eval_X_25.ipynb)
    - [finloop_ultimate_eval_X_25.ipynb](panels/scheme_exploration/finloop_ultimate_eval_X_25.ipynb)
    - [pool_ultimate_eval_X_25.ipynb](panels/scheme_exploration/pool_ultimate_eval_X_25.ipynb)
- segmentation of solar panels on images (YOLO-segmentation model and YOLO+SAM composition)
    - [ariel_segment_with_sam.ipynb](panels/scheme_exploration/ariel_segment_with_sam.ipynb)
    - [finloop_segment.ipynb](panels/scheme_exploration/finloop_segment.ipynb)
    - [finloop_segment_with_sam.ipynb](panels/scheme_exploration/finloop_segment_with_sam.ipynb)
    - [finloop_segment_comparison.ipynb](panels/scheme_exploration/finloop_segment_comparison.ipynb)
    - [pool_segment_with_sam.ipynb](panels/scheme_exploration/pool_segment_with_sam.ipynb)
    - [SAM_segment_comparison.ipynb](panels/scheme_exploration/SAM_segment_comparison.ipynb)
    - [sam_comparison.csv](panels/scheme_exploration/sam_comparison.csv)
    - [yolo_pv_segment_sum_val.py](panels/yolo_pv_segment_sum_val.py)
 
### PV fine-tuning
- mixed datasets
    - [tuning script](tuning/robust_tuning.py)
    - [tuning and baseline results](tuning/evaluation_resultsft21SGD2.pt.csv)
    - [tuning scenario comparisons](tuning/compare_pv_ft_seg.xlsx)
    - [notes](tuning/note.txt)
- pure synthetic dataset
    - [tuning and baseline results](tuning/previous/full_res.csv)
    - [tuning script](tuning/previous/yolo_fine_tune_ariel.py)
- Fresno dataset
    - [tuning script](fresno/fine_tune.py)
    - [high-res img evaluation](fresno/combined_pv_eval_large_res.py)
    - [baseline results](fresno/fresno_results2.csv)

### tree coverage detection & segmentation
- deployment
    - [inference code](tree_area/trees_inference.py)
    - [SLURM script](tree_area/job_inf_tr.sh)
- varia
    - [performance assessment of tree models (Deepforest+SAM, CLIPSeg, SegFormer)](trees/tree_models.ipynb)
    - [tree_area_calculation](tree_area/tree_area_calculation)

## acknowledgements
> Obliczenia wykonano z wykorzystaniem komputerów Centrum Informatycznego Trójmiejskiej Akademickiej Sieci Komputerowej (Computations were carried out using the computers of Centre of Informatics Tricity Academic Supercomputer & Network).

### PV models
- [finloop](https://huggingface.co/finloop/yolov8s-seg-solar-panels)
- [Ariel Drabkin's](https://github.com/ArielDrabkin/Solar-Panel-Detector)
- [Andrew Gray's](https://huggingface.co/andrewgray11/autotrain-solar-panel-object-detection-50559120777)

### datasets
- [Fresno](https://doi.org/10.6084/m9.figshare.3385780)
- [synthetic](https://github.com/Sustainable3/syntetic_data_creation_labels_onclick)
- [3City imagery](https://github.com/Sustainable3/Inference_data)
