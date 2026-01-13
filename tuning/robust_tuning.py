from ultralytics import YOLO
from ultralytics.utils.metrics import Metric
from time import time
'''
fine-tuning script for PV det+seg YOLO model
according to the recipe by prof. PS

developed against catastrophical forgetting

I 2026, MD
'''


BASE_MODEL = 'yolo11l-seg.pt'
BASE_MODEL8 = 'yolov8l-seg.pt'
SEG = 'segment'
ARIEL = 'ariel.pt'
DET = 'detect'
FINLOOP = 'finloop.pt'

PERFORM_DECAY_LIMIT_FACTOR = 0.8

SGD = 'SGD'
ADAMW = 'AdamW'
RMS = 'RMSProp'

PROJECT_NAME = 'ft27'
LR = 0.0005

PILOT_D = './pilotPV_panels.v1i.yolov8-obb/data.yaml' # for full eval
SYNTH_D_NEW2 = './synth_dataset2/data_synth.yaml' # 1st+2nd stage; synth/train+val; 2nd split - no leakage by agumentations in train+test
SYNTH_D_NEW1 = './synth_dataset/data.yaml' # 1st split, previously too
SYNTH_D_OLD = './auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml' # for full eval
RZESZOW_D = './rzeszow_data/rzeszow_data.yaml' # 3rd training stage - rzesz/test; val=val; test=train
SYNTH_NEW_RZESZ_D = 'data_synth2_rzesz.yaml' # 2nd stage, synth/train+val with rzeszow/test for training


def training(model_pth: str, dataset: str, pn: str, stage: str, opt: str, lr: float = 0.001, eps: int = 10, n_frozen: int = 10) -> str:
    '''
    Train a model on a dataset/split
    
    :param model_pth: path to a .pt
    :type model_pth: str
    :param dataset: .yaml path
    :type dataset: str
    :param pn: project name
    :type pn: str
    :param stage: part of a nickname
    :type stage: str
    :param opt: optimiser
    :type opt: str
    :param lr: learning rate
    :type lr: float
    :param eps: no. epochs
    :type eps: int
    :param n_frozen: no. frozen layers
    :type n_frozen: int
    :return: path to the trained model
    :rtype: str
    '''

    model = YOLO(model_pth)
    print('start training', stage)
    t = time()

    model.train(
        data=dataset,
        epochs=eps,
        lr0=lr,
        lrf=0.5,
        batch=16,
        optimizer=opt,
        freeze=n_frozen,

        project=pn,
        name=stage,
        exist_ok=True,
        val=False,

        verbose=True,
        device=0,
        cache=True, # True==ram; disk
        # single_cls=True,
        pretrained=True,
    )

    print('training done', stage, 'in', time()-t)

    return str(model.trainer.best)


def evaluation(model_pth: str, dataset: str, pn: str, stage: str, splt: str = 'test', is_feval: bool = False) -> Metric:
    '''
    Evaluate a model on a dataset/split
    
    :param model_pth: path to a .pt
    :type model_pth: str
    :param dataset: .yaml path
    :type dataset: str
    :param pn: project name
    :type pn: str
    :param stage: part of a nickname
    :type stage: str
    :param splt: the validation dataset split
    :type splt: str
    :param is_feval: for combining all metrics in a single .csv in full evaluation
    :type is_feval: bool
    :return: metrics
    :rtype: SegmentMetrics
    '''

    model = YOLO(model_pth)
    print('eval', stage, splt)
    t = time()

    metrics = model.val(
        data=dataset,
        split=splt,
        project=pn,
        name=f"{stage}_{splt}",
        # single_cls=True
    )

    csv_data = metrics.to_csv(decimals = 3)
    t = time()-t
    print('eval done', stage, splt, 'in', t)
    
    pth = f"{pn}/{pn}_{stage}_{splt}.csv"
    f_mod = 'w'
    tm = '' # save eval time
    if is_feval:
        csv_data = csv_data.split('\n')[1] # skip header
        pth = f"{pn}/{pn}_feval.csv"
        f_mod = 'a'
        tm = f',{t}' # save eval time
    with open(pth, f_mod) as f:
        f.write(f'{stage},{splt},{csv_data}{tm}\n')

    return metrics


def many_eval(model: str, proj_name: str):
    '''
    Evaluate the model on all available datasets and their splits
    
    :param model: YOLO model
    :type model: str
    :param proj_name: project name
    :type proj_name: str
    '''

    ds = 'synth_new2' # train-val-test split by src img
    evaluation(model, SYNTH_D_NEW2, proj_name, ds, 'train', True) # train+val
    evaluation(model, SYNTH_D_NEW2, proj_name, ds, 'val', True)
    evaluation(model, SYNTH_D_NEW2, proj_name, ds, 'test', True)
    
    ds = 'synth_new1' # shuffeled train-val-test split (src img and augmentations may be in any split)
    evaluation(model, SYNTH_D_NEW1, proj_name, ds, 'train', True)
    evaluation(model, SYNTH_D_NEW1, proj_name, ds, 'val', True)
    evaluation(model, SYNTH_D_NEW1, proj_name, ds, 'test', True)

    ds = 'synth_old'
    evaluation(model, SYNTH_D_OLD, proj_name, ds, 'train', True)
    evaluation(model, SYNTH_D_OLD, proj_name, ds, 'val', True)
    evaluation(model, SYNTH_D_OLD, proj_name, ds, 'test', True)
    
    ds = 'mix_synth_rzesz'
    evaluation(model, SYNTH_NEW_RZESZ_D, proj_name, ds, 'train', True)

    ds = 'rzeszow'
    evaluation(model, RZESZOW_D, proj_name, ds, 'train', True) # 2nd + 3rd stage training on rzeszow/test
    evaluation(model, RZESZOW_D, proj_name, ds, 'val', True)
    evaluation(model, RZESZOW_D, proj_name, ds, 'test', True) # rzeszow/train

    ds = 'pilot'
    evaluation(model, PILOT_D, proj_name, ds, 'test', True)


def tune_val(start_id: int, n: int, model: str, tr_dataset: str, val_dataset: str, mode: str, proj_name: str, stage: str, 
             opt: str, lr: float = 0.001, eps: int = 10, n_frozen: int = 10, v_splt: str = 'val') -> tuple[str, int]:
    """
    train and evaluate, repeat n times. stop if the performance drops by 20% from the best
    
    :param start_id: part of a nickname
    :type start_id: int
    :param n: how many times train-evaluate pair should be done
    :type n: int
    :param model: path to a .pt
    :type model: str
    :param tr_dataset: training set, path to a .yaml
    :type tr_dataset: str
    :param val_dataset: test set, path to a .yaml
    :type val_dataset: str
    :param stage: part of a nickname
    :type stage: str
    :param opt: optimiser
    :type opt: str
    :param lr: learning rate
    :type lr: float
    :param eps: no. epochs
    :type eps: int
    :param n_frozen: no. frozen layers
    :type n_frozen: int
    :param v_splt: dataset-specific, train/val/test/etc.
    :type v_splt: str
    :return: model: path to the best model, id: part of a nickname
    :rtype: tuple[str, int]
    """

    best_map = 0
    best_m_f1 = 0
    t = time()
    print('start phase', stage, 'for', n, 'reps')

    for id in range(start_id, start_id + n):
        stg = f'{id}_{stage}'
        model = training(model, tr_dataset, proj_name, stg, opt, lr, eps, n_frozen)
        metrics = evaluation(model, val_dataset, proj_name, stg, v_splt)
        if metrics.box.map < best_map * PERFORM_DECAY_LIMIT_FACTOR \
        or mode == SEG and metrics.seg.f1 < best_m_f1 * PERFORM_DECAY_LIMIT_FACTOR:
            print('performance decay!')
            break
        if metrics.box.map > best_map:
            print('outperformed by box-map')
            best_map = metrics.box.map
        if mode == SEG and metrics.seg.map > best_m_f1:
            print('outperformed by mask-f1')
            best_m_f1 = metrics.seg.f1

    print('done phase', stage, 'in', time()-t)
    return model, id


def main(opt: str = SGD, base: str = BASE_MODEL, mode: str = SEG):
    """
    run 3 phases
    
    :param opt: optimiser
    :type opt: str
    :param base: base model
    :type base: str
    :param mode: YOLO task
    :type mode: str
    """
    pn = f'{PROJECT_NAME}_{opt}'
    if base != BASE_MODEL:
        pn = f'{pn}_{base}'        

    t = time()
    no_layers = len(YOLO(base).model.model)
    print(opt, no_layers, 'layers')

    lr = LR # 0.001
    # lr = 0.0005
    # lr = 0.0001

    mod, id = tune_val(1, 4, base, SYNTH_D_NEW2, RZESZOW_D, mode, pn, 's', opt, lr) # train: synth/train+val, test: rzesz-val
    # many_eval(mod)
    mod, id = tune_val(id, 4, mod, SYNTH_NEW_RZESZ_D, RZESZOW_D, mode, pn, 'sr', opt, lr/2) # train: synth/train+val + rzesz/test, test: rzesz-val
    mod, id = tune_val(id, 4, mod, RZESZOW_D, RZESZOW_D, mode, pn, 'r', opt, lr/5, n_frozen=no_layers-2) # train: rzesz/test, test: rzesz-val

    t = time() - t
    many_eval(mod, pn)
    print(pn, 'fine tuning took', t)


if __name__ == '__main__':
    main(SGD, FINLOOP, SEG)
    main(SGD, BASE_MODEL, SEG)
    main(ADAMW, BASE_MODEL, SEG)
    main(ADAMW, FINLOOP, SEG)
    # main(RMS, BASE_MODEL, SEG)
    # main(SGD, ARIEL, DET)
    # main(ADAMW, ARIEL, DET)
    # main(RMS, ARIEL, DET)

