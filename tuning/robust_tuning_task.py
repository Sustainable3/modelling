import torch
from ultralytics import YOLO
from time import time
'''
fine-tuning script for PV det+seg YOLO model
according to the recipe by prof. PS

developed against catastrophical forgetting

I 2026, MD
'''


BASE_MODEL = 'yolo11l-seg.pt'

PERFORM_DECAY_LIMIT_FACTOR = 0.8
OPTIMISER = 'auto'
OPTIMISER = 'SGD'
OPTIMISER = 'AdamW'

PROJECT_NAME = 'fine_trening'
PROJECT_NAME = 'ft8'
PROJECT_NAME = f'{PROJECT_NAME}_{OPTIMISER}'

PILOT_D = './pilotPV_panels.v1i.yolov8-obb/data.yaml' # for full eval
SYNTH_D_NEW = './synth_dataset2/data.yaml' # 1st+2nd stage; 2nd split - no leakage by agumentations in train+test
SYNTH_D_NEW1 = './synth_dataset/data.yaml' # 1st+2nd stage; 2nd split - no leakage by agumentations in train+test
SYNTH_D_OLD = './auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml' # for full eval
RZESZOW_D = './rzeszow_data/data.yaml' # eval on test
RZESZOW_VAL = './rzeszow_data/data_val.yaml' # 3rd training stage - rzesz/val
SYNTH_NEW_RZESZ_D = 'data_synth2_rzesz.yaml' # 2nd stage, synth/train+val with rzeszow/val for training

def training(model_pth: str, dataset: str, stage: str, lr=0.001, eps = 10, n_frozen=10) -> str:

    model = YOLO(model_pth)
    print('start training', stage)
    t = time()

    model.train(
        data=dataset,
        epochs=eps,
        lr0=lr,
        lrf=1.0,
        batch=16,
        optimizer=OPTIMISER,
        freeze=n_frozen,

        project=PROJECT_NAME,
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


def evaluation(model_pth: str, dataset: str, stage: str, splt='test', is_feval=False):
    '''
    Evaluate a model on a dataset/split
    
    :param model_pth: path to a .pt
    :type model_pth: str
    :param dataset: .yaml path
    :type dataset: str
    :param stage: part of a nickname
    :type stage: str
    :param splt: a dataset split
    :param is_feval: for combining all metrics in a single .csv in full evaluation
    '''

    model = YOLO(model_pth)
    print('eval', stage, splt)
    t = time()

    metrics = model.val(
        data=dataset,
        split=splt,
        project=PROJECT_NAME,
        name=f"{stage}_{splt}",
        # single_cls=True
    )

    csv_data = metrics.to_csv(decimals = 3)
    t = time()-t
    print('eval done', stage, splt, 'in', t)
    
    pth = f"{PROJECT_NAME}/{stage}_{splt}.csv"
    f_mod = 'w'
    tm = '' # save eval time
    if is_feval:
        csv_data = csv_data.split('\n')[1] # skip header
        pth = f"{PROJECT_NAME}/feval.csv"
        f_mod = 'a'
        tm = f',{t}' # save eval time
    with open(pth, f_mod) as f:
        f.write(f'{stage},{splt},{csv_data}{tm}\n')

    return metrics


def many_eval(model):
    '''
    Evaluate the model on all available datasets and their splits
    
    :param model: YOLO model
    '''
    ds = 'synth_new2' # train-val-test split by src img
    evaluation(model, SYNTH_D_NEW, ds, 'train', True)
    evaluation(model, SYNTH_D_NEW, ds, 'val', True)
    evaluation(model, SYNTH_D_NEW, ds, 'test', True)
    
    ds = 'synth_new1' # shuffeled train-val-test split (src img and augmentations may be in any split)
    evaluation(model, SYNTH_D_NEW1, ds, 'train', True)
    evaluation(model, SYNTH_D_NEW1, ds, 'val', True)
    evaluation(model, SYNTH_D_NEW1, ds, 'test', True)

    ds = 'synth_old'
    evaluation(model, SYNTH_D_OLD, ds, 'train', True)
    evaluation(model, SYNTH_D_OLD, ds, 'val', True)
    evaluation(model, SYNTH_D_OLD, ds, 'test', True)

    ds = 'rzeszow'
    evaluation(model, RZESZOW_D, ds, 'train', True)
    evaluation(model, RZESZOW_D, ds, 'val', True)
    evaluation(model, RZESZOW_D, ds, 'test', True)

    ds = 'pilot'
    evaluation(model, PILOT_D, ds, 'test', True)


def tune_val(start_id: int, n: int, model: str, tr_dataset: str, val_dataset: str, stage: str, lr=0.001, eps = 10, n_frozen=10, v_splt='test'):
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
    :param lr: learning rate
    :param eps: no. epochs
    :param n_frozen: no. frozen layers
    :param v_splt: dataset-specific, train/val/test/etc.
    :return model: path to the best model
    :return id: part of a nickname
    """
    best_map = 0
    best_m_f1 = 0
    t = time()
    print('start phase', stage, 'for', n, 'reps')

    for id in range(start_id, start_id + n):
        stg = f'{id}_{stage}'
        model = training(model, tr_dataset, stg, lr, eps, n_frozen)
        metrics = evaluation(model, val_dataset, stg, v_splt)
        if metrics.box.map < best_map * PERFORM_DECAY_LIMIT_FACTOR or metrics.seg.f1 < best_m_f1 * PERFORM_DECAY_LIMIT_FACTOR:
            print('performance decay!')
            break
        if metrics.box.map > best_map:
            print('outperformed by box-map')
            best_map = metrics.box.map
        if metrics.seg.map > best_m_f1:
            print('outperformed by mask-f1')
            best_m_f1 = metrics.seg.f1

    print('done phase', stage, 'in', time()-t)
    return model, id


def main():
    """
    run 3 phases
    """
    # mod = training(BASE_MODEL, SYNTH_D_NEW, '1S', 0.001)
    # metr = evaluation(mod, RZESZOW_D, '1S')
    # mod = training(mod, SYNTH_D_NEW, '2S', 0.001)
    # metr = evaluation(mod, RZESZOW_D, '2S')

    # mod = training(mod, SYNTH_NEW_RZESZ_D, '3SR', 0.0001)
    # metr = evaluation(mod, RZESZOW_D, '3SR')
    # mod = training(mod, SYNTH_NEW_RZESZ_D, '4SR', 0.0001)
    # metr = evaluation(mod, RZESZOW_D, '4SR')
    
    # mod = training(mod, RZESZOW_VAL, '5SR', 0.00001, n_frozen=12)
    # metr = evaluation(mod, RZESZOW_D, '5SR')
    # mod = training(mod, RZESZOW_VAL, '6SR', 0.00001, n_frozen=12)
    # metr = evaluation(mod, RZESZOW_D, '6SR')

    t = time()
    no_layers = len(YOLO(BASE_MODEL).model.model)
    print(OPTIMISER, no_layers, 'layers')

    lr = 0.001
    lr = 0.0005
    # lr = 0.0001
    mod, id = tune_val(1, 3, BASE_MODEL, SYNTH_D_NEW, RZESZOW_D, 's', lr) # train: synth-train, test: rzesz-test
    # many_eval(mod)
    mod, id = tune_val(id, 2, mod, SYNTH_NEW_RZESZ_D, RZESZOW_D, 'sr', lr/2) # train: synth-train+rzesz-val, test: rzesz-test
    mod, id = tune_val(id, 2, mod, RZESZOW_VAL, RZESZOW_D, 'r', lr/5, n_frozen=no_layers-2) # train: rzesz-val, test: rzesz-test
    print('fine tuning took', time()-t)
    many_eval(mod)


if __name__ == '__main__':
    main()
