import torch
from ultralytics import YOLO
'''
fine-tuning script for PV det+seg YOLO model
according to the recipe by prof. PS

developed against catastrophical forgetting

I 2026, MD
'''


BASE_MODEL = 'yolo11l-seg.pt'

OPTIMISER = 'auto'
OPTIMISER = 'SGD'
# OPTIMISER = 'AdamW'

PROJECT_NAME = 'fine_trening'
PROJECT_NAME = 'ft4'
PROJECT_NAME = f'{PROJECT_NAME}_{OPTIMISER}'

PILOT_D = './pilotPV_panels.v1i.yolov8-obb/data.yaml' # for full eval
SYNTH_D_NEW = './synth_dataset2/data.yaml' # 1st+2nd stage; 2nd split - no leakage by agumentations in train+test
SYNTH_D_OLD = './auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml' # for full eval
RZESZOW_D = './rzeszow_data/data.yaml' # eval on test
RZESZOW_VAL = './rzeszow_data/data_val.yaml' # 3rd training stage - rzesz/val
SYNTH_NEW_RZESZ_D = 'data_synth2_rzesz.yaml' # 2nd stage, synth/train+val with rzeszow/val for training

def training(model_pth: str, dataset: str, stage: str, lr=0.001, eps = 10, n_frozen=10) -> str:

    model = YOLO(model_pth)
    print('start training', stage)

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

    print('training done', stage)

    return str(model.trainer.best)


def evaluation(model_pth: str, dataset: str, stage: str, splt='test'):

    model = YOLO(model_pth)
    print('eval', stage)

    metrics = model.val(
        data=dataset,
        split=splt,
        project=PROJECT_NAME,
        name=f"{stage}_{splt}",
        # single_cls=True
    )

    print('eval done', stage)
    csv_data = metrics.to_csv(decimals = 3)
    with open(f"{PROJECT_NAME}/{stage}_{splt}.csv", 'w') as f:
        f.write(csv_data)

    return metrics


def many_eval(model):
    stg = 'feval_synth_new'
    evaluation(model, SYNTH_D_NEW, stg, 'train')
    evaluation(model, SYNTH_D_NEW, stg, 'val')
    evaluation(model, SYNTH_D_NEW, stg, 'test')

    stg = 'feval_synth_old'
    evaluation(model, SYNTH_D_OLD, stg, 'train')
    evaluation(model, SYNTH_D_OLD, stg, 'val')
    evaluation(model, SYNTH_D_OLD, stg, 'test')

    stg = 'feval_rzesz'
    evaluation(model, RZESZOW_D, stg, 'train')
    evaluation(model, RZESZOW_D, stg, 'val')
    evaluation(model, RZESZOW_D, stg, 'test')

    stg = 'feval_pilot'
    evaluation(model, PILOT_D, stg, 'test')


def tune_val(start_id: int, n: int, model: str, tr_dataset: str, val_dataset: str, stage: str, lr=0.001, eps = 10, n_frozen=10, splt='test'):
    """
    train and evaluate, repeat n times
    
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
    :param splt: dataset-specific, train/val/test/etc.
    """
    best_map = 0
    best_m_f1 = 0

    for id in range(start_id, start_id + n):
        stage = f'{id}_{stage}'
        model = training(model, tr_dataset, stage, lr, eps, n_frozen)
        metrics = evaluation(model, val_dataset, stage, splt)
        if metrics.box.map < best_map * 0.8 or metrics.seg.f1 < best_m_f1 * 0.8:
            print('performance decay!')
            break
        if metrics.box.map > best_map:
            print('outperformed by box-map')
            best_map = metrics.box.map
        if metrics.seg.map > best_m_f1:
            print('outperformed by mask-f1')
            best_m_f1 = metrics.seg.f1

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

    lr = 0.001
    # lr = 0.0005
    # lr = 0.0001
    mod, id = tune_val(1, 2, BASE_MODEL, SYNTH_D_NEW, RZESZOW_D, 's', lr) # train: synth-train, test: rzesz-test
    # many_eval(mod)
    mod, id = tune_val(id, 2, mod, SYNTH_NEW_RZESZ_D, RZESZOW_D, 'sr', lr/2) # train: synth-train+rzesz-val, test: rzesz-test
    # TODO: freeze more in 3
    mod, id = tune_val(id, 2, mod, RZESZOW_VAL, RZESZOW_D, 'r', lr/5) # train: rzesz-val, test: rzesz-test
    many_eval(mod)


if __name__ == '__main__':
    main()
