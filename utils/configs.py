from dataclasses import dataclass

from paths import *


@dataclass
class CC359BaseConfig:
    use_accumulate_for_loss = True
    debug = False
    exp_dir = None
    data_len = 45
    input_size = (256, 256)
    base_splits_path = cc359_splits_dir
    base_res_path = cc359_results
    msm = False
    n_channels = 1
    save_pred_every = 500
    epoch_every = 1000
    n_chans_out = 2
    parallel_model = False
    drop_last = False


@dataclass
class CC359ConfigPretrain(CC359BaseConfig):
    source_batch_size = 16
    target_batch_size = 1
    num_steps = 5000
    lr = 1e-3
    sched = True
    sched_gamma = 0.1
    milestones = [8000, 9500]


@dataclass
class CC359ConfigPLPP(CC359BaseConfig):
    source_batch_size = 8
    target_batch_size = 8
    n_clusters = 12
    num_steps = 6500
    lr = 5e-6
    use_slice_num = True
    id_to_num_slices = 'id_to_num_slices.json'
    dist_loss_lambda = 0.9
    sched = True
    sched_gamma = 0.1
    acc_amount = 35
    milestones = [5000, 6000]
    use_adjust_lr = False


@dataclass
class DebugConfigCC359(CC359BaseConfig):
    n_clusters = 2
    source_batch_size = 4
    target_batch_size = 4
    lr = 1e-5
    data_len = 10
    save_pred_every = 5
    debug = True
    epoch_every = 40
    num_steps = 200
    dist_loss_lambda = 0.1
    use_slice_num = True
    id_to_num_slices = 'id_to_num_slices.json'
    sched = False
    use_adjust_lr = False
    acc_amount = 4


@dataclass
class MsmBaseConfig:
    use_accumulate_for_loss = True
    debug = False
    exp_dir = None
    input_size = (384, 384)
    base_splits_path = msm_splits_dir
    base_res_path = msm_results
    msm = True
    n_channels = 3
    save_pred_every = 50
    epoch_every = 50
    num_steps = 2500
    n_chans_out = 2
    drop_last = False


@dataclass
class MsmPretrainConfig(MsmBaseConfig):
    lr = 1e-3
    source_batch_size = 16
    target_batch_size = 1
    parallel_model = False
    sched = False


@dataclass
class MsmConfigPLPP(MsmBaseConfig):
    n_clusters = 12
    lr = 1e-6
    use_slice_num = True
    id_to_num_slices = 'id_to_num_slices_msm.json'
    source_batch_size = 8
    target_batch_size = 8
    dist_loss_lambda = 0.8
    parallel_model = True
    sched = False
    acc_amount = 20
    sched_gamma = 0.1
    milestones = [1000, 2000, 3000]
    use_adjust_lr = False


@dataclass
class DebugMsm(MsmBaseConfig):
    n_clusters = 2
    source_batch_size = 2
    target_batch_size = 2
    lr = 1e-5
    save_pred_every = 5
    debug = True
    epoch_every = 20
    num_steps = 50
    sched = False
    dist_loss_lambda = 0.1
    id_to_num_slices = 'id_to_num_slices_msm.json'
