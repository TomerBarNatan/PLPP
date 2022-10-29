import numpy as np
import json
import wandb
import surface_distance.metrics as surf_dc
import torch
from scipy import ndimage
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm


def after_step(model, config, step_num, epochs, test_ds, val_ds_source, val_ds, args):
    global best_metric
    global low_source_metric
    global prev_d_score
    if step_num % 1 == 0 and step_num != 0:
        if config.msm:
            dice1, sdice1 = get_dice(model, val_ds, args.gpu, config)
            main_metric = dice1
        else:
            dice1, sdice1 = get_sdice(model, val_ds, args.gpu, config)
            main_metric = sdice1
        if val_ds_source is not None:
            if config.msm:
                dice_source, sdice_source = get_dice(model, val_ds_source, args.gpu, config)
                main_metric_source = dice_source
            else:
                dice_source, sdice_source = get_sdice(model, val_ds_source, args.gpu, config)
                main_metric_source = sdice_source
            if main_metric_source < low_source_metric:
                low_source_metric = main_metric_source
                torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_low_source_model.pth')
            wandb.log(
                {f'pseudo_labeling_dice/val_source': dice_source, f'pseudo_labeling_sdice/val_source': sdice_source},
                step=step_num)
        improvement = main_metric / prev_d_score
        print(f"improvement is {improvement}")
        wandb.log({f'pseudo_labeling_dice/val': dice1, f'pseudo_labeling_sdice/val': sdice1,
                   'pseudo_labels/improvement': improvement}, step=step_num)
        prev_d_score = improvement
        print(f'pseudo_labeling_dice is ', dice1)
        print(f'pseudo_labeling_sdice is ', sdice1)
        print('pseudo_labeling taking snapshot ...')

        if main_metric > best_metric:
            best_metric = main_metric
            print("new best metric!")
            torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_model.pth')
    if step_num == 0 or step_num == epochs - 1:

        title = 'end' if step_num != 0 else 'start'
        scores = {}
        if config.msm:
            dice_test, sdice_test = get_dice(model, test_ds, args.gpu, config)
        else:
            dice_test, sdice_test = get_sdice(model, test_ds, args.gpu, config)

        prev_d_score = sdice_test

        scores[f'pseudo_labeling_dice_{title}/test'] = dice_test
        scores[f'pseudo_labeling_sdice_{title}/test'] = sdice_test
        print(f"dice {title} is: {dice_test}")
        print(f"sdice {title} is: {sdice_test}")
        if step_num != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'pseudo_labeling_best_model.pth', map_location='cpu'))
            if config.msm:
                dice_test_best, sdice_test_best = get_dice(model, test_ds, args.gpu, config)
            else:
                dice_test_best, sdice_test_best = get_sdice(model, test_ds, args.gpu, config)
            scores[f'pseudo_labeling_dice_{title}/test_best'] = dice_test_best
            scores[f'pseudo_labeling_sdice_{title}/test_best'] = sdice_test_best
            if val_ds_source is not None:
                model.load_state_dict(
                    torch.load(config.exp_dir / f'pseudo_labeling_low_source_model.pth', map_location='cpu'))
                if config.msm:
                    dice_test_low_source, sdice_test_low_source = get_dice(model, test_ds, args.gpu, config)
                else:
                    dice_test_low_source, sdice_test_low_source = get_sdice(model, test_ds, args.gpu, config)
                scores[f'pseudo_labeling_dice_{title}/test_low_source_on_target'] = dice_test_low_source
                scores[f'pseudo_labeling_sdice_{title}/test_low_source_on_target'] = sdice_test_low_source

        wandb.log(scores, step=step_num)
        json.dump(scores, open(config.exp_dir / f'pseudo_labeling_scores_{title}.json', 'w'))


def sdice(gt, pred, spacing, tolerance=1):
    surface_distances = surf_dc.compute_surface_distances(gt, pred, spacing)

    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance), dice(gt, pred)


def _connectivity_region_analysis(mask):
    label_im, nb_labels = ndimage.label(mask)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im


def dice(gt, pred):
    if gt.shape != pred.shape:
        gt = gt.squeeze(1)
    g = np.zeros(gt.shape)
    p = np.zeros(pred.shape)
    g[gt == 1] = 1
    p[pred == 1] = 1
    return (2 * np.sum(g * p)) / (np.sum(g) + np.sum(p))


def dice_torch(gt, pred, smooth=0):
    if gt.shape != pred.shape:
        gt = gt.squeeze(1)
    g = torch.zeros(gt.shape)
    p = torch.zeros(pred.shape)
    g[gt == 1] = 1
    p[pred == 1] = 1
    return (2 * torch.sum(g * p) + smooth) / (torch.sum(g) + torch.sum(p) + smooth)


def get_sdice(model, ds, device, config):
    if config.debug:
        return 0.5, 0.6
    loader = data.DataLoader(ds, batch_size=1, shuffle=False)
    model.eval()
    prev_id = None
    all_segs = []
    all_preds = []
    done_ids = set()
    all_sdices = []
    all_dices = []
    with torch.no_grad():

        for images, segs, ids, slc_num in tqdm(loader, desc='running test loader', position=0, leave=True):
            id1 = int(ids[0])
            _, output = model(images.to(device))
            if output.shape[1] == 2:
                output = output.cpu().data.numpy()
                output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            else:
                assert output.shape[1] == 1
                output = (nn.Sigmoid()(output) > 0.5).squeeze(1).cpu().data.numpy()
            segs = segs.squeeze(1).numpy().astype(bool)
            if prev_id is None:
                prev_id = id1
            if id1 != prev_id:
                assert id1 not in done_ids
                done_ids.add(id1)
                id1_str = str(id1)
                while len(id1_str) < 3:
                    id1_str = '0' + id1_str
                sdice1, dice1 = sdice(np.stack(all_segs), np.stack(all_preds), ds.spacing_loader('CC0' + id1_str))
                all_sdices.append(sdice1)
                all_dices.append(dice1)
                all_preds = []
                all_segs = []
            prev_id = id1
            all_preds.append(output[0])
            all_segs.append(segs[0])
    return float(np.mean(all_dices)), float(np.mean(all_sdices))


def get_dice(model, ds, device, config):
    if config.debug:
        return 0.5, 0.6
    model.eval()
    dices = []
    with torch.no_grad():
        for id1, images in tqdm(ds.patches_Allimages.items(), desc='running val or test loader', position=0,
                                leave=True):
            segs = ds.patches_Allmasks[id1]
            images = Variable(torch.tensor(images)).to(device)
            _, output = model(images)
            del images
            if output.shape[1] == 2:
                output = output.cpu().data.numpy()
                output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            else:
                assert output.shape[1] == 1
                output = (nn.Sigmoid()(output) > 0.5).squeeze(1).cpu().data.numpy()
            output = _connectivity_region_analysis(output)
            dices.append(dice(segs, output))
    return float(np.mean(dices)), 0
