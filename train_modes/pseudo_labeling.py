from utils import load_model, get_batch, loss_calc
from model.unet import UNet2D
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import json
from metric_utils import get_sdice, get_dice

best_metric = -1
low_source_metric = 1.1

prev_d_score = 0


def pseudo_labels_iterations(model_path, train_loader, target_loader, val_ds, test_ds, val_ds_source, args,
                             config):
    print("pseudo labeling")
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    model = load_model(model, model_path, config.msm)
    torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_best_model.pth')
    model.eval()
    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    train_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    train_loader_iter = iter(train_loader)
    target_loader_iter = iter(target_loader)
    if config.msm:
        optimizer = optim.Adam(model.parameters(),
                               lr=1e-6, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-6)
    iterations = 50
    epochs = 50
    alpha = 0.5
    print(f"alpha is {alpha}")
    for i in tqdm(range(iterations), "iteration", position=0, leave=True):

        pseudo_labeling_after_step(model, config, i, iterations, test_ds, val_ds_source, val_ds, args)
        model_path = config.exp_dir / f'pseudo_labeling_best_model.pth'
        labels_generator = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
        labels_generator = load_model(labels_generator, model_path, config.msm)
        labels_generator.eval()
        # print_seg(None, model, config, os.getcwd(), 1, i)
        for epoch in tqdm(range(epochs), desc="epoch", position=0, leave=True):
            optimizer.zero_grad()
            source_train_batch = get_batch(train_loader, train_loader_iter)
            target_batch = get_batch(target_loader, target_loader_iter)

            source_train_images, source_train_labels, source_train_ids, source_train_slice_nums = source_train_batch
            target_images, _, target_ids, target_slice_nums = target_batch

            _, target_labels_preds = labels_generator(target_images)
            target_labels = torch.argmax(target_labels_preds, dim=1)
            _, target_preds = model(target_images.to(args.gpu))

            target_loss = loss_calc(target_preds, target_labels, gpu=args.gpu)
            _, source_preds = model(source_train_images.to(args.gpu))
            source_loss = loss_calc(source_preds, source_train_labels, gpu=args.gpu)

            total_loss = alpha * source_loss + (1 - alpha) * target_loss
            total_loss.backward()
            optimizer.step()
            del source_train_images
            del target_images
            del source_preds
            del target_preds
            del source_loss
            del target_loss
        # torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_best_model.pth')
    pseudo_labeling_after_step(model, config, iterations - 1, iterations, test_ds, val_ds_source, val_ds, args)


def get_alpha(i, iterations):
    return 1 - (0.5 + ((i + 1) / iterations) if ((i + 1) / iterations) < 0.5 else 1)


def pseudo_labeling_after_step(model, config, step_num, epochs, test_ds, val_ds_source, val_ds, args):
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
        if False:
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
        wandb.log({f'pseudo_labeling_dice/val': dice1, f'pseudo_labeling_sdice/val': sdice1, 'pseudo_labels/improvement': improvement}, step=step_num)
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
