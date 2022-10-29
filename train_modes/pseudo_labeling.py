from utils.util_methods import load_model, get_batch, loss_calc
from utils.unet import UNet2D
import torch
from tqdm import tqdm
from utils.metric_utils import after_step

best_metric = -1
low_source_metric = 1.1
prev_d_score = 0


def pseudo_labels_iterations(model, optimizer, train_loader, target_loader, val_ds, test_ds, val_ds_source, args,
                             config):
    print("pseudo labeling")
    torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_best_model.pth')
    model.eval()

    train_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    train_loader_iter = iter(train_loader)
    target_loader_iter = iter(target_loader)
    iterations = args.pl_iterations
    epochs = args.pl_epochs
    alpha = get_alpha(0, iterations, args)

    for i in tqdm(range(iterations), "iteration", position=0, leave=True):
        after_step(model, config, i, iterations, test_ds, val_ds_source, val_ds, args)
        model_path = config.exp_dir / f'pseudo_labeling_best_model.pth'
        labels_generator = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
        labels_generator = load_model(labels_generator, model_path, config.msm)
        labels_generator.eval()
        get_alpha(i, iterations, args)

        for epoch in tqdm(range(epochs), desc="epoch", position=0, leave=True):
            optimizer.zero_grad()
            source_train_batch = get_batch(train_loader, train_loader_iter)
            target_batch = get_batch(target_loader, target_loader_iter)

            source_train_images, source_train_labels, _, _ = source_train_batch
            target_images, _, _, _ = target_batch

            _, target_labels_preds = labels_generator(target_images)
            target_labels = torch.argmax(target_labels_preds, dim=1)
            _, target_preds = model(target_images.to(args.gpu))

            target_loss = loss_calc(target_preds, target_labels, gpu=args.gpu)
            _, source_preds = model(source_train_images.to(args.gpu))
            source_loss = loss_calc(source_preds, source_train_labels, gpu=args.gpu)

            total_loss = alpha * source_loss + (1 - alpha) * target_loss
            total_loss.backward()
            optimizer.step()
    after_step(model, config, iterations - 1, iterations, test_ds, val_ds_source, val_ds, args)


def get_alpha(iter, iterations, args):
    if args.alpha == -1:
        curr_alpha = 1 - (0.5 + ((iter + 1) / iterations) if ((iter + 1) / iterations) < 0.5 else 1)
        print(f"alpha is increasing, current alpha is {curr_alpha}")
        return curr_alpha
    else:
        return args.alpha
