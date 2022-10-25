from torch.autograd import Variable
from dpipe.io import load
from dataset.cc359_dataset import CC359Ds
from dataset.msm_dataset import MultiSiteMri
from utils import adjust_learning_rate, loss_calc

def pretrain(model, optimizer, scheduler, trainloader, config, args):
    if config.msm:
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                              test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/test_ids.json'), yield_id=True,
                               test=True)
    else:
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                         yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/test_ids.json'), site=args.source,
                          yield_id=True, slicing_interval=1)
    trainloader_iter = iter(trainloader)
    for i_iter in range(config.num_steps):
        model.train()
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, config, args)

        # train with source
        try:
            batch = trainloader_iter.next()
        except StopIteration:
            trainloader_iter = iter(trainloader)
            batch = trainloader_iter.next()

        images, labels = batch
        images = Variable(images).to(args.gpu)

        _, pred = model(images)
        loss_seg = loss_calc(pred, labels, args.gpu)
        loss = loss_seg
        # proper normalization

        loss.backward()
        loss_seg_value += loss_seg.data.cpu().numpy()

        optimizer.step()
        scheduler.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}'.format(
                i_iter, config.num_steps, loss_seg_value))
        after_step(i_iter, model=model, val_ds=val_ds, test_ds=test_ds, val_ds_source=None, config=config, args=args)
