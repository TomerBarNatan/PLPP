from torch.autograd import Variable
from utils.metric_utils import after_step
from utils.util_methods import adjust_learning_rate, loss_calc

def pretrain(model, optimizer, scheduler, trainloader, val_ds, test_ds, config, args):
    trainloader_iter = iter(trainloader)
    for i_iter in range(config.num_steps):
        model.train()
        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, config, args)

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

        loss.backward()
        loss_seg_value += loss_seg.data.cpu().numpy()

        optimizer.step()
        scheduler.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}'.format(
                i_iter, config.num_steps, loss_seg_value))
        after_step(model=model, step_num=i_iter , epochs=config.num_steps ,val_ds=val_ds, test_ds=test_ds, val_ds_source=None, config=config, args=args)
