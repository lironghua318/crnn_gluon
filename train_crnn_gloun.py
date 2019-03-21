import argparse
import logging
import time,os
import mxnet as mx
from mxnet import gluon,nd
from crnn.data_gluon import ImageDataset
from crnn.fit.ctc_metrics import CtcMetrics
from crnn.config import config, default
from crnn.gluons.crnn import CRNN
from mxnet import autograd as ag
import matplotlib
matplotlib.use('Agg')
from gluoncv.utils import TrainingHistory
from mxnet.gluon.data import DataLoader
# from mxnet.gluon.data.vision import transforms

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CRNN network')
    parser.add_argument('--dataset-path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--no-shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--resume', help='continue training', action='store_true')
    # e2e
    parser.add_argument('--resume-from', help='resume training from the model', default=default.resume_from, type=str)
    parser.add_argument('--prefix', help='new model prefix', default=default.prefix, type=str)
    parser.add_argument('--begin-epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end-epoch', help='end epoch of training', default=default.epoch, type=int)
    parser.add_argument('--batch-size', help='batch-size', default=default.batch_size, type=int)

    parser.add_argument('--lr', help='base learning rate', default=default.lr, type=float)
    parser.add_argument('--end-lr', help='end learning rate', default=default.end_lr, type=int)
    parser.add_argument('--lr-decay', help='learning rate decay', default=default.lr_decay, type=int)
    parser.add_argument('--lr-decay-step', help='epochs to decay learning rate', default=default.lr_decay_step, type=int)

    # parser.add_argument('--no-ohem', help='disable online hard mining', action='store_true')
    parser.add_argument('--ctx', help='gpus tu use',default=default.ctx, type=str)
    parser.add_argument('--save-period', help='epochs to save params', default=default.save_period, type=int)
    parser.add_argument('--save-dir', help='path to save params and train plot', default=default.save_dir, type=str)
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    num_workers = args.num_workers
    if len(args.ctx) == 0:
        ctx = [mx.cpu(0)]
    else:
        ctx = [mx.gpu(int(i)) for i in args.ctx.split(',')]
    net = CRNN()
    net.hybridize()
    net.initialize(mx.init.Xavier(), ctx=ctx)
    if args.resume_from:
        net = gluon.nn.SymbolBlock.imports(args.resume_from[:-11]+'symbol.json', ['data'], args.resume_from, ctx=ctx)
        #net.load_parameters(args.resume_from, ctx=ctx)
    batch_size = args.batch_size * len(ctx)

    def test(val_data,ctx):
        metric = CtcMetrics(num_classes = config.num_classes)
        metric.reset()
        for datas,labels in val_data:
            data = gluon.utils.split_and_load(nd.array(datas), ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(nd.array(labels), ctx_list=ctx, batch_axis=0, even_split=False)
            output = [net(X) for X in data]
            metric.update(label, output)
        return metric.get()

    def loss_fn(pred,label):
        pred_ctc = nd.reshape(data=pred, shape=(-4, config.seq_length, -1, 0))
        ctc_loss = gluon.loss.CTCLoss(layout='TNC', label_layout='NT', weight=None)
        # loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=label,blank_label='first')
        ctc_loss = ctc_loss(pred_ctc,label)
        return ctc_loss

    def train(ctx,batch_size):
        #net.initialize(mx.init.Xavier(), ctx=ctx)
        train_data = DataLoader(ImageDataset(root=default.dataset_path, train=True), \
                                batch_size=batch_size,shuffle=True,num_workers=num_workers)
        val_data = DataLoader(ImageDataset(root=default.dataset_path, train=False), \
                              batch_size=batch_size, shuffle=True,num_workers=num_workers)

        # lr_epoch = [int(epoch) for epoch in args.lr_step.split(',')]
        net.collect_params().reset_ctx(ctx)
        lr = args.lr
        end_lr = args.end_lr
        lr_decay = args.lr_decay
        lr_decay_step = args.lr_decay_step
        all_step = len(train_data)
        schedule = mx.lr_scheduler.FactorScheduler(step=lr_decay_step * all_step, factor=lr_decay,
                                                   stop_factor_lr=end_lr)
        adam_optimizer = mx.optimizer.Adam(learning_rate=lr, lr_scheduler=schedule)
        trainer = gluon.Trainer(net.collect_params(), optimizer=adam_optimizer)

        train_metric = CtcMetrics()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        best_val_score = 0

        save_period = args.save_period
        save_dir= args.save_dir
        model_name = args.prefix
        plot_path = args.save_dir
        epochs = args.end_epoch
        frequent = args.frequent
        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            train_loss = 0
            num_batch = 0
            tic_b = time.time()
            for datas,labels in train_data:
                data = gluon.utils.split_and_load(nd.array(datas), ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(nd.array(labels), ctx_list=ctx, batch_axis=0, even_split=False)
                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1
                num_batch += 1
                if num_batch % frequent == 0:
                    train_loss_b =train_loss/(batch_size * num_batch)
                    logging.info('[Epoch %d] [num_bath %d] tain_acc=%f loss=%f time/batch: %f' %
                                 (epoch, num_batch, acc,train_loss_b, (time.time()-tic_b)/num_batch))
            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(val_data,ctx)
            train_history.update([1 - acc, 1 - val_acc])
            train_history.plot(save_path='%s/%s_history.png' % (plot_path, model_name))
            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-crnn-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                         (epoch, acc, val_acc, train_loss, time.time() - tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                symbol_file = os.path.join(save_dir,model_name)
                net.export(path=symbol_file, epoch=epoch)
                # net.save_parameters('%s/crnn-%s-%d.params' % (save_dir, model_name, epoch))

        if save_period and save_dir:
            symbol_file = os.path.join(save_dir,model_name)
            net.export(path=symbol_file, epoch=epoch-1)
            # net.save_parameters('%s/crnn-%s-%d.params' % (save_dir, model_name, epochs - 1))

    # if config.mode == 'hybrid':
    net.hybridize()

    train(ctx,batch_size)

if __name__ == '__main__':
    main()

