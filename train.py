import logging
import time
from pathlib import Path

import torch

from torch import optim, nn

from data.get_data import get_data
from model.get_model import get_model
from sparsity.weight_sparsity import weight_sparsity, register_hook
from util import process, constant
from util.checkpoint import save_checkpoint, load_checkpoint
from util.config import get_config, init_logger
from util.monitor import TensorBoardMonitor, AverageMeter, ProgressMonitor

script_dir = Path.cwd()
args = get_config(default_file=script_dir / 'config.yaml')

output_dir = script_dir / args.output_dir
output_dir.mkdir(exist_ok=True)
log_dir = init_logger(args.name, output_dir, script_dir / 'logging.conf')
logger = logging.getLogger()

if args.device.type == 'cpu' or not torch.cuda.is_available() or args.device.gpu == []:
    args.device.gpu = []
else:
    available_gpu = torch.cuda.device_count()
    for dev_id in args.device.gpu:
        if dev_id >= available_gpu:
            logger.error('GPU device ID {0} requested, but only {1} devices available'
                         .format(dev_id, available_gpu))
            exit(1)
    # Set default device in case the first one on the list
    torch.cuda.set_device(args.device.gpu[0])
    print(torch.cuda.current_device())
    # Enable the cudnn built-in auto-tuner to accelerating training, but it
    # will introduce some fluctuations in a narrow range.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
'''constant'''
constant.args = args
constant.activation_sparsity_num = args.sparsity.activation.layer
constant.smooth = args.sparsity.smooth
constant.use_all_sparsity = args.sparsity.use_all_sparsity
constant.mode = args.sparsity.mode
if args.sparsity.use_all_sparsity is False and args.sparsity.mode != 'none':
    constant.flag = True
else:
    constant.flag = False

trainset, trainloader, testset, testloader = get_data(batch_size=args.dataloader.batch_size,
                                                      dataset=args.dataloader.dataset)

model = get_model(model=args.model, pretrained=args.pre_trained, mode=args.sparsity.mode, smooth=args.sparsity.smooth,
                  num_classes=args.dataloader.num_classes).cuda()
if args.resume.path:
    model, start_epoch, _ = load_checkpoint(
        model, args.resume.path, args.device.type, lean=True)
    print("ok")

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss().cuda()

pymonitor = ProgressMonitor(logger)
tbmonitor = TensorBoardMonitor(logger, log_dir)
# tbmonitor.writer.add_graph(model, input_to_model=trainloader.dataset[0][0].unsqueeze(0).cuda())
monitors = [pymonitor, tbmonitor]
perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
print(len(constant.epo_num))
print(constant.epo_num)
for epoch in range(args.epoch):
    constant.epoch = epoch
    running_loss = 0.0
    t = time.perf_counter()
    print('func start')
    t_top1, t_top5, t_loss = process.train(trainloader, model, criterion, optimizer,
                                           None, epoch, monitors, args)
    v_top1, v_top5, v_loss = process.validate(testloader, model, criterion, epoch, monitors, args)

    tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
    tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1},
                                 epoch)
    tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5},
                                 epoch)
    perf_scoreboard.update(v_top1, v_top5, epoch)
    is_best = perf_scoreboard.is_best(epoch)
    save_checkpoint(epoch, args.model, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
tbmonitor.writer.close()  # close the TensorBoard
logger.info('Program completed successfully ... exiting ...')
logger.info('If you have any questions or suggestions, please visit: github.com/to be continued')
