from utils import utils_data
import os
import torch
from model import rlfn
from math import log10
import time
from options import args as _args
import csv


def train(model, loss_f, optimizer, train_dataloader, epoch, loss_log_path, device):
    epoch_loss = 0
    for iteration, (lr, hr) in enumerate(train_dataloader):
        start_time = time.time()
        lr, hr = lr.to(device), hr.to(device)
        prediction = model(lr)
        optimizer.zero_grad()
        loss = loss_f(prediction, hr)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        logging(loss_log_path,
                type='loss',
                epoch=epoch,
                iteration=iteration,
                loss=loss.item(),
                runtime=time.time() - start_time)

        print(f"===> Epoch[{epoch}]({iteration}/{len(train_dataloader)}): Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}")
    return avg_loss


def validation(model, loss_f, eval_dataloader, device):
    sum_psnr = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            lr, hr = batch[0].to(device), batch[1].to(device)
            prediction = model(lr)
            mse = loss_f(prediction, hr)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr

    avg_psnr = sum_psnr / len(eval_dataloader)
    print(f"===> Avg. PSNR: {avg_psnr:.4f} dB")
    return avg_psnr


def checkpoint(model, optimizer, epoch, model_folder, loss, psnr):
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
        }, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def make_log_file(model_folder, type='loss', load_num=-1):
    if type not in ('loss', 'psnr'):
        print('check your param "type"')
        print(f'expected : loss or psnr, given : {type}')

    log_path = model_folder + f'{type}_log.csv'
    # 저장된 모델을 불러오지 않는 경우
    if load_num == -1:
        with open(log_path, 'w', newline='') as logfile:
            writer = csv.writer(logfile)
            header = None
            if type == 'loss':
                header = ('epoch', 'iteration', 'loss', 'time')
            elif type == 'psnr':
                header = ('epoch', 'psnr', 'time')
            writer.writerow(header)
    # 저장된 모델을 불러오는 경우
    # epoch_23 모델을 불러오는 경우 epoch_24 부터의 기록은 지우기 위한 코드
    else:
        # [:-4] : .csv 떼어내기
        tmp_log_path = log_path[:-4] + '_tmp' + '.csv'
        with open(log_path, 'r', newline='') as readfile:
            reader = csv.reader(readfile)
            with open(tmp_log_path, 'w', newline='') as writefile:
                writer = csv.writer(writefile)
                # header 옮겨적기기
                writer.writerow(next(reader))
                for row in reader:
                    # row[0]에 epoch 정보가 저장되어 있음
                    if int(row[0]) > load_num:
                        break
                    writer.writerow(row)
        os.rename(tmp_log_path, log_path)
        
    return log_path


def logging(file_path, type='loss', **kwargs):
    with open(file_path, 'a', newline='') as logfile:
        writer = csv.writer(logfile)
        row = None

        epoch = kwargs['epoch']
        runtime = kwargs['runtime']
        if type == 'loss':
            iteration = kwargs['iteration']
            loss = kwargs['loss']
            row = (epoch, iteration, loss, runtime)
        elif type == 'psnr':
            psnr = kwargs['psnr']
            row = (epoch, psnr, runtime)
        if row is None:
            return
        writer.writerow(row)


def print_args(kwargs):
    prefix = ' ' * 7
    print(prefix + f"platform : {'Local' if kwargs.on_local else 'GCP'}")
    print(prefix + f"Random Degradation : {'ON' if kwargs.rd else 'OFF (only BICUBIC)'}")
    print(prefix + f"crop size : {kwargs.image_size}")
    print(prefix + f"aug factor : {kwargs.aug_factor}")
    print(prefix + f"upscale factor : {kwargs.upscale_factor}")
    print(prefix + f"batch size : {kwargs.batch_size}")
    print(prefix + f"epochs : {kwargs.epochs}")
    print(prefix + f"step size : {kwargs.step_size}")


def start_train(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    data_dir = os.getcwd() + '/data/' if args.on_local else '/home/shh950422/images/'

    train_dataloader = utils_data.set_dataloader(
        data_dir=data_dir,
        image_size=args.image_size,
        upscale_factor=args.upscale_factor,
        aug_factor=args.aug_factor,
        batch_size=args.batch_size,
        datatype='train',
        random_degradation=False if not args.rd else True)
    eval_dataloader = utils_data.set_dataloader(
        data_dir=data_dir,
        image_size=args.image_size,
        upscale_factor=args.upscale_factor,
        aug_factor=1,
        batch_size=args.batch_size,
        datatype='valid')

    print('===> Building model')
    model = rlfn.RLFN(upscale=args.upscale_factor).to(device)

    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if args.load_num != -1:
        PATH = f'model_zoo/model_x{args.upscale_factor}_/epoch_{args.load_num}.pth'
        if not os.path.isfile(PATH):
            raise FileNotFoundError(f'given num is {args.load_num}. But {PATH} does not exist')

        print('=====> Loading model')
        loaded = torch.load(PATH)
        model.load_state_dict(loaded['model_state_dict'])
        optimizer.load_state_dict(loaded['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5,
                                                last_epoch=args.load_num)

    print_args(args)

    model_folder = f"model_zoo/model_x{args.upscale_factor}_/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    loss_log_path = make_log_file(model_folder, type='loss', load_num=args.load_num)
    psnr_log_path = make_log_file(model_folder, type='psnr', load_num=args.load_num)

    total_train_time = 0
    for epoch in range(args.load_num + 1, args.epochs):
        train_time = 0
        start_time = time.time()

        loss = train(model, mae_loss, optimizer, train_dataloader, epoch, loss_log_path, device)
        psnr = 'skip'

        train_time += time.time() - start_time
        if epoch % 10 == 0:
            psnr = validation(model, mse_loss, eval_dataloader, device)
        start_time = time.time()

        scheduler.step()
        checkpoint(model, optimizer, epoch=epoch, model_folder=model_folder, loss=loss, psnr=psnr)

        train_time += time.time() - start_time
        print(f"train time : {train_time:.3f}sec")
        total_train_time += train_time

        logging(psnr_log_path,
                type='psnr',
                epoch=epoch,
                psnr=psnr,
                runtime=train_time)

    print(f"Average train time : {total_train_time / (args.epochs - args.load_num - 1):.3f}sec")


if __name__ == '__main__':
    start_train(_args)
