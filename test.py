import csv
import os
import time

import torch
import cv2

from utils import utils_data
from utils.utils_image import tensor2np, calculate_psnr, calculate_ssim
from options import args
from model import rlfn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    data_dir = os.getcwd() + '/data/' if args.on_local else '/home/shh950422/images/'

    test_dataloader = utils_data.set_dataloader(data_dir=data_dir, upscale_factor=args.upscale_factor, datatype='test')

    model_dir = f'/model_test/x{args.upscale_factor}/'

    model_names = [model for model in os.listdir(os.getcwd() + model_dir) if model.endswith('.pth')]
    if len(model_names) == 0:
        raise FileNotFoundError(f"There's no model file in {model_dir}")

    model = rlfn.RLFN(upscale=args.upscale_factor).to(device)

    model_state_dict = torch.load('.' + model_dir + model_names[0], map_location=device)['model_state_dict']
    model.load_state_dict(model_state_dict)

    log_file = f'test_x{args.upscale_factor}'
    with open(log_file, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(('index', 'filename', 'height(px)', 'width(px)', 'runtime(ms)', 'psnr(dB)', 'ssim'))

    filenames = list(map(lambda name: name.split('/')[-1], test_dataloader.dataset.filenames))
    total_runtime = 0
    for idx, (lr, hr) in enumerate(test_dataloader):
        hr = hr.to(device)

        start_time = time.time()
        lr = lr.to(device)
        sr = model(lr)
        runtime = 1000 * (time.time() - start_time)
        total_runtime += runtime

        lr_img = tensor2np(lr[0])
        sr_img = tensor2np(sr[0])
        hr_img = tensor2np(hr[0])
        if idx < args.display_num:
            lr_img_bgr = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
            sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            hr_img_bgr = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)

            lr_sr_hr = cv2.hconcat([
                cv2.resize(lr_img_bgr, dsize=(0, 0), fx=args.upscale_factor, fy=args.upscale_factor),
                sr_img_bgr,
                hr_img_bgr])

            cv2.imshow('LR / SR / HR', lr_sr_hr)
            cv2.waitKey(0)

        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YCrCb)[:, :, 0] * 255
        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YCrCb)[:, :, 0] * 255

        psnr = calculate_psnr(sr_y, hr_y)
        ssim = calculate_ssim(sr_y, hr_y)
        # lr.shape[2] : height, lr.shape[3] : width
        print(f'{idx:>5d}, {runtime:>12f} ms, {str(lr.shape[2]) + "x" + str(lr.shape[3]):>12s}, {psnr:>13f} dB, {ssim:>13f}')
        with open(log_file, 'a', newline='') as logfile:
            writer = csv.writer(logfile)
            writer.writerow((idx, filenames[idx], lr.shape[2], lr.shape[3], runtime, psnr, ssim))

    print(f'Avg runtime : {total_runtime / len(test_dataloader)} ms')


if __name__ == '__main__':
    main()
