import argparse
import os
import scipy
import cv2
import pandas as pd
import torch
import tqdm
import yaml
from collections import OrderedDict
from accelerate import Accelerator
import types
import numpy as np
from datetime import datetime
import torch.nn.functional as F

#local library
from utils.plot_output import plot_output_2d
from models.SpatialTimeConvertor import spatial_time_convertor_real, spatial_time_convertor_imag
from utils.bloch_v_t2_Gxy import Gxyt2fBlochsim_fast_jit

def clip_complex_tensor(real_part, imag_part, max_magnitude):

    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

    if torch.any(magnitude > max_magnitude):
        clipped_magnitude = torch.clamp(magnitude, max=max_magnitude)

        scale = clipped_magnitude / magnitude
        real_part *= scale.detach()
        imag_part *= scale.detach()

        return real_part, imag_part
    else:
        return real_part, imag_part

def train_single_epoch(args,
                     spatial_time_convertor_real,
                     spatial_time_convertor_imag,
                     optimizer_real,
                     optimizer_imag,
                     epoch,
                     log_path
                     ):

    spatial_time_convertor_real.train()
    spatial_time_convertor_imag.train()

    train_losses = 0.

    best_loss = 1e10

    #load pre-designed gradients
    data = scipy.io.loadmat(args.gradient_path)
    if args.chop_gradient:
        Gx = -data['Gx'][120:, :]
        Gy = data['Gy'][120:, :]
    else:
        Gx = -data['Gx']
        Gy = data['Gy']
    designed_Gx = torch.tensor(Gx[:,0]).to(torch.float32).to(device).unsqueeze(0)
    designed_Gy = torch.tensor(Gy[:,0]).to(torch.float32).to(device).unsqueeze(0)

    duration = designed_Gx.size(1)

    #load image as target profile
    excitation = cv2.imread(args.profile_path, 0)
    excitation = cv2.resize(excitation, (args.nvox, args.nvox), interpolation=cv2.INTER_NEAREST)
    excitation_y = torch.tensor(excitation).to(torch.float32).unsqueeze(-1)
    excitation_y = excitation_y / excitation_y.max()
    excited_value = np.sin(args.FA * np.pi / 180)
    excitation_y = excitation_y * excited_value
    excitation_x = torch.zeros_like(excitation_y)
    excitation_z = torch.sqrt(1 - excitation_x ** 2 - excitation_y ** 2)
    excitation = torch.cat([excitation_x, excitation_y, excitation_z], dim=-1).to(device).unsqueeze(0)

    #define 2D bloch simulator
    bloch = Gxyt2fBlochsim_fast_jit(duration=duration, nvox=int(excitation.size(1)), nvoy=int(excitation.size(2)), dt=10e-6, t2=[0], FOV=args.FOV)

    imaging_flag = 0
    stop_flag = 0
    for mini_batch in tqdm.tqdm(range(500)):
        if stop_flag > 50:
            return 0

        input_real = spatial_time_convertor_real(excitation.reshape([1,-1]))[:,:duration]
        input_imag = spatial_time_convertor_imag(excitation.reshape([1,-1]))[:,:duration]

        ##clip by b1max
        if args.b1_max > 0:
            input_real, input_imag = clip_complex_tensor(input_real, input_imag, args.b1_max)

        rf_t = torch.cat([(input_real).unsqueeze(-1), (input_imag).unsqueeze(-1)], dim=-1)

        Predict_Profile = bloch(rf_t[:, :, 0], rf_t[:, :, 1], designed_Gx.detach(), designed_Gy.detach())

        loss = F.mse_loss(Predict_Profile, excitation.detach())

        cos_sim = F.cosine_similarity(Predict_Profile.detach().reshape([1, 3, -1]),
                                      excitation.detach().reshape([1, 3, -1], 1)).mean().item()

        optimizer_real.zero_grad()
        optimizer_imag.zero_grad()
        loss.backward()
        optimizer_real.step()
        optimizer_imag.step()

        train_losses += loss.detach().item()

        print(
            f'Epoch: {epoch} | Step: {mini_batch} | Loss: {loss.detach().item()} | Stop Flag {stop_flag} | Imaging Flag {imaging_flag} | Best Loss {best_loss}')

        log['time'].append(datetime.now())
        log['epoch'].append(epoch)
        log['batch'].append(mini_batch)
        log['training_loss'].append(train_losses/(mini_batch+1))
        log['cos_sim'].append(cos_sim)
        pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)

        plot_output_2d(
            designed_rf=rf_t.detach().cpu(),
            designed_Gx=designed_Gx.detach().cpu(),
            designed_Gy=designed_Gy.detach().cpu(),
            name='',
            Predict_Profile=Predict_Profile.detach().cpu(),
            excitation=excitation.detach().cpu(),
            epoch=epoch,
            step=mini_batch,
            log_path=log_path
        )
        imaging_flag = 0

        if loss < best_loss:
            imaging_flag +=1
            stop_flag = 0
            switch_sing = 0

            best_loss = loss.detach().item()
            torch.save(spatial_time_convertor_real.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_real.pth'))  # + '_batch_%d.pth' %batch_idx))
            torch.save(spatial_time_convertor_imag.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_imag.pth'))  # + '_batch_%d.pth' %batch_idx))
        else:
            stop_flag += 1
            switch_sing += 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D RF Pulse Design')
    parser.add_argument('--converter_ckp', default=None, help='checkpoint directory')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--nvox', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--b1_max', type=int, default=700)
    parser.add_argument('--FA', type=int, default=15)
    parser.add_argument('--chop_gradient', type=bool, default=True)
    parser.add_argument('--FOV', type=int, default=16, help='unit in cm')
    parser.add_argument('--gradient_path', type=str, default='data_loader/2D_profile_gradient_with_k_gm2d100_sm25_resampled_1.2315.mat')
    parser.add_argument('--profile_path', type=str, default='AI64_con.png')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    args.taskname = f'2DPulse_training_lr{args.lr}_b1max{args.b1_max}_FA{args.FA}_{datetime.now().strftime("%Y%m%d%H%M")}_{args.notes}'

    # parameter setting
    if args.converter_ckp is not None:
        with open(os.path.join(args.converter_ckp, 'config.yml'), 'r') as f:
            training_args = yaml.safe_load(f)
            training_args = types.SimpleNamespace(**training_args)
            print(args)

    # save setting file
    log_path = ''

    # make log dir
    log_path = 'test' + '_log/' + args.taskname
    os.makedirs(log_path, exist_ok=True)

    # save setting file
    with open(os.path.join(log_path,'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)
    print(args)

    accelerator = Accelerator()
    device = accelerator.device

    #define networks
    spatial_time_convertor_real = spatial_time_convertor_real(args.nvox * args.nvox * 3, 2048,dropout=args.dropout_rate).to(device)
    spatial_time_convertor_imag = spatial_time_convertor_imag(args.nvox * args.nvox * 3, 2048,dropout=args.dropout_rate).to(device)
    optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.parameters(), lr=args.lr)
    optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.parameters(), lr=args.lr)

    if args.converter_ckp is not None:
        spatial_time_convertor_real.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_epoch_real.pth' ))
        spatial_time_convertor_imag.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_epoch_imag.pth'))

    log = OrderedDict([
        ('time', []),
        ('epoch', []),
        ('batch', []),
        ('training_loss', []),
        ('cos_sim', [])
    ])

    #Main loop
    for epoch in range(args.max_epochs):
        epoch_avg_Bloch_loss_val = train_single_epoch(args=args,
                                                    spatial_time_convertor_real=spatial_time_convertor_real,
                                                    spatial_time_convertor_imag=spatial_time_convertor_imag,
                                                    optimizer_real=optimizer_real,
                                                    optimizer_imag=optimizer_imag,
                                                    epoch=epoch,
                                                    log_path=log_path
                                                    )

    print('End!')
    torch.cuda.empty_cache()
