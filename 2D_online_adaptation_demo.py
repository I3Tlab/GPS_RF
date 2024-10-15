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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora

#Local library
from utils.plot_output import plot_output_2d
from models.SpatialTimeConvertor import spatial_time_convertor_real, spatial_time_convertor_imag
from utils.bloch_v_t2_Gxy import Gxyt2fBlochsim_fast_B0_jit

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

    gradient = scipy.io.loadmat(args.gradient_path)


    Gx = -gradient['Gx'][120:,:]#*1.2
    Gy = gradient['Gy'][120:,:]#*1.2

    designed_Gx = torch.tensor(Gx[:, 0]).to(torch.float32).to(device).unsqueeze(0)
    designed_Gy = torch.tensor(Gy[:, 0]).to(torch.float32).to(device).unsqueeze(0)

    duration = designed_Gx.size(1)

    dt = 10e-6

    B_0 = torch.zeros([1, args.nvox, args.nvox]).to(device)
    B_1 = torch.ones([1,args.nvox*args.nvox]).to(device)

    if args.B_0_path is not None:
        B_0 = scipy.io.loadmat(args.B_0_path)
        B_0 = B_0['B0'][:,:,args.slice - 1]
        B_0 = cv2.resize(B_0, (args.nvox, args.nvox), interpolation=cv2.INTER_NEAREST)
        B_0 = torch.tensor(B_0).unsqueeze(0).to(device)
    if args.B_1_path is not None:
        B_1 = scipy.io.loadmat(args.B_1_path)
        B_1 = B_1['B1'][:,:,args.slice - 1]
        B_1 = cv2.resize(B_1, (args.nvox, args.nvox), interpolation=cv2.INTER_NEAREST)
        B_1 = torch.tensor(B_1).unsqueeze(0).to(device)

    bloch = Gxyt2fBlochsim_fast_B0_jit(duration=duration, nvox=int(args.nvox), nvoy=int(args.nvox), dt=dt, t2=[0], FOV=args.FOV)
    excitation = scipy.io.loadmat(f'{args.converter_ckp}/3__133_0.00011015747440978885.mat')

    excitation = cv2.imread(args.profile_path, 0)
    excitation = cv2.resize(excitation, (args.nvox, args.nvox), interpolation=cv2.INTER_NEAREST)
    excitation_y = torch.tensor(excitation).to(torch.float32).unsqueeze(-1)
    excitation_y = excitation_y / excitation_y.max()
    excited_value = np.sin(args.FA * np.pi / 180)
    excitation_y = excitation_y * excited_value
    excitation_x = torch.zeros_like(excitation_y)
    excitation_z = torch.sqrt(1 - excitation_x ** 2 - excitation_y ** 2)
    excitation = torch.cat([excitation_x, excitation_y, excitation_z], dim=-1).to(device).unsqueeze(0)

    imaging_flag = 0
    switch_sing = 0
    stop_flag = 0

    scaler = GradScaler()

    for mini_batch in tqdm.tqdm(range(30)):
        if stop_flag > 10:
            return 0

        optimizer_real.zero_grad()
        optimizer_imag.zero_grad()

        with (autocast(enabled=False)):
            input_real = spatial_time_convertor_real(
                excitation.reshape([1, -1]))
            input_imag = spatial_time_convertor_imag(
                excitation.reshape([1, -1]))
            rf_t = torch.cat([(input_real).unsqueeze(-1), (input_imag).unsqueeze(-1)], dim=-1)
            rf_t = rf_t[:, :duration, :]

            Predict_Profile = bloch(rf_t[:, :, 0], rf_t[:, :, 1], designed_Gx.detach(), designed_Gy.detach(),
                                    B_0.detach(), B_1.detach())
            loss = F.mse_loss(Predict_Profile, excitation.detach())
            scaler.scale(loss).backward()

            scaler.step(optimizer_real)
            scaler.step(optimizer_imag)
            scaler.update()
            if imaging_flag > 0:
                imaging_flag = 0
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

        train_losses += loss.detach().item()

        print(
            f'Epoch: {epoch} | Step: {mini_batch} | Loss: {loss.detach().item()} | Stop Flag {stop_flag} | best loss {best_loss}')

        log['time'].append(datetime.now())
        log['epoch'].append(epoch)
        log['batch'].append(mini_batch)
        log['actor_loss'].append(train_losses/(mini_batch+1))
        pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)

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
    parser = argparse.ArgumentParser(description='2D RF Pulse Design Online Adaptation')
    parser.add_argument('--converter_ckp', default='test_log/2DPulse_training_lr0.001_b1max700_FA15_202403280940_computer_grad_switched_xy_flip_x_jit_AI', help='checkpoint directory')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--nvox', type=int, default=64)
    parser.add_argument('--f_sampling', type=float, default=3.5)
    parser.add_argument('--R', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.707)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--b1_max', type=int, default=0)
    parser.add_argument('--FA', type=int, default=15)
    parser.add_argument('--slice', type=int, default=16)
    parser.add_argument('--gradient_path', type=str, default='data_loader/2D_profile_gradient_with_k_gm2d100_sm25_resampled_1.2315.mat')
    parser.add_argument('--notes', type=str, default=''
    parser.add_argument('--FOV', type=int, default=16, help='unit in cm')
    parser.add_argument('--profile_path', type=str, default='AI64_con.png')
    parser.add_argument('--B_0_path', type=str, default='data_loader/measured_B0_20240415_2_brain.mat')
    parser.add_argument('--B_1_path', type=str, default='data_loader/measured_B1_20240415_brain.mat')
    parser.add_argument('--Adaptation_method', type=str, default='LoRA', help='finetune or LoRA')

    args = parser.parse_args()

    # parameter setting
    if args.converter_ckp is not None:
        with open(os.path.join(args.converter_ckp, 'config.yml'), 'r') as f:
            training_args = yaml.safe_load(f)
            training_args = types.SimpleNamespace(**training_args)
            args.R = training_args.R
            args.alpha = training_args.alpha
            args.f_sampling = training_args.f_sampling
            args.nvox = training_args.nvox
            args.b1_max = training_args.b1_max
            args.FA = training_args.FA
            print(args)

    args.taskname = f'online_learning_B0_B1_adaptation_lr{args.lr}_b1max{args.b1_max}_FA{args.FA}_{datetime.now().strftime("%Y%m%d%H%M")}_{args.notes}_slice{args.slice}_{args.Adaptation_method}'

    if not args.debug:
        # make log dir
        log_path = 'test' + '_log/' + args.taskname
        os.makedirs(log_path, exist_ok=True)

        # save setting file
        with open(os.path.join(log_path,'config.yml'), 'w') as f:
            yaml.dump(vars(args), f)
    else:
        log_path = 'debug'
        os.makedirs(log_path, exist_ok=True)
    print(args)

    accelerator = Accelerator()
    device = accelerator.device

    spatial_time_convertor_real = spatial_time_convertor_real(args.nvox * args.nvox * 3, 2048)
    spatial_time_convertor_imag = spatial_time_convertor_imag(args.nvox * args.nvox * 3, 2048)

    if args.converter_ckp is not None:
        spatial_time_convertor_real.load_state_dict(torch.load(args.converter_ckp + '/convertor_epoch_real.pth'))
        spatial_time_convertor_imag.load_state_dict(torch.load(args.converter_ckp + '/convertor_epoch_imag.pth'))


    if args.Adaptation_method == 'LoRA':
        add_lora(spatial_time_convertor_real)
        add_lora(spatial_time_convertor_imag)
        spatial_time_convertor_real = spatial_time_convertor_real.to(device)
        spatial_time_convertor_imag = spatial_time_convertor_imag.to(device)
        parameters_real = [{"params": list(get_lora_params(spatial_time_convertor_real))}]
        parameters_imag = [{"params": list(get_lora_params(spatial_time_convertor_imag))}]

        optimizer_real = torch.optim.AdamW(parameters_real, lr=args.lr)
        optimizer_imag = torch.optim.AdamW(parameters_imag, lr=args.lr)

    elif args.Adaptation_method == 'finetune':

        for name, para in spatial_time_convertor_real.named_parameters():
            if 'rf_real' not in name:
                para.requires_grad = False

        for name, para in spatial_time_convertor_imag.named_parameters():
            if 'rf_imag_3' not in name:
                para.requires_grad = False

        optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.rf_real.parameters(), lr=args.lr)
        optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.rf_imag_3.parameters(), lr=args.lr)

    elif args.Adaptation_method == 'all':

        optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.parameters(), lr=args.lr)
        optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.parameters(), lr=args.lr)

    spatial_time_convertor_real = spatial_time_convertor_real.to(device)
    spatial_time_convertor_imag = spatial_time_convertor_imag.to(device)
    log = OrderedDict([
        ('time', []),
        ('epoch', []),
        ('batch', []),
        ('actor_loss', []),
    ])


    for epoch in range(args.max_epochs):
        epoch_avg_Bloch_loss_val = train_single_epoch(args=args,
                                                    spatial_time_convertor_real=spatial_time_convertor_real,
                                                    spatial_time_convertor_imag=spatial_time_convertor_imag,
                                                    optimizer_real=optimizer_real,
                                                    optimizer_imag=optimizer_imag,
                                                    epoch=epoch,
                                                    log_path=log_path
                                                    )
        if args.debug:
            break

    print('End!')
    torch.cuda.empty_cache()
