import argparse
import os
import pandas as pd
import scipy.io
import torch
import yaml
import tqdm
from collections import OrderedDict
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch.nn.functional as F

#Local library
from models.SpatialTimeConvertor import spatial_time_convertor_real, spatial_time_convertor_imag
from utils.bloch_v_faster_gradient_step import fBlochsim_v_fast

def plot_output(designed_rf,Predict_Profile, excitation, epoch, batch_idx, nvox):
    frequency = np.linspace(-4096*8, 4096*8, nvox)
    time = np.linspace(0,0.01*designed_rf.size(1), designed_rf.size(1))
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    designed_rf_r = designed_rf[0, :, 0].detach().squeeze().cpu().numpy()
    designed_rf_i = designed_rf[0, :, 1].detach().squeeze().cpu().numpy()
    plt.plot(time, np.abs(designed_rf_r+1j*designed_rf_i), color='red', label='Designed RF amplitude')
    plt.yticks(np.arange(0, 710, 100))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Amplitude [Hz]')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time, np.angle(designed_rf_r+1j*designed_rf_i, deg=True), color='red', label='Designed RF phase')
    plt.yticks(np.arange(-181, 180, 60))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Phase [deg]')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(frequency, np.abs(Predict_Profile.detach().cpu().squeeze().numpy()[:nvox] + 1j*Predict_Profile.detach().cpu().squeeze().numpy()[nvox:2*nvox]), color='red',
             label='GPS Profile', linestyle='--')
    plt.plot(frequency, np.abs(excitation.detach().cpu().squeeze().numpy()[:nvox] + 1j*excitation.detach().cpu().squeeze().numpy()[nvox:2*nvox]), color='blue', label='Target Profile',
             alpha=0.5)
    plt.xlabel('[Hz]')
    plt.ylabel('|M$_{xy}$| [a.u.] ')
    plt.legend()

    plt.tight_layout()

    fig_save_path = f'{log_path}/{epoch}_{batch_idx}_{F.cosine_similarity(Predict_Profile.reshape([1, 3, -1]),excitation.reshape([1, 3, -1], 1)).mean().detach().item()}.png'
    plt.savefig(fig_save_path, format='png')
    print(f'result saved in {fig_save_path}')
    plt.close()

    mat_save_path = fig_save_path.replace('png','mat')
    save_dict = {
        'Designed_RF': designed_rf_r+1j*designed_rf_i,
        'Target_profile': excitation.detach().cpu().squeeze().numpy(),
        'Predict_profile': Predict_Profile.detach().cpu().squeeze().numpy()
    }
    scipy.io.savemat(mat_save_path, save_dict)

def train_single_epoch(args,
                     spatial_time_convertor_real,
                     spatial_time_convertor_imag,
                     optimizer_real,
                     optimizer_imag,
                     epoch
                     ):

    #turn on training mode
    spatial_time_convertor_real.train()
    spatial_time_convertor_imag.train()
    train_losses = 0.

    #define the bloch simulator
    n_points = args.n_points
    nvox = args.nvox
    dt = args.dt
    bloch = fBlochsim_v_fast(duration=n_points, nvox=nvox, dt=dt, f=args.f)

    #load the target profile
    excitation = scipy.io.loadmat(args.exp_path)
    mx = torch.tensor(excitation['mxy'].real).permute(1,0)
    my = torch.tensor(excitation['mxy'].imag).permute(1,0)
    mz = torch.tensor(excitation['mz']).permute(1,0)
    excitation = torch.cat([mx,my,mz], dim=1).to(device).to(torch.float32)

    #initial magnetization
    Mx = torch.zeros(int(nvox))
    Mz = torch.ones(int(nvox))
    My = torch.sqrt(1 - Mx ** 2 - Mz ** 2)
    M_0 = torch.cat([Mx, My, Mz], dim=0).unsqueeze(0).to(device)

    stop_flag = 0
    best_loss = 1e10
    for mini_batch in tqdm.tqdm(range(args.max_epochs)):

        if stop_flag > 20:
            break
        
        # RF prediction
        predict_rf_real = spatial_time_convertor_real(excitation)
        predict_rf_imag = spatial_time_convertor_imag(excitation)

        # rescale to b1_max
        if args.b1_max > 0:
            magnitude = torch.sqrt(predict_rf_real ** 2 + predict_rf_imag ** 2)
            b1_max = torch.max(magnitude)
            predict_rf_real = (predict_rf_real / b1_max) * args.b1_max
            predict_rf_imag = (predict_rf_imag / b1_max) * args.b1_max

        #Bloch simulation
        rf_t = torch.cat([(predict_rf_real).unsqueeze(-1), (predict_rf_imag).unsqueeze(-1)], dim=-1)
        rf_t = rf_t[:,:args.n_points,:]
        predict_profile = bloch(rf_t[:, :, 0], rf_t[:, :, 1], M_0)
        loss = F.mse_loss(predict_profile, excitation.detach())
        cos_sim = F.cosine_similarity(predict_profile.detach().reshape([1, 3, -1]),excitation.detach().reshape([1, 3, -1], 1)).mean().item()

        #backpropogation and weights updating
        optimizer_real.zero_grad()
        optimizer_imag.zero_grad()
        loss.backward()
        optimizer_real.step()
        optimizer_imag.step()

        train_losses += loss.detach().item()

        imaging_flag = 0

        designed_rf = rf_t.detach()
        Predict_Profile = bloch(designed_rf[:, :, 0], designed_rf[:, :, 1], M_0.detach())

        plot_output(
            designed_rf=designed_rf,
            Predict_Profile=Predict_Profile,
            excitation=excitation,
            epoch=epoch,
            batch_idx=mini_batch,
            nvox=int(nvox),
        )

        torch.save(spatial_time_convertor_real.state_dict(),
                   os.path.join('test_log', args.taskname,
                                f'convertor_epoch_real.pth'))
        torch.save(spatial_time_convertor_imag.state_dict(),
                   os.path.join('test_log', args.taskname,
                                f'convertor_epoch_imag.pth'))

        if train_losses / (mini_batch + 1) < best_loss:
            imaging_flag += 1
            stop_flag = 0

            best_loss = train_losses / (mini_batch + 1)

        else:
            stop_flag += 1

        print(f'Epoch: {epoch} | Step: {mini_batch} | Actor Loss: {train_losses/(mini_batch+1)} | stop flag {stop_flag} ')
        log['time'].append(datetime.now())
        log['epoch'].append(epoch)
        log['batch'].append(mini_batch)
        log['actor_loss'].append(train_losses/(mini_batch+1))
        log['cos_sim'].append(cos_sim)
        pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)

        if cos_sim == 1:
            torch.save(spatial_time_convertor_real.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_real_last.pth'))
            torch.save(spatial_time_convertor_imag.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_imag_last.pth'))
            return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D selective RF Pulse Design')
    parser.add_argument('--converter_ckp', default=None, help='checkpoint directory')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--exp_path', default='data_loader/exc_tbw_6p6_pw_2p56ms_d1_0p015_d2_0p015_SLR_ex_ls_excitation_profile_2048_conj.mat')
    parser.add_argument('--notes', default='')
    parser.add_argument('--n_points', type=int, default=256)
    parser.add_argument('--dt', type=float, default=10e-6)
    parser.add_argument('--nvox', type=int, default=2048)
    parser.add_argument('--b1_max', type=int, default=0)
    parser.add_argument('--f', type=float, default=1)

    args = parser.parse_args()

    args.taskname = f'1D_b1max{args.b1_max}_nvox{args.nvox}_lr{args.lr}_n_points{args.n_points}_dt{args.dt}_{datetime.now().strftime("%Y%m%d%H%M")}'

    if args.notes is not None:
        args.taskname += args.notes

    print(args)

    # make log dir
    log_path = 'test' + '_log/' + args.taskname
    os.makedirs(log_path, exist_ok=True)

    # save setting file
    with open(os.path.join(log_path,'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)

    accelerator = Accelerator()
    device = accelerator.device

    #define the networks
    spatial_time_convertor_real = spatial_time_convertor_real(int(args.nvox * 3), args.n_points).to(device)
    spatial_time_convertor_imag = spatial_time_convertor_imag(int(args.nvox * 3), args.n_points).to(device)

    #define optimizer
    optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.parameters(), lr=args.lr)
    optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.parameters(), lr=args.lr)
    
    #load checkpoints if available
    if args.converter_ckp is not None:
        spatial_time_convertor_real.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_epoch_real.pth'))
        spatial_time_convertor_imag.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_epoch_imag.pth'))
    
    #define logger
    log = OrderedDict([
        ('time', []),
        ('epoch', []),
        ('batch', []),
        ('actor_loss', []),
        ('cos_sim',[])
    ])

    #main loop
    for epoch in range(args.max_epochs):
        epoch_avg_Bloch_loss_val = train_single_epoch(args=args,
                                                    spatial_time_convertor_real=spatial_time_convertor_real,
                                                    spatial_time_convertor_imag=spatial_time_convertor_imag,
                                                    optimizer_real=optimizer_real,
                                                    optimizer_imag=optimizer_imag,
                                                    epoch=epoch,
                                                    )
        if epoch_avg_Bloch_loss_val == 1:
            break


    print('End!')
    torch.cuda.empty_cache()
