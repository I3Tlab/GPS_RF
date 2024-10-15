import argparse
import os
import scipy
import pandas as pd
import torch
import tqdm
import yaml
from collections import OrderedDict
from accelerate import Accelerator
import types
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#Local library
from utils.bloch_v_faster_gradient_step_spatial_spectral import SPSPfBlochsim_fast_batch_faster
from models.SpatialTimeConvertor import spatial_time_convertor_real, spatial_time_convertor_imag

def plot_output(designed_rf, Predict_Profile, excitation, S, x, G, epoch, batch_idx, nvox):
    time = np.linspace(0, 10e-6*1000 * designed_rf.size(1), designed_rf.size(1))

    designed_rf_r = designed_rf[0, :, 0].detach().squeeze().cpu().numpy()
    designed_rf_i = designed_rf[0, :, 1].detach().squeeze().cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(time, np.abs(designed_rf_r+1j*designed_rf_i), color='red', label='Designed RF amplitude')
    plt.yticks(np.arange(0,150, 30))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Amplitude [a.u.]')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time, np.angle(designed_rf_r+1j*designed_rf_i, deg=True), color='red', label='Designed RF phase')
    plt.yticks(np.arange(-180, 181, 60))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Phase [deg]')
    plt.legend()

    plt.subplot(2, 2, 3)
    predict_profile_abs = np.transpose(np.abs(Predict_Profile[:, :nvox] + 1j * Predict_Profile[:, nvox:2 * nvox]))
    plt.imshow(predict_profile_abs, extent=[-750, 750, -5, 5], cmap='gray', aspect='auto')
    plt.title('GPS Profile', fontsize=14)
    plt.xlabel('[Hz]')
    plt.ylabel('[cm]')

    plt.subplot(2, 2, 4)
    excitation_abs = np.transpose(np.abs(excitation[:,:nvox] + 1j*excitation[:,nvox:2*nvox]))
    plt.imshow(excitation_abs, extent=[-750, 750, -5, 5], cmap='gray', aspect='auto')
    plt.title('Target_profile', fontsize=14)
    plt.xlabel('[Hz]')
    plt.ylabel('[cm]')


    plt.tight_layout()

    fig_save_path = f'{log_path}/{epoch}_{batch_idx}_{F.cosine_similarity(Predict_Profile.reshape([Predict_Profile.size(0), 3, -1]),excitation.reshape([Predict_Profile.size(0), 3, -1], 1)).mean().detach().item()}.png'
    plt.savefig(fig_save_path, format='png')
    print(f'result saved in {fig_save_path}')
    plt.close()

    mat_save_path = fig_save_path.replace('png','mat')
    save_dict = {
        'Designed_RF': designed_rf_r + 1j * designed_rf_i,
        'Target_profile': excitation.detach().cpu().squeeze().numpy(),
        'Predict_profile': Predict_Profile.detach().cpu().squeeze().numpy(),
        'S': S.numpy(),
        'x': x.numpy(),
        'G': G.numpy()
    }
    scipy.io.savemat(mat_save_path, save_dict)
    print(f'result saved in {mat_save_path}')

def train_single_epoch(args,
                     spatial_time_convertor_real,
                     spatial_time_convertor_imag,
                     optimizer_real,
                     optimizer_imag,
                     epoch
                     ):

    spatial_time_convertor_real.train()
    spatial_time_convertor_imag.train()

    best_loss = 1e10

    data = scipy.io.loadmat(args.data_path)

    #load pre-designed gradient and target profile
    designed_Gx = torch.tensor(data['Gx']).to(torch.float32).to(device)
    S = torch.tensor(data['S']).to(torch.float32).to(device)
    x = torch.tensor(data['x']).to(torch.float32).to(device)
    mx = torch.tensor(data['mxy'].real)
    my = torch.tensor(data['mxy'].imag)
    mz = torch.tensor(data['mz'])
    excitation = torch.cat([mx,my,mz],dim=0).permute(1,0).to(torch.float32).to(device)
    duration = designed_Gx.size(1)

    #define Bloch simulator
    bloch = SPSPfBlochsim_fast_batch_faster(nvox=int(args.nvox), x=x, dt=args.dt, t2=[0])

    stop_flag = 0
    for mini_bactch in tqdm.tqdm(range(args.max_epochs)):

        if stop_flag > 50:
            break

        input_real = spatial_time_convertor_real(excitation)
        input_real = torch.mean(input_real, dim=0, keepdim=True)
        input_imag = spatial_time_convertor_imag(excitation)
        input_imag = torch.mean(input_imag, dim=0, keepdim=True)

        if args.b1_max_adjust > 0:
            magnitude = torch.sqrt(input_real ** 2 + input_imag ** 2)
            b1_max = torch.max(magnitude)
            input_real = (input_real / b1_max) * args.b1_max * args.b1_max_adjust
            input_imag = (input_imag / b1_max) * args.b1_max * args.b1_max_adjust

        rf_t = torch.cat([(input_real).unsqueeze(-1), (input_imag).unsqueeze(-1)], dim=-1)
        rf_t = rf_t[:,:duration,:]

        batch_size = args.n_freq
        Predict_Profile = bloch(rf_t[:, :, 0].repeat([batch_size,1]), rf_t[:, :, 1].repeat([batch_size,1]),
                                 designed_Gx.repeat([batch_size,1]), S.permute(1,0)[:batch_size,:])
        loss = F.mse_loss(Predict_Profile.reshape([batch_size,-1]), excitation[:batch_size,:].detach())

        cos_sim = F.cosine_similarity(Predict_Profile.detach().reshape([batch_size, 3, -1]),
                                      excitation.detach().reshape([batch_size, 3, -1], 1)).mean().item()

        optimizer_real.zero_grad()
        optimizer_imag.zero_grad()
        loss.backward()
        optimizer_real.step()
        optimizer_imag.step()

        print(
            f'Epoch: {epoch} | Step: {mini_bactch} | Loss: {loss.detach().item()} | Stop Flag {stop_flag} ')

        log['time'].append(datetime.now())
        log['epoch'].append(epoch)
        log['batch'].append(mini_bactch)
        log['training_loss'].append(loss.detach().item())
        log['cos_sim'].append(cos_sim)
        pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)

        if loss < best_loss:
            plot_output(
                designed_rf=rf_t.detach().cpu(),
                Predict_Profile=Predict_Profile.detach().cpu(),
                excitation=excitation.detach().cpu(),
                S=S.detach().cpu(),
                x=x.detach().cpu(),
                G=designed_Gx.detach().cpu(),
                epoch=epoch,
                batch_idx=mini_bactch,
                nvox=args.nvox
            )
            stop_flag = 0

            best_loss = loss.detach().item()
            torch.save(spatial_time_convertor_real.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_real.pth'))
            torch.save(spatial_time_convertor_imag.state_dict(),
                       os.path.join('test_log', args.taskname,
                                    f'convertor_epoch_imag.pth'))
            if cos_sim == 1:
                return 1
        else:
            stop_flag += 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D SPSP Designer batched')
    parser.add_argument('--converter_ckp', default=None, help='checkpoint directory')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--nvox', type=int, default=192)
    parser.add_argument('--n_freq', type=int, default=96)
    parser.add_argument('--b1_max', type=float, default=149.8239)
    parser.add_argument('--b1_max_adjust',type=float,default=1)
    parser.add_argument('--dt', type=float, default=10e-6)
    parser.add_argument('--data_path', type=str, default='data_loader/SPSP_TBW_3_SBW_6_pw_23p8ms_exc_width_5mm_water_192x96_conj.mat')
    parser.add_argument('--notes',type=str, default='')
    args = parser.parse_args()

    args.taskname = f'SPSP_batch_lr{args.lr}_b1max{args.b1_max}_b1adjust{args.b1_max_adjust}_{datetime.now().strftime("%Y%m%d%H%M")}_{args.notes}'

    # parameter setting
    if args.converter_ckp is not None:
        with open(os.path.join(args.converter_ckp, 'config.yml'), 'r') as f:
            training_args = yaml.safe_load(f)
            training_args = types.SimpleNamespace(**training_args)

            print(args)

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
    spatial_time_convertor_real = spatial_time_convertor_real(args.nvox * 3, 2560).to(device)
    spatial_time_convertor_imag = spatial_time_convertor_imag(args.nvox * 3, 2560).to(device)
    optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.parameters(), lr=args.lr)
    optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.parameters(), lr=args.lr)

    if args.converter_ckp is not None:
        spatial_time_convertor_real.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_real.pth'))
        spatial_time_convertor_imag.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_imag.pth'))

    log = OrderedDict([
        ('time', []),
        ('epoch', []),
        ('batch', []),
        ('training_loss', []),
        ('cos_sim', [])
    ])

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
