import argparse
import os
import pandas as pd
import scipy.io
import torch
import yaml
import tqdm
from collections import OrderedDict
from accelerate import Accelerator
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#Local library
from models.SpatialTimeConvertor import spatial_time_convertor_real, spatial_time_convertor_imag
from utils.bloch_v_faster_gradient_step import fBlochsim_v_fast_adiabatic

def plot_output(designed_rf, Predict_Profile, excitation, epoch, batch_idx, b1maxs, nvox):
    frequency = np.linspace(-4096, 4096, nvox)
    time = np.linspace(0, 6.25e-5*1000 * designed_rf.size(1), designed_rf.size(1))
    designed_rf_r = designed_rf[0, :, 0].detach().squeeze().cpu().numpy()
    designed_rf_i = designed_rf[0, :, 1].detach().squeeze().cpu().numpy()
    fig_save_path = f'{log_path}/{epoch}_{batch_idx}_{F.cosine_similarity(Predict_Profile.reshape([Predict_Profile.size(0), 3, -1]),excitation.reshape([Predict_Profile.size(0), 3, -1], 1)).mean().detach().item()}.png'

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)

    plt.plot(time, np.abs(designed_rf_r+1j*designed_rf_i), color='red', label='Designed RF amplitude')
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Amplitude [a.u.]')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time, np.angle(designed_rf_r+1j*designed_rf_i, deg=True), color='red', label='Designed RF phase')
    plt.yticks(np.arange(-180, 181, 60))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Phase [deg]')
    plt.legend()
    plt.subplot(2, 1, 2)

    for c, b1max in enumerate(b1maxs[::2]):

        plt.plot(frequency, Predict_Profile[c,2*nvox:].detach().cpu().squeeze().numpy(),label=f'GPS Profile b1max={b1maxs[c]}', linestyle='--')
        plt.plot(frequency, excitation[c,2*nvox:].detach().cpu().squeeze().numpy(), label=f'Target Profile b1max={b1maxs[c]}',
                 alpha=0.5)
    plt.title('Lines: target profiles; Dashed lines: GPS profiles; at B1max [100,2000] Hz')
    plt.xlabel('[Hz]')
    plt.ylabel('|M$_{z}$| [a.u.] ')
   # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(fig_save_path, format='png')
    print(f'result saved in {fig_save_path}')
    plt.close()
    
    mat_save_path = fig_save_path.replace('png','mat')
    save_dict = {
        'Designed_RF': designed_rf_r+1j*designed_rf_i,
        'Target_profile': excitation.detach().cpu().squeeze().numpy(),
        'Predict_profile': Predict_Profile.detach().cpu().squeeze().numpy(),
        'b1maxs':b1maxs
    }
    scipy.io.savemat(mat_save_path, save_dict)
    print(f'result saved in {mat_save_path}')

def val_single_epoch(args,
                     spatial_time_convertor_real,
                     spatial_time_convertor_imag,
                     optimizer_real,
                     optimizer_imag,
                     epoch
                     ):

    spatial_time_convertor_real.train()
    spatial_time_convertor_imag.train()

    best_loss = 1e10

    nvox = args.nvox
    dt = args.dt
    bloch = fBlochsim_v_fast_adiabatic(nvox=int(nvox), dt=dt, f_factor=args.f)

    data = scipy.io.loadmat(args.exp_path)

    #define b1maxs
    b1maxs = torch.linspace(args.b1_max / 200, args.b1_max, 200).unsqueeze(1)
    n_b1 = b1maxs.size(0)
    sample_factor = 10
    start = sample_factor - 1
    b1maxs = b1maxs[start:n_b1:sample_factor, :].to(device).to(torch.float32)

    # target profiles
    mx = torch.tensor(data['mxy'].real).permute(1, 0)[start:n_b1:sample_factor, :]
    my = torch.tensor(data['mxy'].imag).permute(1, 0)[start:n_b1:sample_factor, :]
    mz = torch.tensor(data['mz']).permute(1, 0)[start:n_b1:sample_factor, :]
    excitation = torch.cat([mx, my, mz], dim=1).to(device).to(torch.float32)

    single_best_loss = 1e6
    single_stop = 0
    for mini_batch in tqdm.tqdm(range(args.max_epochs)):

        predict_rf_real = spatial_time_convertor_real(excitation)
        predict_rf_real = torch.mean(predict_rf_real,dim=0, keepdim=True)

        predict_rf_imag = spatial_time_convertor_imag(excitation)
        predict_rf_imag = torch.mean(predict_rf_imag, dim=0, keepdim=True)

        b1max_pred = torch.max(torch.absolute(predict_rf_real+1j*predict_rf_imag))

        predict_rf_real = predict_rf_real/b1max_pred
        predict_rf_imag = predict_rf_imag/b1max_pred

        norm_rf_t = torch.cat([(predict_rf_real).unsqueeze(-1), (predict_rf_imag).unsqueeze(-1)], dim=-1).detach()

        predict_rf_real = predict_rf_real * b1maxs
        predict_rf_imag = predict_rf_imag * b1maxs

        rf_t = torch.cat([(predict_rf_real).unsqueeze(-1), (predict_rf_imag).unsqueeze(-1)], dim=-1)

        # start from single profile learning
        if single_best_loss > 1e-09 and single_stop < 200:
            predict_profile = bloch(rf_t[0, :, 0].unsqueeze(0), rf_t[0, :, 1].unsqueeze(0))
            loss = F.mse_loss(predict_profile, excitation[0, :].unsqueeze(0).detach())

            optimizer_real.zero_grad()
            optimizer_imag.zero_grad()
            loss.backward()
            optimizer_real.step()
            optimizer_imag.step()

            print(f'Epoch: {epoch} | Step: {mini_batch} | single Loss: {loss.detach().item()} | single stop: {single_stop} ')

            predict_profiles = torch.zeros([b1maxs.size(0), int(nvox * 3)]).to(device)
            for i in tqdm.tqdm(range(rf_t.size(0))):
                predict_profile = bloch(rf_t[i, :, 0].unsqueeze(0).detach(), rf_t[i, :, 1].unsqueeze(0).detach())
                predict_profiles[i, :] = predict_profile.detach()

            cos_sim = F.cosine_similarity(predict_profiles.detach().reshape([predict_profiles.size(0), 3, -1]),
                                          excitation.detach().reshape([predict_profiles.size(0), 3, -1], 1)).mean().item()

            if loss < single_best_loss:
                single_best_loss = loss.detach().item()
                single_stop = 0
                plot_output(
                    designed_rf=norm_rf_t.detach().cpu(),
                    Predict_Profile=predict_profiles.detach().cpu(),
                    excitation=excitation.detach().cpu(),
                    epoch=epoch,
                    batch_idx=mini_batch,
                    b1maxs=b1maxs.detach().cpu().squeeze().numpy(),
                    nvox=nvox
                )

            else:
                single_stop += 1

            log['time'].append(datetime.now())
            log['epoch'].append(epoch)
            log['batch'].append(mini_batch)
            log['training_loss'].append(loss.detach().item())
            log['cos_sim'].append(cos_sim)

            pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)


            continue

        # move to all profiles learning
        loss = 0
        predict_profiles = torch.zeros([b1maxs.size(0), int(nvox*3)]).to(device)
        for i in tqdm.tqdm(range(rf_t.size(0))):
            predict_profile = bloch(rf_t[i, :, 0].unsqueeze(0), rf_t[i, :, 1].unsqueeze(0))
            loss += F.mse_loss(predict_profile, excitation[i,:].unsqueeze(0).detach())
            predict_profiles[i,:] = predict_profile.detach()

        loss = loss/(i+1)
        optimizer_real.zero_grad()
        optimizer_imag.zero_grad()
        loss.backward()
        optimizer_real.step()
        optimizer_imag.step()

        cos_sim = F.cosine_similarity(predict_profiles.detach().reshape([predict_profiles.size(0), 3, -1]),
                                      excitation.detach().reshape([predict_profiles.size(0), 3, -1], 1)).mean().item()

        print(f'Epoch: {epoch} | Step: {mini_batch} | Training Loss: {loss.detach().item()} ')

        if loss.detach().item() < best_loss:

            plot_output(
                designed_rf=norm_rf_t.detach().cpu(),
                Predict_Profile=predict_profiles.detach().cpu(),
                excitation=excitation.detach().cpu(),
                epoch=epoch,
                batch_idx=mini_batch,
                b1maxs=b1maxs.detach().cpu().squeeze().numpy(),
                nvox=nvox
            )

            best_loss = loss.detach().item()
            torch.save(spatial_time_convertor_real.state_dict(),
                       os.path.join('test_log', args.taskname, f'convertor_epoch_real.pth')) 
            torch.save(spatial_time_convertor_imag.state_dict(),
                       os.path.join('test_log', args.taskname, f'convertor_epoch_imag.pth'))

        log['time'].append(datetime.now())
        log['epoch'].append(epoch)
        log['batch'].append(mini_batch)
        log['training_loss'].append(loss.detach().item())
        log['cos_sim'].append(cos_sim)
        pd.DataFrame(log).to_csv(log_path + '/log.csv', index=False)

        if cos_sim == 1:
            break
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adiabatic RF Pulse Design')
    parser.add_argument('--converter_ckp', default=None, help='checkpoint directory')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--exp_path', type=str, default='data_loader/HS1_R_8_pw_8ms_128_elm_excitation_profile_2048_pm8k_conj.mat')
    parser.add_argument('--notes', default='')
    parser.add_argument('--n_points', type=int, default=128)
    parser.add_argument('--dt', type=float, default=6.25e-05)
    parser.add_argument('--b1_max', type=int, default=2000)
    parser.add_argument('--nvox', type=int, default=2048)
    parser.add_argument('--f', type=float, default=2)

    args = parser.parse_args()
    args.notes = f'{args.notes}_{args.dt}'
    args.taskname = f'adiabatic_b1max{args.b1_max}_nvox{int(args.nvox*args.f)}_f{args.f}_mloss_lr{args.lr}_n_points{args.n_points}_{datetime.now().strftime("%Y%m%d%H%M")}'

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

    #define networks
    spatial_time_convertor_real = spatial_time_convertor_real(int(args.nvox * 3), args.n_points).to(device)
    spatial_time_convertor_imag = spatial_time_convertor_imag(int(args.nvox * 3 ), args.n_points).to(device)
    optimizer_real = torch.optim.AdamW(spatial_time_convertor_real.parameters(), lr=args.lr)
    optimizer_imag = torch.optim.AdamW(spatial_time_convertor_imag.parameters(), lr=args.lr)

    # load checkpoints if available
    if args.converter_ckp is not None:
        spatial_time_convertor_real.load_state_dict(
            torch.load(args.converter_ckp + '/convertor_epoch_real.pth'))
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
        epoch_avg_Bloch_loss_val = val_single_epoch(args=args,
                                                    spatial_time_convertor_real=spatial_time_convertor_real,
                                                    spatial_time_convertor_imag=spatial_time_convertor_imag,
                                                    optimizer_real=optimizer_real,
                                                    optimizer_imag=optimizer_imag,
                                                    epoch=epoch,
                                                    )

    print('End!')
    torch.cuda.empty_cache()
