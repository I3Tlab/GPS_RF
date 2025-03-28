import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
import numpy as np

def plot_output_2d(designed_rf,designed_Gx, designed_Gy, name, Predict_Profile, excitation, epoch, step, log_path):

    time = np.linspace(0, 10e-6*1000 * designed_rf.size(1), designed_rf.size(1))

    designed_rf_r = designed_rf[0, :, 0].detach().squeeze().cpu().numpy()
    designed_rf_i = designed_rf[0, :, 1].detach().squeeze().cpu().numpy()

    plt.figure(figsize=(12, 16))
    plt.subplot(3, 2, 1)

    plt.plot(time, np.abs(designed_rf_r+1j*designed_rf_i), color='red', label='Designed RF amplitude')
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Amplitude [Hz]')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time, np.angle(designed_rf_r+1j*designed_rf_i, deg=True), color='red', label='Designed RF phase')
    plt.yticks(np.arange(-180, 181, 60))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Phase [deg]')
    plt.legend()

    designed_Gx = designed_Gx.detach().squeeze().cpu().numpy()
    designed_Gy = designed_Gy.detach().squeeze().cpu().numpy()

    plt.subplot(3, 2, 3)
    plt.plot(time, designed_Gx, color='red', label='Designed Gx')
    plt.title('Gx')
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Gradient [Hz/cm]')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time, designed_Gy, color='red', label='Designed Gy')
    plt.title('Gy')
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Gradient [Hz/cm]')
    plt.legend()

    Predict_Profile_x = Predict_Profile[:, :, :, 0].squeeze()
    Predict_Profile_y = Predict_Profile[:, :, :, 1].squeeze()
    Predict_Profile_z = Predict_Profile[:, :, :, 2].squeeze()

    excitation_x = excitation[:, :, :, 0].squeeze()
    excitation_y = excitation[:, :, :, 1].squeeze()
    excitation_z = excitation[:, :, :, 2].squeeze()

    height_cm = 15
    width_cm = 15

    plt.subplot(3, 2, 5)
    plt.imshow(torch.sqrt(Predict_Profile_x**2+Predict_Profile_y**2)/torch.max(torch.sqrt(Predict_Profile_x**2+Predict_Profile_y**2)), cmap='gray', interpolation='nearest', vmin=0, vmax=1)#, extent=[0, width_cm, 0, height_cm])
    plt.title('GPS profile')
    plt.xlabel('cm')
    plt.ylabel('cm')
    plt.colorbar()

    plt.subplot(3, 2, 6)
    plt.imshow(torch.sqrt(excitation_x**2+excitation_y**2)/torch.max(torch.sqrt(excitation_x**2+excitation_y**2)), cmap='gray', interpolation='nearest', vmin=0, vmax=1)#, extent=[0, width_cm, 0, height_cm])
    plt.title('Target profile')
    plt.xlabel('cm')
    plt.ylabel('cm')
    plt.colorbar()
    
    plt.tight_layout()

    fig_save_path = f'{log_path}/{epoch}_{name}_{step}_{F.mse_loss(Predict_Profile, excitation).mean().detach().item()}.png'

    plt.savefig(fig_save_path, format='png')
    print(f'result saved in {fig_save_path}')

    plt.close()

    mat_save_path = fig_save_path.replace('png','mat')
    save_dict = {
        'Designed_RFr': designed_rf_r,
        'Designed_RFi': designed_rf_i,
        'Designed_Gx': designed_Gx,
        'Designed_Gy': designed_Gy,
        'Target_profile': excitation.detach().cpu().squeeze().numpy(),
        'Predict_profile': Predict_Profile.detach().cpu().squeeze().numpy()
    }
    scipy.io.savemat(mat_save_path, save_dict)
