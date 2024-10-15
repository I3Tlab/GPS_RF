import torch
import time
import os
import torch.jit as jit

class SPSPfBlochsim_fast_batch_faster(jit.ScriptModule):
    def __init__(self, nvox=None, x=None, dt=None, t2=None):
        super(SPSPfBlochsim_fast_batch_faster, self).__init__()
        self.t1 = 0.0
        self.nvox = nvox
        self.dt = dt
        self.t2 = t2
        self.GAMMA = 4257.747892
        self.x = x

    @jit.script_method
    def forward(self, rRF, iRF, Gx, S):
        sM_x = self.x
        t2 = [0.0]
        mx = torch.zeros(rRF.size()[0], self.nvox, len(t2)).cuda()
        my = torch.zeros(rRF.size()[0], self.nvox, len(t2)).cuda()
        mz = torch.ones(rRF.size()[0], self.nvox, len(t2)).cuda()
        m = torch.cat([mx, my, mz], -1)

        for i in range(rRF.size(1)):
            m = torch.reshape(m, (rRF.size()[0], m.size()[1] * m.size()[2], 1))

            sM_f = (self.GAMMA*torch.einsum('b, bc->bc', Gx[:, i], sM_x) + S).unsqueeze(1)

            b1 = torch.einsum('b,bcs->bcs', torch.sqrt(rRF[:, i] ** 2 + iRF[:, i] ** 2), torch.ones(rRF.size()[0], 1, self.nvox*S.size(1)).cuda()) #[b, 1, 256]

            beff = torch.sqrt(b1 ** 2 + sM_f ** 2)

            phi = torch.einsum('b,bsc->bsc', torch.atan2(iRF[:,i],rRF[:,i]),
                               torch.ones(rRF.size()[0], 1, self.nvox).cuda())

            beta = torch.atan2(sM_f, b1)
            theta = 2 * torch.pi * beff * self.dt

            col1 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col2 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col3 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()

            col1[:, 0:1, :] = torch.cos(phi) * (torch.square(torch.cos(beta)) * torch.cos(phi) + torch.sin(beta) * (
                        torch.sin(phi) * torch.sin(theta) + torch.cos(phi) * torch.sin(beta) * torch.cos(
                    theta))) + torch.sin(phi) * (torch.cos(theta) * torch.sin(phi) -
                                                 torch.cos(phi) * torch.sin(beta) * torch.sin(theta))
            col1[:, 1:2, :] = torch.sin(phi) * (
                        torch.cos(phi) * torch.cos(theta) + torch.sin(beta) * torch.sin(phi) * torch.sin(theta)) \
                              - torch.cos(phi) * (
                                          torch.square(torch.cos(beta)) * torch.sin(phi) - torch.sin(beta) * (
                                              torch.cos(phi) * torch.sin(theta) - torch.sin(beta) * torch.cos(
                                          theta) * torch.sin(phi)))
            col1[:, 2:, :] = torch.cos(phi) * (
                        torch.cos(beta) * torch.sin(beta) - torch.cos(beta) * torch.sin(beta) * torch.cos(
                    theta)) + torch.cos(beta) * torch.sin(phi) * torch.sin(theta)

            col2[:, 0:1, :] = torch.cos(phi) * (
                        torch.cos(theta) * torch.sin(phi) - torch.cos(phi) * torch.sin(beta) * torch.sin(theta)) \
                              - torch.sin(phi) * (
                                          torch.square(torch.cos(beta)) * torch.cos(phi) + torch.sin(beta) * (
                                              torch.sin(phi) * torch.sin(theta) + torch.cos(phi) * torch.sin(
                                          beta) * torch.cos(theta)))
            col2[:, 1:2, :] = torch.cos(phi) * (
                        torch.cos(phi) * torch.cos(theta) + torch.sin(beta) * torch.sin(phi) * torch.sin(theta)) \
                              + torch.sin(phi) * (
                                          torch.square(torch.cos(beta)) * torch.sin(phi) - torch.sin(beta) * (
                                              torch.cos(phi) * torch.sin(theta) - torch.sin(beta) * torch.cos(
                                          theta) * torch.sin(phi)))
            col2[:, 2:3, :] = torch.cos(beta) * torch.cos(phi) * torch.sin(theta) - torch.sin(phi) * (
                        torch.cos(beta) * torch.sin(beta) - torch.cos(beta) * torch.sin(beta) * torch.cos(theta))

            col3[:, 0:1, :] = torch.cos(beta) * torch.cos(phi) * torch.sin(beta) - torch.cos(beta) * (
                        torch.sin(phi) * torch.sin(theta) + torch.cos(phi) * torch.sin(beta) * torch.cos(theta))
            col3[:, 1:2, :] = - torch.cos(beta) * (
                        torch.cos(phi) * torch.sin(theta) - torch.sin(beta) * torch.cos(theta) * torch.sin(
                    phi)) - torch.cos(beta) * torch.sin(beta) * torch.sin(phi)
            col3[:, 2:3, :] = torch.square(torch.sin(beta)) + torch.square(torch.cos(beta)) * torch.cos(theta)

            colstacked = torch.stack([col1, col2, col3], dim=3)
            colinterleaved = torch.flatten(colstacked, start_dim=2, end_dim=3)
            aa = colinterleaved.transpose(1, 2).unsqueeze_(-1)
            bb = aa.reshape(rRF.size()[0], int(colinterleaved.size()[-1] / 3), 3, 3)
            R_ = bb.transpose(2, 3).permute(0, 2, 3, 1)

            R = torch.diag_embed(R_, 0, 1, 2)
            R = R.reshape([rRF.size()[0], self.nvox, self.nvox * 3, 3]).permute(0, 1, 3, 2).reshape(
                [rRF.size()[0], self.nvox * 3, self.nvox * 3]).transpose(2, 1)
            m = torch.bmm(R.to_sparse(), m)

        mx = m[:, 0:m.size()[1]:3]
        my = m[:, 1:m.size()[1]:3]
        mz = m[:, 2:m.size()[1]:3]
        out = torch.cat([mx, my, mz], 1).squeeze(1).squeeze(2)
        return out
