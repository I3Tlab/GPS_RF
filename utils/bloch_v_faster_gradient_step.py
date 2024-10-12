import torch
import torch.jit as jit

class fBlochsim_v_fast(jit.ScriptModule):
    def __init__(self, duration=1024, nvox=256, dt=80e-6, f=1):
        super(fBlochsim_v_fast, self).__init__()
        self.nvox = nvox
        self.duration = duration
        self.dt = dt
        self.f = f

    @jit.script_method
    def forward(self, rRF, iRF, mt):
        sM_f = torch.linspace(int(-4096*8*self.f), int(4096*8*self.f), self.nvox).cuda()
        mx = mt[:,:self.nvox].unsqueeze(-1)
        my = mt[:,self.nvox:self.nvox*2].unsqueeze(-1)
        mz = mt[:,self.nvox*2:].unsqueeze(-1)

        for i in range(rRF.size(1)):
            m = torch.cat((mx, my, mz), -1).cuda()
            m = torch.reshape(m, (rRF.size()[0], m.size()[1] * m.size()[2], 1))

            b1 = torch.einsum('b,bsc->bsc', torch.sqrt(rRF[:,i]**2+iRF[:,i]**2),
                              torch.ones(rRF.size()[0], 1, sM_f.size()[0]).cuda()).type(torch.float32)
            beff = torch.sqrt(b1 ** 2 + sM_f ** 2)
            phi = torch.einsum('b,bsc->bsc', torch.atan2(iRF[:,i],rRF[:,i]),
                               torch.ones(rRF.size(0), 1, sM_f.size()[0]).cuda()).type(torch.float32) #the angle between RF and Z
            beta = torch.atan2(sM_f, b1)
            theta = 2 * torch.pi * beff * self.dt

            col1 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col2 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col3 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()

            col1[:, 0:1, :] = torch.cos(phi)*(torch.square(torch.cos(beta))*torch.cos(phi) + torch.sin(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta))) + torch.sin(phi)*(torch.cos(theta)*torch.sin(phi) -
                                                                                                                                                                                                        torch.cos(phi)*torch.sin(beta)*torch.sin(theta))
            col1[:, 1:2, :] = torch.sin(phi)*(torch.cos(phi)*torch.cos(theta) + torch.sin(beta)*torch.sin(phi)*torch.sin(theta))\
                              - torch.cos(phi)*(torch.square(torch.cos(beta))*torch.sin(phi) - torch.sin(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)))
            col1[:, 2:, :] = torch.cos(phi)*(torch.cos(beta)*torch.sin(beta) - torch.cos(beta)*torch.sin(beta)*torch.cos(theta)) + torch.cos(beta)*torch.sin(phi)*torch.sin(theta)

            col2[:, 0:1, :] = torch.cos(phi)*(torch.cos(theta)*torch.sin(phi) - torch.cos(phi)*torch.sin(beta)*torch.sin(theta))\
                              - torch.sin(phi)*(torch.square(torch.cos(beta))*torch.cos(phi) + torch.sin(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta)))
            col2[:, 1:2, :] = torch.cos(phi)*(torch.cos(phi)*torch.cos(theta) + torch.sin(beta)*torch.sin(phi)*torch.sin(theta))\
                              + torch.sin(phi)*(torch.square(torch.cos(beta))*torch.sin(phi) - torch.sin(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)))
            col2[:, 2:3, :] =  torch.cos(beta)*torch.cos(phi)*torch.sin(theta) - torch.sin(phi)*(torch.cos(beta)*torch.sin(beta) - torch.cos(beta)*torch.sin(beta)*torch.cos(theta))

            col3[:, 0:1, :] = torch.cos(beta)*torch.cos(phi)*torch.sin(beta) - torch.cos(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta))
            col3[:, 1:2, :] = - torch.cos(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)) - torch.cos(beta)*torch.sin(beta)*torch.sin(phi)
            col3[:, 2:3, :] = torch.square(torch.sin(beta)) + torch.square(torch.cos(beta))*torch.cos(theta)

            colstacked = torch.stack([col1, col2, col3], dim=3)
            colinterleaved = torch.flatten(colstacked, start_dim=2, end_dim=3)
            aa = colinterleaved.transpose(1, 2).unsqueeze_(-1)
            bb = aa.reshape(rRF.size()[0], int(colinterleaved.size()[-1] / 3), 3, 3)
            R_ = bb.transpose(2, 3).permute(0, 2, 3, 1)

            R = torch.diag_embed(R_, 0, 1, 2)
            R = R.reshape([rRF.size()[0], self.nvox, self.nvox * 3, 3]).permute(0, 1, 3, 2).reshape(
                [rRF.size()[0], self.nvox * 3, self.nvox * 3]).transpose(2, 1)

            m = torch.sparse.mm(R.squeeze(0).to_sparse(), m.squeeze(0))
            m = m.unsqueeze(0)

            mx = m[:, 0:m.size()[1]:3]
            my = m[:, 1:m.size()[1]:3]
            mz = m[:, 2:m.size()[1]:3]

        out = torch.cat([mx, my, mz], 1).squeeze(1).squeeze(2)  # .detach()
        return out

class fBlochsim_v_fast_adiabatic(jit.ScriptModule):
    def __init__(self,nvox=None, dt=None, f_factor=None):
        super(fBlochsim_v_fast_adiabatic, self).__init__()
        self.nvox = nvox
        self.dt = dt
        self.f_factor = f_factor

    @jit.script_method
    def forward(self, rRF, iRF):
        t2 = [0.0]
        sM_f = torch.linspace(-4096*self.f_factor, 4096*self.f_factor, self.nvox).cuda()
        mx = torch.zeros(rRF.size()[0], self.nvox, len(t2)).cuda()
        my = torch.zeros(rRF.size()[0], self.nvox, len(t2)).cuda()
        mz = torch.ones(rRF.size()[0], self.nvox, len(t2)).cuda()

        m = torch.cat([mx, my, mz], -1)

        for i in range(iRF.size(1)):
            m = torch.reshape(m, (rRF.size()[0], m.size()[1] * m.size()[2], 1))
            b1 = torch.einsum('b,bsc->bsc', torch.sqrt(rRF[:,i]**2+iRF[:,i]**2), torch.ones(rRF.size()[0], 1, sM_f.size()[0]).cuda().type(torch.float32))
            beff = torch.sqrt(b1 ** 2 + sM_f ** 2)

            phi = torch.einsum('b,bsc->bsc', torch.atan2(iRF[:,i],rRF[:,i]),
                               torch.ones(iRF.size()[0], 1, sM_f.size()[0]).cuda().type(torch.float32))
            beta = torch.atan2(sM_f, b1)
            theta = 2 * torch.pi * beff * self.dt

            col1 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col2 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()
            col3 = torch.zeros(rRF.size()[0], 3, beta.size()[-1]).cuda()

            col1[:, 0:1, :] = torch.cos(phi)*(torch.square(torch.cos(beta))*torch.cos(phi) + torch.sin(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta))) + torch.sin(phi)*(torch.cos(theta)*torch.sin(phi) -
                                                                                                                                                                                                        torch.cos(phi)*torch.sin(beta)*torch.sin(theta))
            col1[:, 1:2, :] = torch.sin(phi)*(torch.cos(phi)*torch.cos(theta) + torch.sin(beta)*torch.sin(phi)*torch.sin(theta))\
                              - torch.cos(phi)*(torch.square(torch.cos(beta))*torch.sin(phi) - torch.sin(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)))
            col1[:, 2:, :] = torch.cos(phi)*(torch.cos(beta)*torch.sin(beta) - torch.cos(beta)*torch.sin(beta)*torch.cos(theta)) + torch.cos(beta)*torch.sin(phi)*torch.sin(theta)

            col2[:, 0:1, :] = torch.cos(phi)*(torch.cos(theta)*torch.sin(phi) - torch.cos(phi)*torch.sin(beta)*torch.sin(theta))\
                              - torch.sin(phi)*(torch.square(torch.cos(beta))*torch.cos(phi) + torch.sin(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta)))
            col2[:, 1:2, :] = torch.cos(phi)*(torch.cos(phi)*torch.cos(theta) + torch.sin(beta)*torch.sin(phi)*torch.sin(theta))\
                              + torch.sin(phi)*(torch.square(torch.cos(beta))*torch.sin(phi) - torch.sin(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)))
            col2[:, 2:3, :] =  torch.cos(beta)*torch.cos(phi)*torch.sin(theta) - torch.sin(phi)*(torch.cos(beta)*torch.sin(beta) - torch.cos(beta)*torch.sin(beta)*torch.cos(theta))

            col3[:, 0:1, :] = torch.cos(beta)*torch.cos(phi)*torch.sin(beta) - torch.cos(beta)*(torch.sin(phi)*torch.sin(theta) + torch.cos(phi)*torch.sin(beta)*torch.cos(theta))
            col3[:, 1:2, :] = - torch.cos(beta)*(torch.cos(phi)*torch.sin(theta) - torch.sin(beta)*torch.cos(theta)*torch.sin(phi)) - torch.cos(beta)*torch.sin(beta)*torch.sin(phi)
            col3[:, 2:3, :] = torch.square(torch.sin(beta)) + torch.square(torch.cos(beta))*torch.cos(theta)

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
