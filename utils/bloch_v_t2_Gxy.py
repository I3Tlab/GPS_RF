import torch.jit as jit
import torch

class Gxyt2fBlochsim_fast_B0_jit(jit.ScriptModule): #online learing faster 2D simulation
    def __init__(self, duration=None, nvox=None, nvoy=None, dt=None, t2=None, FOV=None):
        super(Gxyt2fBlochsim_fast_B0_jit, self).__init__()
        self.t1 = 0.0
        self.nvox = nvox
        self.nvoy = nvoy
        self.duration = duration
        self.dt = dt
        self.t2 = t2
        self.GAMMA = 4257.747892
        self.FOV = FOV

    @jit.script_method
    def forward(self, rRF, iRF, Gx, Gy, B_0, B_1):
        sRF_dt = self.dt
        sM_x = torch.linspace(-self.FOV/2, self.FOV/2, self.nvox).cuda()
        sM_y = torch.linspace(-self.FOV/2, self.FOV/2, self.nvoy).cuda()

        mx = torch.zeros(rRF.size()[0], self.nvox, self.nvoy, len(self.t2)).cuda()
        my = torch.zeros(rRF.size()[0], self.nvox, self.nvoy, len(self.t2)).cuda()
        mz = torch.ones(rRF.size()[0], self.nvox, self.nvoy,  len(self.t2)).cuda()

        for i in range(rRF.size(-1)):
            m = torch.cat((mx.unsqueeze(-1), my.unsqueeze(-1), mz.unsqueeze(-1)), -1)
            m = m.reshape([m.size(0), m.size(1)*m.size(2)*m.size(-1),1])

            sM_f = (B_0 + self.GAMMA * (torch.einsum('b, c->bc', Gx[:, i], sM_y).unsqueeze(1) +
                                 torch.einsum('b, c->bc', Gy[:, i], sM_x).unsqueeze(-1))).reshape(sM_x.size(0)*sM_y.size(0)).unsqueeze(0)
            b1 = torch.einsum('b,bs->bs', torch.sqrt(rRF[:,i]**2+iRF[:,i]**2), B_1.reshape([rRF.size(0), self.nvox*self.nvoy]))
            beff = torch.sqrt(b1 ** 2 + sM_f ** 2)
            phi = torch.einsum('b,bs->bs', torch.atan2(iRF[:,i], rRF[:,i]), torch.ones(rRF.size(0), self.nvox*self.nvoy).cuda())
            beta = torch.atan2(sM_f, b1)
            theta = 2 * torch.pi * beff * sRF_dt

            col1 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()
            col2 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()
            col3 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()

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

            colstacked = torch.stack([col1, col2, col3], dim=-1)
            colinterleaved = torch.flatten(colstacked, start_dim=2, end_dim=3)
            aa = colinterleaved.transpose(1, 2).unsqueeze_(-1)
            bb = aa.reshape(rRF.size()[0], int(colinterleaved.size()[-1] / 3), 3, 3)
            R_ = bb.transpose(2, 3).permute(0, 2, 3, 1)
            R = torch.diag_embed(R_, 0, 1, 2)

            R = R.reshape([rRF.size()[0], self.nvox*self.nvoy, self.nvox*self.nvoy * 3, 3]).permute(0, 1, 3, 2).reshape(
                [rRF.size()[0], self.nvox*self.nvoy * 3, self.nvox*self.nvoy * 3]).transpose(2, 1)
            m = torch.sparse.mm(R.squeeze(0).to_sparse(), m.squeeze(0))
            m = m.unsqueeze(0)

            mx = m[:, 0:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)
            my = m[:, 1:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)
            mz = m[:, 2:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)

        m = torch.cat([mx, my, mz], -1)

        return m

class Gxyt2fBlochsim_fast_jit(jit.ScriptModule):
    def __init__(self, duration=None, nvox=None, nvoy=None, dt=None, t2=None, FOV=None):
        super(Gxyt2fBlochsim_fast_jit, self).__init__()
        self.t1 = 0.0
        self.nvox = nvox
        self.nvoy = nvoy
        self.duration = duration
        self.dt = dt
        self.t2 = t2
        self.GAMMA = 4257.747892
        self.FOV = FOV #cm

    @jit.script_method
    def forward(self, rRF, iRF, Gx, Gy):
        sRF_dt = self.dt
        sM_x = torch.linspace(-self.FOV/2, self.FOV/2, self.nvox).cuda()
        sM_y = torch.linspace(-self.FOV/2, self.FOV/2, self.nvoy).cuda()
        mx = torch.zeros(rRF.size()[0], self.nvox, self.nvoy, len(self.t2)).cuda()
        my = torch.zeros(rRF.size()[0], self.nvox, self.nvoy, len(self.t2)).cuda()
        mz = torch.ones(rRF.size()[0], self.nvox, self.nvoy,  len(self.t2)).cuda()
        #sRF_b1 = rRF + iRF * 1j

        for i in range(rRF.size(-1)):
            m = torch.cat((mx.unsqueeze(-1), my.unsqueeze(-1), mz.unsqueeze(-1)), -1)
            m = m.reshape([m.size(0), m.size(1)*m.size(2)*m.size(-1),1])

            sM_f = self.GAMMA * (torch.einsum('b, c->bc', Gx[:, i], sM_y).unsqueeze(1) +
                                 torch.einsum('b, c->bc', Gy[:, i], sM_x).unsqueeze(-1)).reshape(sM_x.size(0)*sM_y.size(0)).unsqueeze(0)

            b1 = torch.einsum('b,bs->bs', torch.sqrt(rRF[:,i]**2+iRF[:,i]**2),
                              torch.ones(rRF.size(0), self.nvox*self.nvoy).cuda())
            beff = torch.sqrt(b1 ** 2 + sM_f ** 2)

            phi = torch.einsum('b,bs->bs', torch.atan2(iRF[:,i],rRF[:,i]), torch.ones(rRF.size(0), self.nvox*self.nvoy).cuda()) # angle between RF and Z
            beta = torch.atan2(sM_f, b1) # angle between M and G
            theta = 2 * torch.pi * beff * sRF_dt # rotation angle
            col1 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()
            col2 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()
            col3 = torch.zeros(rRF.size()[0], 3, self.nvox*self.nvoy).cuda()

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

            colstacked = torch.stack([col1, col2, col3], dim=-1)
            colinterleaved = torch.flatten(colstacked, start_dim=2, end_dim=3)
            aa = colinterleaved.transpose(1, 2).unsqueeze_(-1)
            bb = aa.reshape(rRF.size()[0], int(colinterleaved.size()[-1] / 3), 3, 3)
            R_ = bb.transpose(2, 3).permute(0, 2, 3, 1)
            R = torch.diag_embed(R_, 0, 1, 2)

            R = R.reshape([rRF.size()[0], self.nvox*self.nvoy, self.nvox*self.nvoy * 3, 3]).permute(0, 1, 3, 2).reshape(
                [rRF.size()[0], self.nvox*self.nvoy * 3, self.nvox*self.nvoy * 3]).transpose(2, 1)
            m = torch.sparse.mm(R.squeeze(0).to_sparse(), m.squeeze(0))
            m = m.unsqueeze(0)

            mx = m[:, 0:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)
            my = m[:, 1:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)
            mz = m[:, 2:m.size()[1]:3, :].reshape(m.size(0), self.nvox, self.nvoy, 1)

        m = torch.cat([mx, my, mz], -1)

        return m
