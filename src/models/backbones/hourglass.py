import torch
from torch import nn
import torch.nn.functional as F

from models.layers import Conv, Residual, Merge


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassBisected(nn.Module):
    def __init__(
            self,
            block,
            nblocks,
            in_planes,
            depth=4
    ):
        super(HourglassBisected, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                _res = []
                if j == 1:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                else:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                    _res.append(self._make_residual(block, nblocks, in_planes))

                res.append(nn.ModuleList(_res))

            if i == 0:
                _res = []
                _res.append(self._make_residual(block, nblocks, in_planes))
                _res.append(self._make_residual(block, nblocks, in_planes))
                res.append(nn.ModuleList(_res))

            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1_1 = self.hg[n - 1][0][0](x)
        up1_2 = self.hg[n - 1][0][1](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1][0](low1)

        if n > 1:
            low2_1, low2_2, latent = self._hourglass_foward(n - 1, low1)
        else:
            latent = low1
            low2_1 = self.hg[n - 1][3][0](low1)
            low2_2 = self.hg[n - 1][3][1](low1)

        low3_1 = self.hg[n - 1][2][0](low2_1)
        low3_2 = self.hg[n - 1][2][1](low2_2)

        up2_1 = F.interpolate(low3_1, scale_factor=2)
        up2_2 = F.interpolate(low3_2, scale_factor=2)
        out_1 = up1_1 + up2_1
        out_2 = up1_2 + up2_2

        return out_1, out_2, latent

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


class StackedHourglass(nn.Module):
    # 8, 256, 16, False, 0
    def __init__(
        self,
        nstack=4,
        inp_dim=256,
        oup_dim=21,
        recur_hg=4,
        start=True,
        bn=False,
        increase=0,
        **kwargs
    ):
        super(StackedHourglass, self).__init__()
        self.start = start
        self.nstack = nstack
        # self.pool = nn.AdaptiveAvgPool2d(32)

        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim),
            nn.MaxPool2d(2, 2),  # added extra
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(recur_hg, inp_dim, bn, increase),
            ) for i in range(nstack)]
        )

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)]
        )

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack-1)])
        self.nstack = nstack

    def forward(self, x):
        # x: [b, 3, 256, 256]
        if self.start:
            x = self.pre(x)  # torch.Size([b, 256, 32, 32])
            img_features = x

        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)  # torch.Size([b, 256, 32, 32])
            feature = self.features[i](hg)  # torch.Size([b, 256, 32, 32])
            preds = self.outs[i](feature)  # torch.Size([b, n_joints, 32, 32])
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                # [b, inp_dim, 32, 32] + [b, inp_dim, 32, 32] + [b, inp_dim, 32, 32] = [b, inp_dim, 32, 32]
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        out_hm = torch.stack(combined_hm_preds, 1)
        out_feature = feature

        # return torch.stack(combined_hm_preds, 1), feature # [b, nstack, n_joints, 32, 32], [b, 256, 32, 32]
        return [img_features if self.start else None,  # early image features
                out_hm,  # heatmaps
                out_feature]  # output features before heatmap


class StackedBisectedHourglass(nn.Module):
    def __init__(
        self,
        nstack=4,
        inp_dim=256,
        oup_dim=21,
        recur_hg=4,
        start=True,
        nblocks=1,
        block=Residual,
    ):
        super(StackedBisectedHourglass, self).__init__()
        self.start = start
        self.njoints = oup_dim
        self.nstacks = nstack
        self.in_planes = inp_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, 64, 128)
        self.layer2 = self._make_residual(block, nblocks, 128, 256)
        self.layer3 = self._make_residual(block, nblocks, 256, inp_dim)

        ch = inp_dim

        hg2b, res1, res2, fc1, _fc1, fc2, _fc2 = [], [], [], [], [], [], []
        hm, _hm, mask, _mask = [], [], [], []
        for i in range(self.nstacks):  # 2
            hg2b.append(HourglassBisected(block, nblocks, ch, depth=recur_hg))
            res1.append(self._make_residual(block, nblocks, ch, ch))
            res2.append(self._make_residual(block, nblocks, ch, ch))
            fc1.append(self._make_fc(ch, ch))
            fc2.append(self._make_fc(ch, ch))
            hm.append(nn.Conv2d(ch, self.njoints, kernel_size=1, bias=True))
            mask.append(nn.Conv2d(ch, 1, kernel_size=1, bias=True))

            if i < self.nstacks-1:
                _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _hm.append(nn.Conv2d(self.njoints, ch, kernel_size=1, bias=False))
                _mask.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

        self.hg2b = nn.ModuleList(hg2b)  # hgs: hourglass stack
        self.res1 = nn.ModuleList(res1)
        self.fc1 = nn.ModuleList(fc1)
        self._fc1 = nn.ModuleList(_fc1)
        self.res2 = nn.ModuleList(res2)
        self.fc2 = nn.ModuleList(fc2)
        self._fc2 = nn.ModuleList(_fc2)
        self.hm = nn.ModuleList(hm)
        self._hm = nn.ModuleList(_hm)
        self.mask = nn.ModuleList(mask)
        self._mask = nn.ModuleList(_mask)

    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, self.relu)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [b, 3, 256, 256]
        l_hm, l_mask, l_enc = [], [], []
        if self.start:
            x = self.conv1(x)  # x: (N,64,128,128)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.maxpool(x)  # x: (N,128,64,64)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.maxpool(x)  # added extra
            img_features = x

        for i in range(self.nstacks):  # 2
            y_1, y_2, _ = self.hg2b[i](x)

            y_1 = self.res1[i](y_1)
            y_1 = self.fc1[i](y_1)
            est_hm = self.hm[i](y_1)
            l_hm.append(est_hm)

            y_2 = self.res2[i](y_2)
            y_2 = self.fc2[i](y_2)
            est_mask = self.mask[i](y_2)
            l_mask.append(est_mask)

            if i < self.nstacks-1:
                _fc1 = self._fc1[i](y_1)
                _hm = self._hm[i](est_hm)
                _fc2 = self._fc2[i](y_2)
                _mask = self._mask[i](est_mask)
                x = x + _fc1 + _fc2 + _hm + _mask
                l_enc.append(x)
            else:
                l_enc.append(x + y_1 + y_2)
        assert len(l_hm) == self.nstacks
        out_hm, out_mask, out_feature = torch.stack(l_hm, 1), torch.stack(l_mask, 1), torch.stack(l_enc, 1)
        # return out_hm, out_mask, out_feature[:, -1]
        return [img_features if self.start else None,
                out_hm,
                out_feature[:, -1],
                out_mask]


if __name__ == "__main__":
    NSTACK = 4
    IN_DIM = 256
    OUT_DIM = 21
    RECUR_HG = 2

    model = StackedHourglass(
        nstack=NSTACK,
        inp_dim=IN_DIM,
        oup_dim=OUT_DIM,
        recur_hg=RECUR_HG,
        start=True,
    )

    sample = torch.rand(2, 3, IN_DIM, IN_DIM)  # [b, c, h, w]
    # torch.Size([b, 256, 32, 32]), torch.Size([b, nstack, 21, 32, 32]), torch.Size([b, 256, 32, 32])
    img_feat, out_hm, out_feat = model(sample)

    print(img_feat.shape, out_hm.shape, out_feat.shape)

    model = StackedBisectedHourglass(
        nstack=NSTACK,
        inp_dim=IN_DIM,
        oup_dim=OUT_DIM,
        recur_hg=RECUR_HG,
        start=True,
    )

    img_feat, out_hm, out_feat, out_mask = model(sample)
    # torch.Size([b, 256, 32, 32]), torch.Size([b, nstack, 21, 32, 32]), torch.Size([b, 256, 32, 32]), torch.Size([b, nstack, 1, 32, 32])
    print(img_feat.shape, out_hm.shape, out_feat.shape, out_mask.shape)
