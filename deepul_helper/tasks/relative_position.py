import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """Small shared (siamese) CNN encoder for one CIFAR10-sized patch."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).flatten(start_dim=1)  # (N, out_dim)


class RelativePosition(nn.Module):
    """Context Prediction (Doersch et al., 2015): predict which of 8
    directions a random query patch sits in relative to a fixed center patch."""
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    # 0 1 2
    # 3 _ 4
    # 5 6 7
    OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    def __init__(self, dataset, n_classes, patch_size=8, gap=2, jitter=1):
        super().__init__()
        if dataset not in ('cifar10', 'cifar100'):
            raise Exception('RelativePosition currently only supports cifar10-sized (32x32) inputs')

        self.patch_size = patch_size
        self.gap = gap
        self.jitter = jitter
        self.encoder = PatchEncoder(out_dim=256)
        self.latent_dim = 256

        self.classifier_head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8)
        )
        self.dataset = dataset
        self.n_classes = n_classes

    def construct_classifier(self):
        return nn.Sequential(
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )

    def _extract_patches(self, images):
        B, C, H, W = images.shape
        p, g, j = self.patch_size, self.gap, self.jitter
        stride = p + g
        grid_span = 3 * stride
        origin = (H - grid_span) // 2
        cell_offset = (stride - p) // 2

        centers, queries, labels = [], [], []
        for b in range(B):
            label = random.randrange(8)
            dr, dc = self.OFFSETS[label]

            cr = origin + stride + cell_offset + random.randint(-j, j)
            cc = origin + stride + cell_offset + random.randint(-j, j)
            qr = origin + (1 + dr) * stride + cell_offset + random.randint(-j, j)
            qc = origin + (1 + dc) * stride + cell_offset + random.randint(-j, j)

            cr, cc = self._clamp(cr, H, p), self._clamp(cc, W, p)
            qr, qc = self._clamp(qr, H, p), self._clamp(qc, W, p)

            centers.append(images[b, :, cr:cr+p, cc:cc+p])
            queries.append(images[b, :, qr:qr+p, qc:qc+p])
            labels.append(label)

        centers = torch.stack(centers, dim=0)
        queries = torch.stack(queries, dim=0)
        labels = torch.tensor(labels, dtype=torch.long, device=images.device)

        centers = self._color_drop(centers)
        queries = self._color_drop(queries)
        return centers, queries, labels

    @staticmethod
    def _clamp(start, size, patch):
        return max(0, min(start, size - patch))

    @staticmethod
    def _color_drop(patches):
        # Zero out 2 of 3 color channels (per batch) so the model can't use
        # chromatic aberration as a shortcut -- see Demo 5 in the notebook.
        out = patches.clone()
        keep = random.randrange(3)
        for c in range(3):
            if c != keep:
                out[:, c] = out[:, c].mean()
        return out

    def forward(self, images):
        centers, queries, labels = self._extract_patches(images)
        z_center = self.encoder(centers)
        z_query = self.encoder(queries)

        logits = self.classifier_head(torch.cat([z_center, z_query], dim=1))
        loss = F.cross_entropy(logits, labels)

        pred = logits.argmax(dim=-1)
        acc = pred.eq(labels).float().mean() * 100.
        return dict(Loss=loss, Acc1=acc), self.encoder(images)  # WHOLE_IMAGE_FEATURES

    def encode(self, images):
        return self.encoder(images)  # WHOLE_IMAGE_FEATURES