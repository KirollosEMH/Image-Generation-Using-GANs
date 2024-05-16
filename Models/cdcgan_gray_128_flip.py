import torch
import torch.nn as nn

# class cDCGAN_Generator_Gray_128_Flip(nn.Module):
#     def __init__(self, z_dim, channels_img, features_g,num_classes,img_size,embed_size):
#         super(cDCGAN_Generator_Gray_128_Flip, self).__init__()
#         self.img_size = img_size
#         self.generator = nn.Sequential(
#             # Input: N x (z_dim * 2) x 1x1
#             self._block(z_dim * 2, features_g * 32, 4, 1, 0),  # N x features_g * 32 x 4x4
#             self._block(features_g * 32, features_g * 16, 4, 2, 1),  # N x features_g * 16 x 8x8
#             self._block(features_g * 16, features_g * 8, 4, 2, 1),  # N x features_g * 8 x 16x16
#             self._block(features_g * 8, features_g * 4, 4, 2, 1),  # N x features_g * 4 x 32x32
#             self._block(features_g * 4, features_g * 2, 4, 2, 1),  # N x features_g * 2 x 64x64
#             nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),  # N x channels_img x 128x128
#             nn.Tanh()  # [-1, 1]
#         )
#         self.embed = nn.Embedding(num_classes, embed_size)
#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 bias=False
#             ), # deconvolution
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#     def forward(self, x, labels):
#         # latent vector z: N x z_dim x 1 x 1
#         embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
#         x = torch.cat([x, embedding], dim=1)
#         return self.generator(x)


class cDCGAN_Generator_Gray_128_Flip(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        super(cDCGAN_Generator_Gray_128_Flip, self).__init__()
        self.img_size = img_size
        self.generator = nn.Sequential(
            # Input: N x (z_dim * 2) x 1x1
            self._block(z_dim * 2, features_g * 32, 4, 1,
                        0),  # N x features_g * 32 x 4x4
            self._block(features_g * 32, features_g * 16, 4,
                        2, 1),  # N x features_g * 16 x 8x8
            self._block(features_g * 16, features_g * 8, 4,
                        2, 1),  # N x features_g * 8 x 16x16
            self._block(features_g * 8, features_g * 4, 4,
                        2, 1),  # N x features_g * 4 x 32x32
            self._block(features_g * 4, features_g * 2, 4,
                        2, 1),  # N x features_g * 2 x 64x64
            # N x channels_img x 128x128
            nn.ConvTranspose2d(features_g * 2, channels_img,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),  # deconvolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, labels):
        # latent vector z: N x z_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.generator(x)
