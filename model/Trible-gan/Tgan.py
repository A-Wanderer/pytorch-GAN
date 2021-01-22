import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")#培训次数
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")#批量大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")#亚当：学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")#梯度一阶动量的衰减
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")#梯度一阶动量的衰减
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")#批处理生成期间要使用的cpu线程数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")#潜空间的维数
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")#每个图像维度的大小
parser.add_argument("--channels", type=int, default=1, help="number of image channels")#图像通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")#图像样本间隔
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(cuda)
print(torch.cuda.current_device())
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator1 = Generator()
generator2 = Generator();
discriminator = Discriminator()

if cuda:
    generator1.cuda()
    generator2.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G1 = torch.optim.Adam(generator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G2 = torch.optim.Adam(generator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G1toG2 = torch.optim.Adam(generator1.parameters(), lr=opt.lr*0.1, betas=(opt.b1, opt.b2))
optimizer_G2toG1 = torch.optim.Adam(generator2.parameters(), lr=opt.lr*0.1, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator1
        # -----------------

        optimizer_G1.zero_grad()

        # Sample noise as generator input
        z1 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs1 = generator1(z1)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs1), valid)

        g_loss.backward()
        optimizer_G1.step()


        # -----------------
        #  Train Generator1
        # -----------------

        optimizer_G2.zero_grad()

        # Sample noise as generator input
        z2 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs2 = generator2(z2)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs2), valid)

        g_loss.backward()
        optimizer_G2.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss1 = adversarial_loss(discriminator(gen_imgs1.detach()), fake)
        fake_loss2 = adversarial_loss(discriminator(gen_imgs2.detach()), fake)
        d_loss = (real_loss + fake_loss1 + fake_loss2) / 3

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator1 - Generator2
        # ---------------------

        z3 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs1 = generator1(z3)
        gen_imgs2 = generator2(z3)

        if(fake_loss1 > fake_loss2):#now generator2 is better
            optimizer_G1toG2.zero_grad()
            target = discriminator(gen_imgs2.detach())
            target = Variable(target, requires_grad=False)
            G1toG2loss = adversarial_loss(discriminator(gen_imgs1), target)
            G1toG2loss.backward()
            optimizer_G1toG2.step()

        else:#now generator1 is better
            optimizer_G2toG1.zero_grad()
            target = discriminator(gen_imgs1.detach())
            target = Variable(target, requires_grad=False)
            G1toG2loss = adversarial_loss(discriminator(gen_imgs2), target)
            G1toG2loss.backward()
            optimizer_G2toG1.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs1.data[:25], "images/1+%d.png" % batches_done, nrow=5, normalize=True)
            save_image(gen_imgs2.data[:25], "images/2+%d.png" % batches_done, nrow=5, normalize=True)