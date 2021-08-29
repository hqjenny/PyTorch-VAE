import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 log: bool = True,
                 norm: bool = True,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        # self.scale = torch.tensor([2 ** 16] * in_channels).double()
        # self.scale = torch.tensor([64, 4096, 4096, 512, 256, 512, 2**16, 256, 4096, 2**18]).double()
        scale = [64, 4096, 4096, 256, 2**16, 4096, 2**18]
        self.scale = torch.tensor(scale).double()
        self.log_scale = torch.log(self.scale.double())

        # self.scale = torch.tensor([2 ** 18] * in_channels).double()
        # self.encode_scale = torch.tensor([2 ** 18] * in_channels + [2**24]).double()
        self.encode_scale = torch.tensor(scale+[2**24]).double()
        self.log_encode_scale = torch.log(self.encode_scale.double())

        self.norm = norm
        self.log = log

        modules = []
        if hidden_dims is None:
            hidden_dims = [1024,2048]

        # To account for the extra label channel
        in_dim = in_channels + 1 
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 1, stride=1, padding  = 1),                    
#                     nn.BatchNorm2d(h_dim),
                     nn.Linear(in_dim, h_dim),
#                     nn.BatchNorm1d(h_dim),
                     nn.LeakyReLU())
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim+1, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
#                    nn.ConvTranspose2d(hidden_dims[i],
#                                       hidden_dims[i + 1],
#                                       kernel_size=1,
#                                       stride = 1,
#                                       padding=1,
#                                       output_padding=1),
#                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
#                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
#                            nn.ConvTranspose2d(hidden_dims[-1],
#                                               hidden_dims[-1],
#                                               kernel_size=1,
#                                               stride=1,
#                                               padding=1,
#                                               output_padding=1),
#                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
#                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
#                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
#                                      kernel_size= 1, padding= 1),
                            nn.Linear(hidden_dims[-1], in_channels),
                            nn.Sigmoid()
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print("input", input)

        encode_input = input
        encode_scale = self.encode_scale
        if self.log:
            encode_input = torch.log(encode_input.double())
            encode_scale = self.log_encode_scale
        if self.norm:
            encode_input = torch.div(encode_input, encode_scale)
            max_norm_input = torch.max(encode_input).item()
            print("max norm input", max_norm_input)
            assert(max_norm_input <= 1)
        #else:
        #    self.max_norm_input = torch.max(encode_input).item()
        #    encode_input = torch.div(encode_input, self.max_norm_input)

        result = self.encoder(encode_input.double())
 
        # result = self.encoder(norm_input.double())
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        encode_scale = self.scale
        decode_result = result
        if self.log:
            encode_scale = self.log_scale
        if self.norm:
            decode_result = torch.mul(result.double(), encode_scale)
        #else: 
        #    decode_result = torch.mul(result.double(), self.max_norm_input)
        if self.log:
            #decode_result = torch.exp(torch.mul(decode_result, math.log(2)))
            decode_result = torch.exp(decode_result)
        return decode_result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].double() / 10**10
        #embedded_class = self.embed_class(y)
        #embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        #embedded_input = self.embed_data(input)

        embedded = y.view(-1, 1)
        # x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = torch.cat([input.double(), embedded.double()], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, embedded], dim=1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        norm_recons = torch.div(recons.double(), self.scale)
        norm_input = torch.div(input.double(), self.scale)

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(norm_recons, norm_input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]
