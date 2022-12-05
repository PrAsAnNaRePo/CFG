import torch
from torch import nn
from models.text_encoder import GPT
from models.VQvae import Encoder, VectorQuantizer, Decoder


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, latent_shape=64) -> None:
        super().__init__()
        self.latent_shape = latent_shape
        self.embed_dim = embed_dim
        self.gpt = GPT(
            vocab_size=vocab_size,
            d_model=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads
        ) # input(BATCH_SIZE, seq_len) -> output(BATCH_SIZE, seq_len, embed_dim) ---> (BATCH_SIZE, embed_dim, latent_space, l_space)

        self.toshape = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(embed_dim, latent_shape*latent_shape))
        )
    
    def forward(self, text_tokens):
        text_embedding = self.gpt(text_tokens)
        return self.toshape(text_embedding).reshape(-1, self.embed_dim, self.latent_shape, self.latent_shape)


class CFG(nn.Module):
    def __init__(self,
                vocab_size,
                 num_text_encoder_layers,
                  num_heads,
                   h_dim,
                    res_h_dim,
                     n_res_layers,
                     n_embeddings,
                      embedding_dim,
                       beta,
                        unet_filter=32,
                         save_img_embedding_map=False,
                          device='cpu'):
        super(CFG, self).__init__()

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embedding_dim,
            num_layers=num_text_encoder_layers,
            num_heads=num_heads
        )

        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, filters=unet_filter)

        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device=device)

        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, filters=unet_filter)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None
    
    def get_text_embedding(self, tokens):
        return self.text_encoder(tokens)

    def encode(self, img):
        return self.encoder(img)

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, img, text):

        z_e = self.encode(img)
        text_embed = self.get_text_embedding(text)

        z_e = self.pre_quantization_conv(z_e) # (B, h_dim, latent, latent) -> (B, embedd_dim, latent, latent)
        
        embedding_loss, z_q, perplexity, min_encodings, _ = self.vector_quantization(
            z_e*text_embed)

        x_hat = self.decode(z_q)
        
        return embedding_loss, x_hat, perplexity, min_encodings