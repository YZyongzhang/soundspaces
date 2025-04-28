import torch
import torch.nn as nn
import numpy as np
import librosa
import pdb
import matplotlib.pyplot as plt
import random
import math
class VisualEmbed(nn.Module):
    def __init__(self,img_chanel , img_size , embed_dim):
        super().__init__()
        self.img_chanel = img_chanel
        self.img_size = img_size
        self.patch_size = 32
        self.embed_dim = embed_dim
        self.grid_size = self.img_size // self.patch_size 
        self.visual_embed = nn.Conv2d(
            self.img_chanel, 
            embed_dim, 
            kernel_size=32, 
            stride=32
            )
    def forward(self,visual):
        # pdb.set_trace()
        visual = visual.permute(0,3,2,1)
        x = self.visual_embed(visual)
        # x shape (batchsize , embed_dim , self.grid_size , self.grid_size)
        embeding = x.flatten(2)
        # flatten tensor  all behand dim=2
        # embeding shape (batchsize , embed_dim , grid**2)
        return embeding.transpose(1,2)
class AudioEmbed(nn.Module):
    def __init__(self,audio_chanel):
        super().__init__()
        self.audio_chanel = audio_chanel
        self.sr = 18000
        self.n_fft = 512 # 282 * 2 
        self.hop_length = 282 # 18000 // 282 = 64
        self.patch_size = 64
        self.embed_dim = 128
        self.n_mels = 128
    def forward(self,audio):
        # pdb.set_trace()
        _mel = self._mel_features(audio)
        # _mel shape (batchsize , 128 ， 64)
        return _mel.transpose(-1,-2)
    def _mel_features(self,audio):
        mel_features = []
        audio = audio.cpu().numpy()
        for i in range(audio.shape[0]):
            left_channel = audio[i, 0, :]
            right_channel = audio[i, 1, :]
            
            mel_left = librosa.feature.melspectrogram(y=left_channel, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            mel_right = librosa.feature.melspectrogram(y=right_channel, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            
            # masked
            # 1. slip patch 4*4
            # 2. masked patch = 0
            mel_left = self._mask_mel(mel_left)
            mel_right = self._mask_mel(mel_right)
            combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 18000//282 = 64)
            mel_features.append(combined_mel)
        return torch.from_numpy(np.array(mel_features))
    def _mask_mel(self,_mel , mask_num = 8):
        # slip step = 4 * int
        time_size, freq_size = _mel.shape
        patch_size = (16,16)
        patch_stride = 16
        mask_positions = set()
        time_choices = list(range(0, time_size - patch_size[0] + 1, patch_stride))
        freq_choices = list(range(0, freq_size - patch_size[1] + 1, patch_stride))
        # pdb.set_trace()
        for _ in range(mask_num):
            while True:
                patch_time = random.choice(time_choices)
                patch_freq = random.choice(freq_choices)
                
                patch_position = (patch_time, patch_freq)
                
                if patch_position not in mask_positions:
                    mask_positions.add(patch_position)
                    _mel[patch_time:patch_time+patch_size[0], patch_freq:patch_freq+patch_size[1]] = 0
                    break
        return _mel

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        
        # Create the positional encodings once and store them as a buffer
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x, max_len=None):
        if max_len is None:
            max_len = x.size(1)
        return x + self.pe[:, :max_len].to(x.device)

class TransformerModelEncode(nn.Module):
    def __init__(self, num_heads, num_layers, hidden_dim, max_len=5000):
        super(TransformerModelEncode, self).__init__()
        
        
        # Positional encoding for both visual and audio
        self.visual_positional_encoding = PositionalEncoding(hidden_dim, max_len)
        self.audio_positional_encoding = PositionalEncoding(hidden_dim, max_len)
        
        # Learnable channel embedding for audio
        self.channel_embedding = nn.Embedding(2, hidden_dim)  # For left and right channels
        
        # Transformer encoders for visual and audio features
        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        self.shared_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )

    def forward(self, visual_features, audio_features, audio_channel):
        
        # Add channel embedding to audio features
        channel_embeddings = self.channel_embedding(audio_channel)
        audio_features = audio_features + channel_embeddings
        
        # Pass through separate transformer encoders
        visual_features = self.visual_transformer(visual_features)
        audio_features = self.audio_transformer(audio_features)
        
        # You can combine the visual and audio features here, e.g., concatenation
        eAV = torch.cat((visual_features, audio_features), dim=1)
        
        fAV = self.shared_transformer(eAV)
        
        return fAV
class TransformerModelDecoder(nn.Module):
    def __init__(self, hidden_dim, masked_dim, num_heads, num_layers, output_dim):
        super(TransformerModelDecoder, self).__init__()
        
        # Projection layer to reduce the dimensionality of fAV
        self.projection = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Learnable embedding for masked audio tokens
        self.masked_embedding = nn.Parameter(torch.randn(masked_dim))
        
        # Shared audio-visual transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim // 2, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        # Second transformer decoder for refinement
        self.refinement_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim // 2, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        # Final output layer to predict masked binaural audio tokens
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, fAV, masked_tokens):
        # Project the input features to a lower dimension
        gAV = self.projection(fAV)
        
        # Append learnable masked embedding to the projected features
        gAV = torch.cat((gAV, self.masked_embedding.expand(masked_tokens, -1)), dim=1)
        
        # Pass through the shared transformer decoder
        decoder_output = self.transformer_decoder(gAV)
        
        # Refinement through second transformer decoder
        refined_output = self.refinement_decoder(decoder_output)
        
        # Predict the missing binaural audio tokens
        pma = refined_output
        
        return pma

def test():
    ## test audio
    # """
    Audio = AudioEmbed(audio_chanel=2)
    audio = torch.rand((256,2,18000)) # batch_size , channel , rate
    audioembed = Audio(audio)
    print(audioembed.shape) # batch_size , channel , patches , feature (batchsize , 2,64,128)
    # pdb.set_trace()
    # cmap = plt.colormaps['hot']   
    # cmap = cmap(np.arange(cmap.N))  
    # cmap[0, :] = [0, 0, 0, 1]  
    # cmap[1:, :] = [1, 1, 0, 1]
    # new_cmap = plt.cm.colors.ListedColormap(cmap)
    # plt.imshow(embed[0][0], cmap=new_cmap, interpolation='nearest', origin='lower')
    # plt.title('Masked Mel-Spectrogram')
    # plt.savefig('./masked_mel_spectrogram.png')
    # """
    
    ## test visual
    # """
    Visual = VisualEmbed(img_chanel=4 , img_size=256 , embed_dim=128)
    visual = torch.rand((256,256,256,4)) # batchsize , rgba
    visualembed = Visual(visual)
    print(visualembed.shape) # batch_size, patches , feature (batchsize ,64,128)
    # """
    
    ## test transformerencode
    """
    """
    encoder = TransformerModelEncode()
test()
    