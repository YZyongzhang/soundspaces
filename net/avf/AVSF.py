import torch
import torch.nn as nn
import numpy as np
import librosa
import pdb
import matplotlib.pyplot as plt
import random
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


class AudioVisualEncoder(nn.Module):
    def __init__(self, visual_token_dim, audio_token_dim, hidden_dim, num_heads, num_layers, num_channels=2, num_mels=128, max_length=1024):
        super(self).__init__()
        
        self.visual_token_dim = visual_token_dim
        self.audio_token_dim = audio_token_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_mels = num_mels
        self.max_length = max_length
        
        # Linear layers for token encoding
        self.visual_linear = nn.Linear(visual_token_dim, hidden_dim)
        self.audio_linear = nn.Linear(audio_token_dim, hidden_dim)
        
        # Positional embeddings for visual and audio
        self.visual_positional_encoding = nn.Parameter(torch.randn(max_length, hidden_dim))  # sin/cos embedding
        self.audio_positional_encoding = nn.Parameter(torch.randn(max_length, hidden_dim))  # sin/cos embedding
        
        # Learnable channel embedding for audio
        self.channel_embedding = nn.Embedding(num_channels, hidden_dim)  # Embedding for left and right channels
        
        # Transformer Encoder Layers for Visual and Audio
        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), 
            num_layers=num_layers
        )
        
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), 
            num_layers=num_layers
        )
    
    def sinusoidal_positional_encoding(self, length, dim):
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pos_embedding = torch.zeros(length, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding
    
    def forward(self, visual_tokens, audio_tokens, audio_channels):
        # 1. Visual Tokens Encoding
        visual_features = self.visual_linear(visual_tokens)  # 使用线性层编码视觉特征
        visual_features = visual_features + self.visual_positional_encoding[:visual_tokens.size(0), :]  # 添加位置编码
        
        # 2. Audio Tokens Encoding
        audio_features = self.audio_linear(audio_tokens)  # 使用线性层编码音频特征
        audio_features = audio_features + self.audio_positional_encoding[:audio_tokens.size(0), :]  # 添加位置编码
        
        # 添加音频通道嵌入
        audio_features = audio_features + self.channel_embedding(audio_channels).unsqueeze(0)
        
        # 3. Pass through Transformer Encoder
        visual_encoded = self.visual_transformer(visual_features)  # 通过transformer编码视觉特征
        audio_encoded = self.audio_transformer(audio_features)    # 通过transformer编码音频特征
        
        return visual_encoded, audio_encoded
def test():
    ## test audio
    """
    Audio = AudioEmbed(audio_chanel=2)
    audio = torch.rand((256,2,18000)) # batch_size , channel , rate
    embed = Audio(audio)
    print(embed.shape) # batch_size , channel , patches , feature (batchsize , 2,64,128)
    # pdb.set_trace()
    cmap = plt.colormaps['hot']   
    cmap = cmap(np.arange(cmap.N))  
    cmap[0, :] = [0, 0, 0, 1]  
    cmap[1:, :] = [1, 1, 0, 1]
    new_cmap = plt.cm.colors.ListedColormap(cmap)
    plt.imshow(embed[0][0], cmap=new_cmap, interpolation='nearest', origin='lower')
    plt.title('Masked Mel-Spectrogram')
    plt.savefig('./masked_mel_spectrogram.png')
    """
    
    ## test visual
    """
    Visual = VisualEmbed(img_chanel=4 , img_size=256 , embed_dim=128)
    visual = torch.rand((256,256,256,4)) # batchsize , rgba
    embed = Visual(visual)
    print(embed.shape) # batch_size, patches , feature (batchsize ,64,128)
    """
test()
    