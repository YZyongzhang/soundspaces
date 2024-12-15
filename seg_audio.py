from torchvggish import vggish
import torch
from easydict import EasyDict as edict
class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea
class Config:
    def __init__(self) -> None:
        pass
    def device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    def config(self):
        cfg = edict()
        cfg.BATCH_SIZE = 4 # default 4
        cfg.LAMBDA_1 = 0.5 # default: 0.5
        cfg.MASK_NUM = 10 # 10 for fully supervised
        cfg.NUM_CLASSES = 71 # 70 + 1 background

        cfg.TRAIN = edict()

        cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
        cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "./torchvggish/vggish-10086976.pth"
        cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = True #! notice
        cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
        cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "./torchvggish/vggish_pca_params-970ea276.pth"
        return cfg
if __name__ == "__main__":
    config_ = Config()
    vggish = audio_extractor(config_.config() , config_.device())
    print(vggish)