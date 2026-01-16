import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- Model Components ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        return [x1, x2, x3]


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        tokens = self.vit.forward_features(x)
        feat_map = tokens[:, 1:, :]
        S = int((feat_map.shape[1]) ** 0.5)
        feat_map = rearrange(feat_map, 'b (h w) d -> b d h w', h=S, w=S)
        return feat_map


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RoadSegModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_ch)
        self.transformer_encoder = TransformerEncoder()
        self.fuse = nn.Conv2d(256 + 768, 256, kernel_size=1)

        self.dec1 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        cnn_feats = self.cnn_encoder(x)
        vit_feat = self.transformer_encoder(x)
        cnn_feat_resized = F.interpolate(cnn_feats[-1], size=vit_feat.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([cnn_feat_resized, vit_feat], dim=1)
        fused = self.fuse(fused)

        x = self.dec1(fused, cnn_feats[1])
        x = self.dec2(x, cnn_feats[0])
        x = self.final(x)
        return torch.sigmoid(x)


# --- Image Input + Inference ---
def load_image(path, target_size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img_tensor


def show_result(image_tensor, mask_tensor):
    image = image_tensor.squeeze().permute(1, 2, 0).numpy()
    mask = mask_tensor.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Road Mask")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadSegModel().to(device)
    model.eval()

    # Load input image
    image_path = "/content/Massachusettsimages.jpeg"  # 🔁 Replace with your actual image path
    image_tensor = load_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        print(f"Output min: {output.min().item()}, Output max: {output.max().item()}")
        plt.imshow(output.squeeze().cpu().detach().numpy(), cmap='jet')
        plt.title("Raw Model Output")
        plt.axis("off")
        plt.show()
        output_mask = (output > 0.1).float()

    # Visualize
    show_result(image_tensor.cpu(), output_mask.cpu())
