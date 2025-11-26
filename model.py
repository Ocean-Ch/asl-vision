import torch
import torch.nn as nn
from torchvision import models

class ASLResNetLSTM(nn.Module):
    def __init__(self, num_classes=2000, lstm_hidden=256, num_lstm_layers=2):
        super(ASLResNetLSTM, self).__init__()
        
        # 1. load pretrained MobileNetV2
        # use weights='DEFAULT' which is the modern way to load ImageNet weights
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # only need the "features" part of MobileNet, not the classifier
        self.feature_extractor = self.backbone.features 
        
        # freeze weights so GPU does not cry
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # global average pooling to flatten spatial dims (7x7 -> 1x1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. LSTM
        # mobilenetv2 output is 1280 dim
        self.lstm = nn.LSTM(
            input_size=1280, 
            hidden_size=lstm_hidden, 
            num_layers=num_lstm_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # 3. Classification Head
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # input shape: (Batch, Frames, Channels, Height, Width)
        b, t, c, h, w = x.size()
        
        # flatten batch and time to pass through CNN as "images"
        c_in = x.view(b * t, c, h, w)
        
        # CNN forward (no gradients for efficiency)
        with torch.no_grad():
            features = self.feature_extractor(c_in) # Out: (B*T, 1280, 7, 7)
            features = self.pool(features)          # Out: (B*T, 1280, 1, 1)
            features = features.flatten(1)          # Out: (B*T, 1280)
            
        # reshape back to sequence for LSTM: (Batch, Frames, 1280)
        lstm_in = features.view(b, t, -1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_in)
        
        # take the output of the last frame only
        last_frame = lstm_out[:, -1, :] 
        
        # classifier
        return self.fc(last_frame)