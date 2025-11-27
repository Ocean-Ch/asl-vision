import torch
import torch.nn as nn
from torchvision import models

class ASLResNetLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for ASL video recognition.
    
    This model processes video sequences by:
    1. using MobileNetV2 to extract features from each frame (spatial understanding)
    2. using LSTM to process the sequence of frame features (temporal understanding)
    3. using a fully connected layer for classification
    
    The CNN backbone is FROZEN (not trained) to save computation and memory,
    while the LSTM and FC layer are trainable.
    
    Args:
        num_classes (int): Number of sign language words to classify (default: 2000)
        lstm_hidden (int): Size of LSTM hidden state (default: 256)
        num_lstm_layers (int): Number of stacked LSTM layers (default: 2)
    
    Input shape: (batch, frames, channels, height, width) - e.g., (8, 32, 3, 224, 224)
    
    Output shape: (batch, num_classes) - logits (raw scores) for each class
    """
    def __init__(self, num_classes: int = 2000, lstm_hidden: int = 256, num_lstm_layers: int = 2, frozenCNN: bool = True):
        """
        Initializes the model architecture.
        
        Sets up three main components:
        1. MobileNetV2 feature extractor (pretrained, frozen)
        2. LSTM for temporal sequence processing
        3. Fully connected layer for final classification
        """
        # required for all PyTorch nn.Module classes
        # set up the module infrastructure (parameter tracking, etc.)
        super(ASLResNetLSTM, self).__init__()
        
        # ========== 1. CNN Feature Extractor (MobileNetV2) ==========
        # load pretrained MobileNetV2 model (lightweight CNN architecture)
        # weights='DEFAULT' loads ImageNet pretrained weights
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # extract only the feature extraction part (conv layers)
        # MobileNetV2 has two parts: 'features' (conv layers) and 'classifier' (final FC layer)
        # only need 'features' since we'll add our own classifier
        self.feature_extractor = self.backbone.features 
        
        # freeze the CNN weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = not frozenCNN
            
        # global average pooling: reduces spatial dimensions from (7, 7) to (1, 1)
        # converts the 2D feature maps into 1D feature vectors
        # AdaptiveAvgPool2d: automatically adapts to input size and outputs exactly (1, 1)
        # input: (batch, channels, height, width) -> output: (batch, channels, 1, 1)
        # We reduce the spatial dimensions to 1x1 to get a single rich feature vector per frame
        # This is because we no longer care about where in the frame a detector has fired,
        # only how strongly it fired overall.
        # This allows us to capture the presence and strength of a pattern irrespective of its location.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========== 2. LSTM Network ==========
        # mobilenetv2 outputs 1280-dimensional feature vectors per frame
        MOBILENET_FEATURE_SIZE = 1280
        # randomly sets 30% of activations to 0 during training to prevent overfitting
        LSTM_DROPOUT = 0.3

        self.lstm = nn.LSTM(
            input_size=MOBILENET_FEATURE_SIZE,
            hidden_size=lstm_hidden,  # Size of LSTM's internal hidden state (memory)
            num_layers=num_lstm_layers,  # Stack multiple LSTM layers for more complex patterns
            batch_first=True,  # sets input format to (batch, sequence_length, features) instead of (sequence_length, batch, features)
            dropout=LSTM_DROPOUT
        )
        
        # ========== 3. Classification Head ==========
        # Final FC layer that maps LSTM output to class predictions
        # Input: LSTM hidden state - shape: (batch, lstm_hidden)
        # Output: num_classes logits - shape: (batch, num_classes)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Called automatically on model(inputs) or model.forward(inputs) call.
        
        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)
               ex: (8, 32, 3, 224, 224) = 8 videos, 32 frames each, RGB images of 224x224
        
        Returns:
            Tensor of shape (Batch, num_classes) containing logits (raw scores) for each gloss class
        
        Process:
        1. reshape video frames to process each frame through CNN
        2. extract features from each frame using MobileNetV2
        3. reshape features back into a sequence
        4. process sequence through LSTM to capture temporal patterns
        5. use final frame's LSTM output for classification
        """
        # Unpack input tensor dimensions
        # b = batch size (number of videos in this batch)
        # t = time/frames (number of frames per video)
        # c = channels (3 for RGB)
        # h = height, w = width
        b, t, c, h, w = x.size()
        
        # Reshape: combine batch and time dimensions
        # This allows us to process all frames through the CNN at once
        # From (Batch, Frames, C, H, W) to (Batch*Frames, C, H, W)
        # Example: (8, 32, 3, 224, 224) -> (256, 3, 224, 224)
        # Now each frame is treated as a separate "image" in a larger batch
        c_in = x.view(b * t, c, h, w)
        
        # extract features from each frame using MobileNetV2
        # torch.no_grad() disables gradient computation
        # since we're not training the CNN, we don't need gradients
        with torch.no_grad():
            # pass frames through MobileNetV2 feature extractor
            # output: (b*t, 1280, 7, 7) - 1280 feature maps of size 7x7 per frame
            features = self.feature_extractor(c_in)
            
            # global average pooling: reduce spatial dimensions
            # (b*t, 1280, 7, 7) -> (b*t, 1280, 1, 1)
            features = self.pool(features)
            
            # flatten: remove spatial dimensions, keep only feature dimension
            # (b*t, 1280, 1, 1) -> (b*t, 1280)
            # flatten(1) flattens dimensions starting from index 1 (keeps batch dimension)
            features = features.flatten(1)
            
        # reshape back to sequence format for LSTM (batch, time, features)
        # from (b*t, 1280) back to (b, t, 1280)
        # -1 tells PyTorch to infer this dimension automatically
        lstm_in = features.view(b, t, -1)
        
        # lstm_out: output at each time step, shape (batch, time, lstm_hidden)
        # _: hidden state tuple (we don't need it here, so we use _ to ignore it)
        lstm_out, _ = self.lstm(lstm_in)
        
        # extract the output from the last frame only
        # use the last frame because it has "seen" all previous frames through the LSTM
        # [:, -1, :] means: all batches, last frame (-1), all features
        # shape: (batch, lstm_hidden)
        last_frame = lstm_out[:, -1, :] 
        
        # final classification: map LSTM output to class predictions
        # output shape: (batch, num_classes) - one score per gloss class
        return self.fc(last_frame)