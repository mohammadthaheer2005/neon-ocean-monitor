import torch
import torch.nn as nn
import numpy as np

class ConvLSTMCell(nn.Module):
    """
    A single cell of the ConvLSTM network. 
    Processes spatial data (ocean grids) while maintaining temporal memory (trends).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis (Inputs + Hidden State)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class NeonOceanBrain(nn.Module):
    """
    The Main AI Model: ST-ConvLSTM-Attention (Simplified).
    Inputs: Sequence of Ocean Maps [Batch, Time, Channels, Height, Width]
    Outputs: Predicted Map for Next Time Step
    """
    def __init__(self, input_channels=1, hidden_dim=16, kernel_size=(3,3)):
        super(NeonOceanBrain, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size, bias=True)
        # Final layer to map back to 1 channel (Algae prediction)
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1) 
        
    def forward(self, x):
        """
        Forward pass through time.
        x shape: [Batch, Time, Channels, Height, Width]
        """
        batch_size, seq_len, _, height, width = x.size()
        h, c = self.cell.init_hidden(batch_size, (height, width))
        
        # Iterate through time steps
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            h, c = self.cell(input_t, (h, c))
            
        # Prediction based on last hidden state
        prediction = self.final_conv(h)
        return torch.sigmoid(prediction) # Normalize output between 0 and 1

    def predict_bloom_risk(self, input_np_array):
        """
        User-friendly inference method.
        Input: Numpy array of shape (Time, H, W)
        Output: Risk Heatmap (H, W)
        """
        self.eval()
        with torch.no_grad():
            # Prepare tensor: Add Batch & Channel dims -> [1, Time, 1, H, W]
            tensor_in = torch.FloatTensor(input_np_array).unsqueeze(0).unsqueeze(2)
            
            # Forward pass
            output = self.forward(tensor_in)
            
            # Squeeze back to [H, W]
            return output.squeeze().numpy()

if __name__ == "__main__":
    # Sanity Check
    print("🧠 Initializing NeonOcean Brain...")
    model = NeonOceanBrain()
    
    # Create dummy input: 5 Days of data, 64x64 grid
    dummy_input = torch.randn(1, 5, 1, 64, 64) 
    
    output = model(dummy_input)
    print(f"✅ Neural Pass Successful. Output Shape: {output.shape}")
