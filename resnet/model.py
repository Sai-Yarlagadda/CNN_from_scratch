import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        My custom ResidualBlock

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----

        self.conv2d_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0),
            nn.BatchNorm2d(out_channels)
        )

        self.Relu = nn.ReLU()

        

    def forward(self, x):
       
        # ----- TODO -----
        output = self.conv2d_1(x)
        output = self.conv2d_2(output)
        shortcut = self.shortcut(x)
        output += shortcut
        output = self.Relu(output)

        return output


class MyResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        """
        My custom ResNet.

        [input]
        * in_channels  : input channel number
        * num_classes  : number of classes

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----

        self.conv_initial = nn.Conv2d(in_channels=in_channels,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride = 1,
                                      padding = 1)
        
        self.batch_norm_initial = nn.BatchNorm2d(num_features=64)
        self.initial_Relu = nn.ReLU()

        self.block1 = ResidualBlock(in_channels=64,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride =2)
        
        self.block2 = ResidualBlock(in_channels=128,
                                    out_channels=256,
                                    kernel_size=3,
                                    stride=2)
        
        self.block3 = ResidualBlock(in_channels=256,
                                    out_channels=512,
                                    kernel_size=3,
                                    stride=2)
                
        self.AvgPool_2d = nn.AvgPool2d(kernel_size=4)

        self.Linear_1d = nn.Linear(512, out_features=num_classes)
        self.Flatten = nn.Flatten()
        self.softmax = nn.Softmax()

        #raise NotImplementedError


    def forward(self, x, return_embed=False):
        """
        Forward path.

        [input]
        * x             : input data
        * return_embed  : whether return the feature map of the last conv layer or not

        [output]
        * output        : output data
        * embedding     : the feature map after the last conv layer (optional)
        
        [hint]
        * See the instruction PDF for network details
        * You want to set return_embed to True if you are dealing with CAM
        """

        # ----- TODO -----
        output = self.conv_initial(x)
        output = self.batch_norm_initial(output)
        output = self.initial_Relu(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.AvgPool_2d(output)
        output = self.Flatten(output)
        output = self.Linear_1d(output)
        output =  self.softmax(output)


        return output


def init_weights_kaiming(m):

    """
    Kaming initialization.

    [input]
    * m : torch.nn.Module

    [hint]
    * Refer to the course slides/recitations for more details
    * Initialize the bias term in linear layer by a small constant, e.g., 0.01
    """

    if isinstance(m, nn.Conv2d):
        # ----- TODO -----
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # ----- TODO -----
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


if __name__ == "__main__":

    # set model
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)
    
    # sanity check
    input = torch.randn((64, 3, 32, 32), requires_grad=True)
    output = net(input)
    print(output.shape)
