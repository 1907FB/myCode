import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

from torchvision.models import resnet18

with torch.cuda.device(0):
  net = resnet18(pretrained=True).cuda()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

