# idx: resnet18: layer2: 9, layer4: 19
# idx: convnet: layer2: , layer4: 76
# act2
# epo_num = [i for i in range(0, 200, 22)]
# act4
    # resnet18
# epo_num = [i-1 for i in range(0, 216, 4)]
epo_num = [-1, 3, 6, 13, 15, 18, 25, 27, 30, 37, 39, 42, 49, 51, 54, 61, 63, 66, 73, 75, 78, 85, 87, 90, 97, 99, 102, 109, 111, 114, 121, 123, 126, 133, 135, 138, 145, 147,
150, 157, 159, 162, 169, 171, 174, 181, 183, 186, 193, 195, 198, 205, 207, 211]
    # convnet
# epo_num = [i-1 for i in range(0, 432, 6)]
    # less
# epo_num = [i-1 for i in range(0, 432, 8)]
# act4 2x epoch
# epo_num = [i for i in range(0, 420, 16)]
epoch = 0
idx = 0
smooth = True
use_all_sparsity = False
activation_sparsity_num = 0
flag = False
mode = 'none'
args = None
valid_cal = {}
print(epo_num)
