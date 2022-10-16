import matplotlib.pyplot as plt
import numpy as np


def print_hist2(data1, data2, name):
    # plt.xlim(-0.1, 0.1)
    fi = plt.figure()
    '''打印data1'''
    data1 = data1.flatten().cpu().detach().numpy()
    mean1 = np.mean(data1)
    std1 = np.std(data1)
    plt.hist(data1, 'auto', density=True, facecolor='g', alpha=0.75 )
    plt.text(0.015, 32, (r'$\mu=%.2f' % mean1) + (',\ \sigma=%.2f' % std1) + '$')
    '''打印data2'''
    data2 = data2.flatten().cpu().detach().numpy()
    mean2 = np.mean(data2)
    std2 = np.std(data2)
    plt.ylabel('Data distribution')
    plt.hist(data2, 'auto', density=True, facecolor='r', alpha=0.75)
    plt.text(0.015, 35, (r'$\mu=%.2f' % mean2) + (',\ \sigma=%.2f' % std2) + '$')
    plt.savefig('./result/act/act' + str(name) + ".png")
    plt.close(fi)


def print_hist3(data1, data2, data3, name):
    # plt.xlim(-0.1, 0.1)
    fi = plt.figure()
    '''打印data1'''
    data1 = data1.flatten().cpu().detach().numpy()
    print(data1.shape)
    mean1 = np.mean(data1)
    std1 = np.std(data1)
    plt.hist(data1, 'auto', density=True, facecolor='g', alpha=0.75)
    plt.text(0.015, 32, (r'$\mu=%.2f' % mean1) + (',\ \sigma=%.2f' % std1) + '$')
    '''打印data2'''
    data2 = data2.flatten().cpu().detach().numpy()
    print(data2.shape)
    mean2 = np.mean(data2)
    std2 = np.std(data2)
    plt.ylabel('Data distribution')
    plt.hist(data2, 'auto', density=True, facecolor='r', alpha=0.75)
    plt.text(0.015, 35, (r'$\mu=%.2f' % mean2) + (',\ \sigma=%.2f' % std2) + '$')
    '''打印data3'''
    data3 = data3.flatten().cpu().detach().numpy()
    print(data3.shape)
    mean3 = np.mean(data3)
    std3 = np.std(data3)
    plt.ylabel('Data distribution')
    plt.hist(data3, 'auto', density=True, facecolor='b', alpha=0.75)
    plt.text(0.015, 28, (r'$\mu=%.2f' % mean3) + (',\ \sigma=%.2f' % std3) + '$')
    plt.savefig('./result/act/act' + str(name) + "g2.png")
    plt.close(fi)

