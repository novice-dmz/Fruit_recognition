import utils
import numpy as np
'''
    此为测试功能文件，不需看
'''
# a = [0.1, 0.2, 0.3, 0.1]
# b = [0.4, 0.5, 0.6, 0.4]
# arr = np.array([a, b])
# np.savetxt('output/test.txt', arr)
#load = np.loadtxt('output/test.txt')
#c, d = load.tolist()
# utils.save_dictionary('output/accuracy.txt', c)
utils.draw_acc_graph('output/accuracy.txt')
