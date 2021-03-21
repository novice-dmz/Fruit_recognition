import utils
'''
    此为测试功能文件，不需看
'''
a = [0.1, 0.2, 0.3]
b = [0.4, 0.5, 0.6]
c = {a[i]: b[i] for i in range(len(a))}
utils.save_dictionary('output/accuracy.txt', c)
utils.draw_acc_graph('output/accuracy.txt')
