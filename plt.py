import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    # 读取csv文件中2,3列的数据，且转化为float类型
    i = 0
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
        i = i+1
    return x, y

def selfplot(path1,path2,path3,label1,label2,label3,title):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    plt.figure()
    # 读取四个文件
    if path1!=None:
        x2, y2 = readcsv(path1)
        plt.plot(x2, y2, color='red', label=label1)
    if path2!=None:
        x, y = readcsv(path2)
        plt.plot(x, y, 'g', label=label2)
    #
    if path3!=None:
        x1, y1 = readcsv(path3)
        plt.plot(x1, y1, color='black', label=label3)
    #
    # x4, y4 = readcsv("run_.-tag-acc.csv")
    # plt.plot(x4, y4, color='blue', label='acc')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # ylim和xlim的最大最小值根据csv文件的极大极小值决定
    # plt.ylim(0, 2)
    # plt.xlim(0, 50)

    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(fontsize=16)
    plt.title(title)
    plt.ylim([5.78,6])
    plt.show()

if __name__ == "__main__":
    # selfplot("logs/tsp_50/28/val_avg_reward.csv","logs/tsp_50/23/val_avg_reward.csv", "logs/tsp_50/16/val_avg_reward.csv",
    #          "multi_contrast_loss with modified baseline","noWnoG","baseline","TSP50 val_avg_reward seed = 1234")

    # selfplot("logs/tsp_50/13/avg_cost.csv", "logs/tsp_50/12/avg_cost.csv",
    #          "logs/tsp_50/14/avg_cost.csv",
    #          "3060_CURL=node_embedding", "3060_CURL=False yuancode", "3060_CURL=false doitagain",
    #          "TSP50 avg_cost")
    # selfplot("logs/tsp_20/21/c_node_loss.csv", "logs/tsp_20/20/c_node_loss.csv",
    #          None,
    #          "No W", "W", "3060_CURL=false doitagain",
    #          "TSP20 c_loss")
    selfplot("logs/tsp_50/43/val_avg_reward.csv", "logs/tsp_50/42/val_avg_reward.csv",
            "logs/tsp_50/14/val_avg_reward.csv",
             "CONCAT fused invariant", "PLUS fused invariant", "AM", "TSP50")
    # selfplot("logs/tsp_50/30/c_node_loss.csv", "logs/tsp_50/28/c_node_loss.csv",
    #          None,
    #          "POCE_no_update", "POCE", "baseline", "TSP50 c node loss seed = 1234")