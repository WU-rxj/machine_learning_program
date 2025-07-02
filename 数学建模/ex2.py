# 用python生成离散型随机变量
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置随机种子，保证结果可重现
np.random.seed(42)

def generate_and_plot_discrete_rv(size=1000):
    """生成几种常见的离散型随机变量并可视化"""
    # 1. 二项分布 Binomial(n, p)
    # 模拟抛硬币10次，正面朝上(p=0.5)的次数
    n, p = 10, 0.5
    binomial_rv = np.random.binomial(n, p, size)
    
    # 2. 泊松分布 Poisson(lambda)
    # 模拟单位时间内随机事件发生的次数，如每小时到达的顾客数
    lam = 3.0  # 平均每小时3个顾客
    poisson_rv = np.random.poisson(lam, size)
    
    # 3. 几何分布 Geometric(p)
    # 模拟直到第一次成功所需的试验次数，如直到投中篮球所需的投篮次数
    p = 0.3  # 每次投篮的成功概率为0.3
    geometric_rv = np.random.geometric(p, size)
    
    # 4. 自定义离散分布
    # 模拟骰子点数，但使用自定义概率
    # 例如：一个有偏骰子，点数1-6的概率不同
    xk = np.arange(1, 7)  # 可能的值：1,2,3,4,5,6
    pk = (0.1, 0.1, 0.1, 0.2, 0.2, 0.3)  # 对应的概率
    custom_rv = np.random.choice(xk, size=size, p=pk)
    
    # 可视化这些随机变量
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 绘制二项分布
    axes[0, 0].hist(binomial_rv, bins=range(0, n+2), alpha=0.7)
    axes[0, 0].set_title('二项分布 Bin(10, 0.5)')
    axes[0, 0].set_xlabel('成功次数')
    
    # 绘制泊松分布
    max_poisson = max(poisson_rv)
    axes[0, 1].hist(poisson_rv, bins=range(0, max_poisson+2), alpha=0.7)
    axes[0, 1].set_title('泊松分布 Poisson(3)')
    axes[0, 1].set_xlabel('事件发生次数')
    
    # 绘制几何分布（限制x轴范围以便观察）
    axes[1, 0].hist(geometric_rv, bins=range(1, 16), alpha=0.7)
    axes[1, 0].set_title('几何分布 Geom(0.3)')
    axes[1, 0].set_xlabel('直到成功的尝试次数')
    
    # 绘制自定义离散分布
    axes[1, 1].hist(custom_rv, bins=range(1, 8), alpha=0.7)
    axes[1, 1].set_title('自定义离散分布（有偏骰子）')
    axes[1, 1].set_xlabel('骰子点数')
    
    plt.tight_layout()
    plt.savefig('discrete_distributions.png')
    plt.show()

if __name__ == "__main__":
    generate_and_plot_discrete_rv()