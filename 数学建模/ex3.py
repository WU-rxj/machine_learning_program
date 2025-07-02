# 用取舍法生成连续分布随机数
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
# 添加导入 scipy.special 中的 beta 函数
from scipy.special import beta as scipy_beta

# 1. 使用取舍法生成正态分布随机数
def rejection_normal(size=1000, mu=0, sigma=1):
    # 使用均匀分布作为提议分布
    # 正态分布在 mu±4*sigma 范围内包含了99.99%的概率质量
    x_min, x_max = mu - 4*sigma, mu + 4*sigma
    y_max = 1/(sigma * np.sqrt(2*np.pi))  # 正态分布的最大值
    
    samples = []
    count = 0  # 记录总尝试次数
    
    while len(samples) < size:
        # 从均匀分布生成x和y
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, y_max)
        
        # 计算在x处的正态分布密度
        pdf_value = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
        
        # 取舍判断
        if y <= pdf_value:
            samples.append(x)
        
        count += 1
    
    acceptance_rate = size / count
    return np.array(samples), acceptance_rate

# 2. 使用取舍法生成指数分布随机数
def rejection_exponential(size=1000, scale=1.0):
    # 使用均匀分布作为提议分布
    x_max = 5 * scale  # 覆盖大部分指数分布的概率质量
    y_max = 1/scale    # 指数分布在x=0处的最大值
    
    samples = []
    count = 0
    
    while len(samples) < size:
        x = np.random.uniform(0, x_max)
        y = np.random.uniform(0, y_max)
        
        # 计算在x处的指数分布密度
        pdf_value = (1/scale) * np.exp(-x/scale)
        
        if y <= pdf_value:
            samples.append(x)
        
        count += 1
    
    acceptance_rate = size / count
    return np.array(samples), acceptance_rate

# 3. 使用取舍法生成自定义分布随机数（以Beta分布为例）
def rejection_custom(size=1000, alpha=2, beta=5):
    # 使用均匀分布作为提议分布
    x_min, x_max = 0, 1
    # Beta分布的最大值
    if alpha > 1 and beta > 1:
        mode = (alpha - 1) / (alpha + beta - 2)
        # 使用 scipy_beta 替换 np.beta
        y_max = mode**(alpha-1) * (1-mode)**(beta-1) / (scipy_beta(alpha, beta))
    else:
        y_max = 3.0  # 保守估计
    
    samples = []
    count = 0
    
    while len(samples) < size:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, y_max)
        
        # 计算在x处的Beta分布密度
        if 0 < x < 1:  # Beta分布的定义域是(0,1)
            # 使用 scipy_beta 替换 np.beta
            pdf_value = x**(alpha-1) * (1-x)**(beta-1) / (scipy_beta(alpha, beta))
        else:
            pdf_value = 0
        
        if y <= pdf_value:
            samples.append(x)
        
        count += 1
    
    acceptance_rate = size / count
    return np.array(samples), acceptance_rate

# 主函数：生成并可视化不同分布的随机数
def main():
    # 设置随机数种子以保证结果可重复
    np.random.seed(42)
    
    # 样本量
    sample_size = 5000
    
    # 1. 生成正态分布随机数
    normal_samples, normal_rate = rejection_normal(sample_size)
    print(f"正态分布接受率: {normal_rate:.4f}")
    
    # 2. 生成指数分布随机数
    exp_samples, exp_rate = rejection_exponential(sample_size)
    print(f"指数分布接受率: {exp_rate:.4f}")
    
    # 3. 生成自定义Beta分布随机数
    beta_samples, beta_rate = rejection_custom(sample_size)
    print(f"Beta分布接受率: {beta_rate:.4f}")
    
    # 可视化结果
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 正态分布直方图和理论密度函数
    axes[0].hist(normal_samples, bins=50, density=True, alpha=0.6, label='取舍法生成样本')
    x = np.linspace(-4, 4, 1000)
    axes[0].plot(x, norm.pdf(x), 'r-', label='理论密度函数')
    axes[0].set_title(f'正态分布 N(0,1) - 接受率: {normal_rate:.4f}')
    axes[0].legend()
    
    # 指数分布直方图和理论密度函数
    axes[1].hist(exp_samples, bins=50, density=True, alpha=0.6, label='取舍法生成样本')
    x = np.linspace(0, 5, 1000)
    axes[1].plot(x, expon.pdf(x), 'r-', label='理论密度函数')
    axes[1].set_title(f'指数分布 Exp(1) - 接受率: {exp_rate:.4f}')
    axes[1].legend()
    
    # Beta分布直方图和理论密度函数
    from scipy.stats import beta
    axes[2].hist(beta_samples, bins=50, density=True, alpha=0.6, label='取舍法生成样本')
    x = np.linspace(0, 1, 1000)
    axes[2].plot(x, beta.pdf(x, 2, 5), 'r-', label='理论密度函数')
    axes[2].set_title(f'Beta分布 Beta(2,5) - 接受率: {beta_rate:.4f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('rejection_sampling.png')
    plt.show()

if __name__ == "__main__":
    main()