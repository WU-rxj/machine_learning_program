# 随机模拟方法
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK JP']
def simulate_rain(n, m, num_simulations=100000):
    """
    模拟n天假期中连续m天下雨的概率
    
    参数:
    n: 假期总天数
    m: 连续下雨天数
    num_simulations: 模拟次数
    
    返回:
    连续m天下雨的概率
    """
    count = 0
    for _ in range(num_simulations):
        # 生成n天的天气情况，True表示下雨，False表示不下雨
        weather = np.random.random(n) < 0.5
        
        # 检查是否有连续m天下雨
        has_consecutive_rainy_days = False
        for i in range(n - m + 1):
            if all(weather[i:i+m]):
                has_consecutive_rainy_days = True
                break
        
        if has_consecutive_rainy_days:
            count += 1
    
    return count / num_simulations

# 测试不同假期长度和连续下雨天数的情况
def main():
    # 固定m=3，变化n
    n_values = list(range(3, 21))
    probabilities = []
    
    for n in n_values:
        prob = simulate_rain(n, 3)
        probabilities.append(prob)
        print(f"假期长度为{n}天，连续3天下雨的概率约为: {prob:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, probabilities, marker='o')
    plt.title('假期长度与连续3天下雨概率的关系')
    plt.xlabel('假期长度(天)')
    plt.ylabel('连续3天下雨的概率')
    plt.grid(True)
    plt.savefig('rain_probability.png')
    plt.show()
    
    # 固定n=10，变化m
    m_values = list(range(1, 11))
    probabilities = []
    
    for m in m_values:
        prob = simulate_rain(10, m)
        probabilities.append(prob)
        print(f"假期长度为10天，连续{m}天下雨的概率约为: {prob:.4f}")
    
    # 绘制概率变化图
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, probabilities, marker='o', color='red')
    plt.title('连续下雨天数与概率的关系(假期长度为10天)')
    plt.xlabel('连续下雨天数')
    plt.ylabel('概率')
    plt.grid(True)
    plt.savefig('consecutive_rain_probability.png')
    plt.show()

if __name__ == "__main__":
    main()