import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def load_bacterial_data(file_path):
    """
    从文件中加载细菌生长实验数据

    参数:
        file_path (str): 数据文件路径

    返回:
        tuple: 包含时间和酶活性测量值的元组
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    try:
        # 尝试使用逗号作为分隔符加载数据
        data = np.loadtxt(file_path, delimiter=',')
    except ValueError:
        # 如果逗号分隔符失败，尝试使用空格分隔符
        try:
            data = np.loadtxt(file_path)
        except ValueError as e:
            # 如果仍然失败，尝试更灵活的加载方式
            print(f"使用标准方法加载失败: {e}")
            print("尝试使用更灵活的加载方式...")

            # 读取原始数据
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 处理每行数据
            processed_data = []
            for line in lines:
                # 跳过空行
                if not line.strip():
                    continue

                # 分割行数据，处理各种可能的分隔符
                parts = line.strip().replace(',', ' ').split()

                # 确保每行有两个数值
                if len(parts) >= 2:
                    try:
                        # 转换为浮点数
                        processed_data.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        # 忽略无法转换的行
                        continue

            # 转换为numpy数组
            if processed_data:
                data = np.array(processed_data)
            else:
                raise ValueError("无法从文件中提取有效数据")

    # 提取时间和活性数据
    t = data[:, 0]
    activity = data[:, 1]
    return t, activity


def V_model(t, tau):
    """
    V(t)模型函数

    参数:
        t (float or numpy.ndarray): 时间
        tau (float): 时间常数

    返回:
        float or numpy.ndarray: V(t)模型值
    """
    return 1 - np.exp(-t / tau)


def W_model(t, A, tau):
    """
    W(t)模型函数

    参数:
        t (float or numpy.ndarray): 时间
        A (float): 比例系数
        tau (float): 时间常数

    返回:
        float or numpy.ndarray: W(t)模型值
    """
    return A * (np.exp(-t / tau) - 1 + t / tau)


def fit_model(t, data, model_func, p0):
    """
    使用curve_fit拟合模型

    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        p0 (list): 初始参数猜测

    返回:
        tuple: 拟合参数及其协方差矩阵
    """
    popt, pcov = curve_fit(model_func, t, data, p0=p0)
    return popt, pcov


def plot_results(t, data, model_func, popt, title):
    """
    绘制实验数据与拟合曲线

    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        popt (numpy.ndarray): 拟合参数
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(t, data, label='Experimental Data', color='blue', alpha=0.6)

    t_fit = np.linspace(min(t), max(t), 500)
    plt.plot(t_fit, model_func(t_fit, *popt), 'r-',
             label=f'Fit: {model_func.__name__} (τ={popt[-1]:.3f})')

    plt.title(title, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Enzyme Activity', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载数据
    data_dir = "D:\PYTHON\pythonProject\计算物理\细菌"  # 请替换为你的数据目录
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")

    # 拟合V(t)模型
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0])
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f}")

    # 拟合W(t)模型
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0])
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f}, τ = {popt_W[1]:.3f}")

    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, 'V(t) Model Fit')
    plot_results(t_W, W_data, W_model, popt_W, 'W(t) Model Fit')
