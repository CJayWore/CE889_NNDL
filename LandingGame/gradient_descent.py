import numpy as np

def differentiate(f_function, x, h=1e-5):
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        x_step = np.array(x, dtype=np.float64)  # 复制 x 避免修改原始值
        x_step[i] += h
        f_x_h = f_function(x_step)  # f(x + h)
        x_step[i] -= 2 * h
        f_x_minus_h = f_function(x_step)  # f(x - h)
        gradient[i] = (f_x_h - f_x_minus_h) / (2 * h)  # 中心差分公式
    return gradient

def apply_gradient_descent(x0, f_function, eta, num_steps, min_grad=1e-5):
    x = x0
    for i in range(num_steps):
        grad = differentiate(f_function, x)
        if np.linalg.norm(grad) < min_grad:
            break
        x = x - grad * eta
        # print("x=%.3f, f=%.3f" % (x, f_function(x))) # show progress
    return x

# Test
# def test_function(x):
#     return np.sum(x**2)
#
# x0 = np.array([2.0, 3.0])  # 初始点
# eta = 0.1
# num_steps = 100
#
# result = apply_gradient_descent(x0, test_function, eta, num_steps)
#
# print("优化后的结果:", result)
# print("函数值:", test_function(result))




