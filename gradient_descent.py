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

def apply_gradient_descent(x0, f_function, eta, num_steps):
    x=x0
    for i in range(num_steps):
        x = x - differentiate(f_function, x) * eta
        # print("x=%.3f, f=%.3f" % (x.numpy(), f_function(x).numpy())) # show progress
    return x

# test
# def f_function(x):
#    return x*x
# x0=tf.constant(10.0, tf.float32)
# x=apply_gradient_descent(x0, f_function, eta=0.1, num_steps=100)
# print("Optimized x: ", x.numpy())



