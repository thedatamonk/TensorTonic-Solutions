
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def y(x):
        return a*(x**2) + b*x + c

    def dy(x):
        return 2*a*x + b

    # Write code here
    i = 0
    x = x0
    while i < steps:
        x = x - lr * dy(x)
        i+=1

    return x

    