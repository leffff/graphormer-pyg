def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x