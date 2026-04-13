def add(a, b):
    return a + b

def multiply(x, y):
    result = 1
    if x > 0:
        result = x * y
        if y > 0:
            result *= 2
    return result

def process_data(data):
    for item in data:
        print(item)
