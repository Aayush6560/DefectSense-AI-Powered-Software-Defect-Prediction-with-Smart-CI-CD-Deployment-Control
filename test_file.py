def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_product(numbers):
    result = 1
    for num in numbers:
        if num != 0:
            result *= num
    return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        return a + b
