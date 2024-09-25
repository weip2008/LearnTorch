"""
different function use different way to generate different dataset
using func_dict dictionary is easy to switch to different data generate method 
"""
def func1():
    """
    generate dataset for training Fix len,
    """
    print("func1()...")

def func2():
    print("func2()...")

def func3():
    print("func3()...")

if __name__ == "__main__":
    func_dict = {1:func1, 2:func2, 3:func3}

    func = func_dict.get(3)
    func()

    func = func_dict.get(5, func1) # use func1 as default function
    func()