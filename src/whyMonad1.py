class Failure():
    def __init__(self, value, failed=False): # box the value
        self.value = value
        self.failed = failed
    def get(self):
        return self.value
    def is_failed(self):
        return self.failed
    def __repr__(self):
        return "[" + ', '.join([str(self.value), str(self.failed)]) + "]"

    def bind(self, f):
        if self.failed:
            return self
        try:
            x = f(self.get())
            return Failure(x)
        except:
            return Failure(None, True)

    def __mul__(self, f):
        return self.bind(f)

    def __or__(self, f):
        return self.bind(f)

def neg(num):
    return -num

if __name__ == '__main__':
    x = 'a'
    y = Failure(x).bind(int)
    print(type(y))
    print(y)

    y = Failure(x).bind(int).bind(neg).bind(str)
    print(y)

    y = Failure(x) * int * str
    print(y)

    x = 'ZZZ'
    y = Failure(x).bind(int)
    print(type(y))
    print(y)

    if y.failed: # check for failure first
        print(f"Could not convert the value {x}.")
    else:
        print(y.value) # unbox the value

    