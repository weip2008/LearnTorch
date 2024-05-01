class MyTensor:
    number = 1.2
    def __init__(self, a,ndim):
        self.a = a
        self.ndim = ndim

    def __repr__(self):
       return f'({self.a})'
        
    def ndimesion(self):
        return self.ndim


def tensor(a):
    print(a)

number = 10