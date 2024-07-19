"""
Create a range1 class which iterate start from index 1, include stop.
"""
class range1:
    def __init__(self, *args):
        self.start = 1
        self.step = 1
        if len(args)==1:
            self.stop = args[0]
        if len(args)==2:
            self.start = args[0]
            self.stop = args[1]
        if len(args)==3:
            self.start = args[0]
            self.stop = args[1]
            self.step = args[2]
        self.current = self.start
    
    def __next__(self):
        if self.current > self.stop:
            raise StopIteration
        result = self.current
        self.current += self.step
        return result
            

    def __iter__(self):
        self.current = self.start
        return self

if __name__ == '__main__':
    r = range1(10)
    i = iter(r)
    print(next(i))
    print(next(i))
    print()
    for i in r:
        print(i)
    print()
    r = range1(5, 10)
    for i in r:
        print(i)
    print()
    r = range1(5, 10, 2)
    for i in r:
        print(i)
