class Solution:
    def myPow(self, x: float, n: int) -> float:
        
        if n == 0:
            return 1
        
        res = 1.0
        r = []
        
        d = abs(n)
        
        while d !=1 :
            r.append(d%2)
            d = d//2
        r.append(1)
        r.reverse()
        # print(r)
        '''
        for i in range(abs(n)):
            res *= x
        '''
        for i in range(len(r)):
            res *= res
            if r[i] == 1:
                res *= x
        
        if n < 0:
            return 1.0/res
        else:
            return res
        