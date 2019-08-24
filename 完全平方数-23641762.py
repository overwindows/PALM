class Solution(object):
    def numSquares(self, n):
        _m = [n]*(n+1)
        for k in range(1, n+1):
            if k*k > n:
                break
            _m[k*k] = 1
        
        if _m[n] == 1:
            return _m[n]
        
        for i in range(2,n+1):
            if _m[i] == 1:
                continue
            for ii in range(1, int((i+1)**0.5)):
                #if ii*ii > i:
                #    break
                _m[i] = min(_m[i], _m[ii*ii]+_m[i-ii*ii])   
        return _m[n]