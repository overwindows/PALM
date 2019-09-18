class Solution:
    def mySqrt(self, x: int) -> int:
        st = 1
        ed = x
        while st < ed:
            mid = (st+ed)//2
            if mid**2 > x:
                ed = mid-1
            elif mid**2 < x:
                st = mid+1
            else:
                return mid
        
        if st**2 > x:
            return st-1
        else:
            return st

        '''
        s = 1
        if x == 1:
            return 1
        for i in range(x):
            if i**2 >= x:
                s = i
                break
        
        if s**2 <= x:
            return s
        else:
            return s-1
        '''