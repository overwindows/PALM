class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        neg = False
        if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
            neg = True
        
        ret =  abs(dividend)//abs(divisor)
        
        if neg:
            return ret*-1
        else:
            return min(ret,2**31-1)