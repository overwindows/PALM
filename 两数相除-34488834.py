class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        neg = False
        if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
            neg = True
    
        # ret =  abs(dividend)//abs(divisor)
            
        dividend = abs(dividend)
        divisor = abs(divisor)
        
        cnt = 0
        
        div = []
        fac = []
        i = 0
        while dividend >= divisor:
            div.append(divisor)
            divisor += divisor
            if i == 0:
                fac.append(1)
            else:
                fac.append(fac[-1]+fac[-1])
            i += 1
        
        #print(fac)
        
        if divisor == 1:
            ret = dividend
        else:
            #print(div, dividend)
            len_div = len(div)
            for i in range(len_div):
                if dividend >= div[len_div-1-i]:
                    cnt += fac[len_div-1-i]
                    dividend -= div[len_div-1-i]

            '''    
            while dividend >= divisor:
                dividend -= divisor
                cnt += 1
            '''
            ret = cnt
        
        if neg:
            return ret*-1
        else:
            return min(ret,2**31-1)