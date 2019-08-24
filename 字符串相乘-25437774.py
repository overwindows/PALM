class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0':
            return '0'
        lst_num1 = list(num1)
        lst_num2 = list(num2)
        
        lst_num1.reverse()
        lst_num2.reverse()
        
        _sum = []
        _res = []        
        # [6,5,4]
        for n1 in lst_num1:
            #[3,2,1]
            _mul = []
            add = 0
            for n2 in lst_num2:
                m = int(n1)*int(n2) + add
                add = m //10
                m = m%10
                _mul.append(m)
            if add > 0:
                _mul.append(add)
            _sum.append(_mul)
        #print(_sum)
        rows = len(_sum)
        cols = 0
        for r in range(rows):
            for _ in range(r):
                _sum[r].insert(0,0)
            cols = max(cols, len(_sum[r]))
        # print(_sum, cols)
        add = 0
        for c in range(cols):
            __sum = 0
            for r in range(rows):
                if len(_sum[r]) > c:
                    __sum += _sum[r][c]
            __sum += add
            add = __sum // 10
            __sum = __sum % 10
            _res.append(__sum)
        if add > 0:
            _res.append(add)
        _res.reverse()
        
        return ''.join(map(str,_res))
                
                
            
        
                
                