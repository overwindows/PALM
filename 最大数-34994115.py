import functools
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        if len(nums) == 0:
            return 0
        if len(nums) ==1:
            return ''.join(map(str,nums))
        
        all_zeros = True
        for n in nums:
            if n > 0:
                all_zeros = False
                break
        if all_zeros:
            return '0'
        
        def mycmp(x,y):
            x = x.copy()
            y = y.copy()
            len_x = len(x)
            len_y = len(y)
            #lens = max(len_x, len_y)
            if len_x < len_y:
                for _ in range(len_y-len_x):
                    x.append(x[0])
                if x[0] == 1:
                    x[-1] = 2
            if len_x > len_y:
                for _ in range(len_x-len_y):
                    y.append(y[0])
                if y[0] == 1:
                    y[-1] = 2
            
            lens = max(len_x,len_y)
            for i in range(lens):
                if x[i] > y[i]:
                    return -1          
                elif x[i] < y[i]:
                    return 0
                else: #x[i] == y[i]
                    continue
            if len_x > len_y:
                return -1
            else:
                return 0
        a = []
        for num in nums:
            a.append(list(map(int, list(str(num)))))
        #a.reverse()
        a.sort(key=functools.cmp_to_key(mycmp))
        
        num = []
        for x in a:
            for y in x:
                num.append(y)
        return ''.join(map(str,num))