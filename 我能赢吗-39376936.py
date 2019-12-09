class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        if desiredTotal <= maxChoosableInteger:
            return True
        nums = [x for x in range(1, maxChoosableInteger+1)]
        #print(sum(nums))
        if sum(nums) < desiredTotal:
            return False
        mem = {}

        def _canWin(desired, choosableInt) -> bool:
            if len(choosableInt) == 1:
                return desired <= choosableInt[0]

            for x in choosableInt:
                new_choosableInt = choosableInt[:]
                new_choosableInt.remove(x)
                new_desired = desired-x
                sign = '#'.join(list(map(str,new_choosableInt)))
                sign = str(new_desired)+'|'+sign
                
                if new_desired <=0 :
                    return True
                else:
                    if sign not in mem:
                        mem[sign] =  _canWin(new_desired, new_choosableInt)
                    if not mem[sign]:
                        return True
            
            return False
        
        ret = _canWin(desiredTotal, nums)
        #print(mem)
        return ret 

'''
10
11
15
100
4
6
18
188
'''