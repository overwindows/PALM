class Solution:
    def bulbSwitch(self, n: int) -> int:
        cnt = 0
        
        if n == 1:
            return 1
        end = int(n ** 0.5) + 1
        
        for i in range(1,end+1):
            if i*i <= n:
                cnt += 1
        
        return cnt
        