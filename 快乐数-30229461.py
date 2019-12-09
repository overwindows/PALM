class Solution:
    def isHappy(self, n: int) -> bool:
        dup = set()
        
        while n not in dup:
            dup.add(n)
            val = 0
            while n:
                val += (n%10)**2
                n = n//10
            n = val
            if n==1:
                return True
        
        return False