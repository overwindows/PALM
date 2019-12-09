class Solution:
    def countBits(self, num: int) -> List[int]:
        dp = [0] * (num+1)
        n = 1
        for i in range(1,num+1):
            if i < 2 ** n:
                dp[i] = 1 + dp[i - 2**(n-1)]
            elif i == (2**n):
                dp[i] = 1
                n += 1
        
        return dp