class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if len(coins) < 1:
            return -1
        if amount < 1:
            return 0
        if min(coins) > amount:
            return -1
        dp = [-1] * (amount+1)
        dp[0] = 0
        for c in coins:
            if c <= amount:
                dp[c] = 1
        for n in range(amount+1):
            for c in coins:
                if n > c and dp[n-c] != -1:
                    if dp[n] != -1:
                        dp[n] = min(dp[n],dp[n-c]+1)
                    else:
                        dp[n] = dp[n-c]+1
            #print(dp)
        
        return dp[amount]
        