class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        s = [0] * len(prices)
        b = [0] * len(prices)
        b[0] = -prices[0]
        for i in range(1, len(prices)):
            s[i] = max(s[i-1], prices[i]+b[i-1])
            
            if i >1:
                b[i] = max(s[i-2] - prices[i], b[i-1])
            else:
                b[i] = max(0-prices[i], b[i-1])
        
        return max(s[-1],b[-1])