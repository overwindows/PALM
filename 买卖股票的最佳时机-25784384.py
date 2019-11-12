class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0:
            return 0
        _max = prices[0]
        _min = prices[0]
        
        max_profit = 0
        
        for price in prices:
            if price > _min:
                max_profit = max(price-_min, max_profit)
            
            if price < _min:
                _min = price
                
        return max_profit                