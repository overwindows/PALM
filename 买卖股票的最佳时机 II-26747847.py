class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """ 
        if len(prices) < 2:
            return 0
        
        maxProf = 0
        for i in range(1,len(prices)):
            maxProf += max(prices[i]-prices[i-1],0) 
        
        return maxProf
        
        '''                                      
        st = 0
        if prices[0] >= prices[1]:
            for i in range(1, len(prices)):
                if prices[i] <= prices[st]:
                    st = i
                else:
                    break
        
        pivot = prices[st]
        max_profit = 0

        for j in range(st+1, len(prices)):
            #sell
            if prices[j] > prices[st]:
                cur_profit = prices[j] - prices[st]
                left_profit = self.maxProfit(prices[j+1:len(prices)])
                
                profit = cur_profit + left_profit
                max_profit = max(max_profit, profit)
        '''
        return max_profit