class Solution:
    def numDecodings(self, s: str) -> int:
        # f(n-1),f(n-2)
        s = list(map(int,s))
        len_s = len(s)
        dp = [0] * len_s
        
        if len_s < 1 or s[0] == 0:
            return 0
        
        if len_s == 1:
            if s[0] > 0:
                return 1
            else:
                return 0
                
        dp[0] = 1
        # print(s)
        if s[1] > 0:
            if s[0]*10+s[1] < 27: 
                dp[1] = 2
            else:
                dp[1] = 1
        else:
            if s[1]+s[0]*10 > 26:
                return 0
            
            if len_s == 2:
                return 1
            else:
                dp[1] = dp[0]
        
        for i in range(2,len_s):
            if s[i] == 0:
                if s[i-1] == 0:
                    return 0
                if s[i]+s[i-1]*10 < 27:
                    dp[i] = dp[i-2]
                    continue
                else:
                    return 0
                           
            
            
            if s[i-1] > 0 and s[i]+s[i-1]*10 < 27:
                dp[i] = dp[i-1] + dp[i-2]
            else:
                dp[i] = dp[i-1]
        
        return dp[-1]
            
        
    
    