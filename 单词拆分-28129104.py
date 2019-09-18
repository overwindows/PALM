class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        s = list(s)
        len_s = len(s)
        if len_s == 0:
            return True
        dp = [False] * (len_s)
        for i in range(len_s):
            #print(dp)
            for word in wordDict:
                w_len = len(word)
                if i < w_len-1:
                    continue
                
                if i+1 == w_len and list(word) == s[:w_len]:
                    dp[i] = True
                    break
                #print(i, i-w_len+1, word, s[i-w_len+1:i+1])
                if i+1 > w_len and dp[i-w_len] and list(word) == s[i-w_len+1:i+1]:
                    dp[i] = True
                    break
        
        return dp[len_s-1]
                
    