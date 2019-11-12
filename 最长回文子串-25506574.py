class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        l_st = 0
        l_ed = 0
        n = len(s)
        mem = [[False]*n for _ in range(n)]
        for intvl in range(n):
            for st in range(n):
                if st + intvl == n:
                    break
                ed = st + intvl
                if (intvl == 0) or (intvl == 1 and s[st] == s[ed]) or (mem[st+1][ed-1] == True and s[st] == s[ed]):
                    mem[st][ed] = True
                    if intvl >= l_ed-l_st:
                        l_st = st
                        l_ed = ed           
        return s[l_st:l_ed+1]