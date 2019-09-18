class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        len_s = len(s)
        len_t = len(t)
        ix = 0
        
        for i in range(len_s):
            found = False
            for j in range(ix, len_t):
                if s[i] == t[j]:
                    ix = j+1
                    found = True
                    break
                
            if not found:
                return False
        
        return True