class Solution:
    def firstUniqChar(self, s: str) -> int:
        if len(s) == 0:
            return -1
        if len(s) == 1:
            return 0
        
        d = [0]*26
        for c in s:
            d[ord(c)-ord('a')] += 1
        for i in range(len(s)):
            if d[ord(s[i])-ord('a')] == 1:
                return i
        return -1
