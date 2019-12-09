class Solution:
    def countSubstrings(self, s: str) -> int:
        if len(s) == 0:
            return 0
        if len(s) == 1:
            return 1
        m = [[1] for _ in range(len(s))]
        s = list(s)

        for i in range(len(s)):
            if i > 0:
                for j in range(len(m[i-1])):
                    l = m[i-1][j]
                    if i-l >0 and s[i] == s[i-l-1]:
                        m[i].append(l+2)
                if s[i] == s[i-1]:
                    m[i].append(2)
        
        #print(m)
        cnt = 0
        for lst in m:
            cnt += len(lst)
        
        return cnt