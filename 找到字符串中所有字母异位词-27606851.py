class Solution:
    def myHash(self, lst):
        h = [0]*26
        for x in lst:
            h[ord(x)-ord('a')]+=1
        return h
        
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p=list(p)
        s=list(s)
        if len(s) == 0 or len(p) == 0 or len(p) > len(s):
            return []
        h_p = self.myHash(p)
        ret = []
        
        h_s = [0]*26
        for i in range(len(p)):
            h_s[ord(s[i])-ord('a')]+=1
        
        if h_s == h_p:
            ret.append(0)
            
        for i in range(1,len(s)-len(p)+1):
            h_s[ord(s[i-1])-ord('a')] -= 1
            h_s[ord(s[i+len(p)-1])-ord('a')] += 1

            if h_s == h_p:
                ret.append(i)
        return ret