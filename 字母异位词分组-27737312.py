class Solution:
    def hash(self,s):
        h = [0] * 26
        for c in s:
            h[ord(c)-ord('a')] += 1
        return '#'.join(map(str,h))
            
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = {}
        for s in strs:
            key = self.hash(s)
            val = s
            if key not in d:
                d[key] = []
            d[key].append(val)
        
        return d.values()
        
                
                