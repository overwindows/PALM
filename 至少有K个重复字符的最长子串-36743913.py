class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if k < 2:
            return len(s)
        ch = [0]*26
        for c in s:
            ch[ord(c)-ord('a')] += 1
        for i in range(26):
            if ch[i] < k:
                s=s.replace(chr(i+97),'#')
        subs = s.split('#')
        if len(subs) == 1:
            return len(subs[0])
        l = 0
        for sub_s in subs:
            if sub_s:
                l = max(self.longestSubstring(sub_s, k),l)
        
        return l



