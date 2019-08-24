class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        _s = s
        prev_max_len = 0
        max_len = 0
        i= -1
        skip = 0
        while i < len(s):
            prev_max_len = max(max_len, prev_max_len)
            step = 1
            if skip > 0:
                #print(skip)
                i = skip
                skip = 0
            else:
                i += step
            _s = s[i:i+256]
            
            max_len = 0
            max_substr = [0]*256
            ix = 0
            #print(_s)
            for c in _s:
                ix += 1
                if max_substr[ord(c)] == 0:
                    max_substr[ord(c)] = ix + i
                    max_len += 1
                else:
                    skip = max_substr[ord(c)]
                    #print(c, ix,i, max_substr[ord(c)])
                    max_substr = [0]*256
                    prev_max_len = max(max_len, prev_max_len)
                    break
                    #max_substr[ord(c)] = ix + i
                    #max_len = 1
        
        return max(max_len, prev_max_len)
                
            