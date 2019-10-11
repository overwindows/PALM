class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        len_haystack = len(haystack)
        len_needle = len(needle)
        
        if len_needle == 0:
            return 0
        
        if len_needle > len_haystack:
            return -1
        
        for i in range(len_haystack-len_needle+1):
            match = True
            ix = i
            for j in range(len_needle):
                if needle[j] == haystack[ix]:
                    ix += 1
                else:
                    match = False
                    break
            if match:
                return i
        
        return -1
        
                    
        