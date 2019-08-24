class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        nums = len(strs)
        min_len = -1
        
        for s in strs:
            if min_len > -1:
                min_len = min(min_len, len(s))
            else:
                min_len = len(s)
        
        pre = []
        
        for i in range(min_len):
            _dup = None
            j = 0
            for _ in range(nums):
                if _dup and _dup==strs[j][i]:
                    j+=1
                elif _dup is None:
                    _dup = strs[j][i]
                    j+=1
                else:
                    break
            if j == nums:
                pre.append(_dup)
            else:
                break
         
        return ''.join(pre)
