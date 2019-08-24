class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        if len(s) <= 1:
            return s
        st = 0
        ed = len(s)-1
        
        while st < ed:
            tmp = s[st]
            s[st] = s[ed]
            s[ed] = tmp
            st += 1
            ed -= 1
        return s