class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        strs = s.split()
        
        if len(strs) == 0:
            return ''
        
        
        st = 0
        ed = len(strs)-1
        
        while st < ed:
            tmp = strs[st]
            strs[st] = strs[ed]
            strs[ed] = tmp
            st += 1
            ed -= 1
        new_s = list(' '.join(strs))
        
        st = 0
        ed = len(new_s)-1
        
        while st < ed:
            tmp = new_s[st]
            new_s[st] = new_s[ed]
            new_s[ed] = tmp
            st += 1
            ed -= 1
        
        return ''.join(new_s)