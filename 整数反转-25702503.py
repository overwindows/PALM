class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 0:
            factor = -1.0
        else:
            factor = 1.0
            
        abs_x = abs(x)
        lst = list(str(abs_x))
        
        l = len(lst)
        
        st = 0
        ed = l-1
        
        while st < ed:
            tmp = lst[st]
            lst[st] = lst[ed]
            lst[ed] = tmp
            st += 1
            ed -= 1
        
        val = int(int(''.join(lst)) * factor)
        
        if val >= -2**31 and val < 2**31:
            return val
        else:
            return 0
            
        
        