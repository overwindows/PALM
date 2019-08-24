class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        
        lst = list(str(x))
        
        st = 0
        ed = len(lst) - 1
        
        while lst[st] == lst[ed] and st < ed:
            st += 1
            ed -= 1
        
        if st >= ed:
            return True
        else:
            return False
            
        