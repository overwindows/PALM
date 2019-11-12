class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        cnt = 1
        while n&(n-1):
            cnt += 1
            n = n&(n-1)
        
        return cnt