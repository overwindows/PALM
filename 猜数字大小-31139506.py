# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        start = 1
        end = n
        
        while start < end:
            guess_n = (start+end) // 2
            ret = guess(guess_n)
            
            if ret == 0:
                return guess_n
            elif ret == 1:
                start = guess_n+1
            elif ret == -1:
                end = guess_n-1
        
        return start
        
        
        