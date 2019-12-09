# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        # 1 2 3 4 5 6 7
        # 1 2 3 4 5 6 7
        while True:
            a = (rand7()-1)*7 + rand7()
            if a <= 10:
                return a