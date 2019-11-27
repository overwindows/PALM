class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        stack = []
        while n:
            stack.append(n%2)
            n = n//2
        lens = len(stack)
        for _ in range(32-lens):
            stack.append(0)
        #print(stack)
        ret = 0
        for i in range(32):
            ret += 2**i * stack.pop()
        return ret