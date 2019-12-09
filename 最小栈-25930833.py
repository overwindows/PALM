class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.Min = []
        
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if len(self.Min) == 0 or x <= self.Min[-1]:
            self.Min.append(x)

    def pop(self):
        """
        :rtype: None
        """
        x = self.stack.pop()
        if x == self.Min[-1]:
            self.Min.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]        

    def getMin(self):
        """
        :rtype: int
        """
        return  self.Min[-1]
        

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()