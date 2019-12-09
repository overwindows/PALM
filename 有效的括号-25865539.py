class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        lst = list(s)
        
        for s in lst:
            if len(stack) > 0:
                top = stack[-1]
                if (top == '(' and s==')') or (top == '{' and s == '}') or (top == '[' and s == ']'):
                    stack.pop()
                    continue
                else:
                    stack.append(s)
            else:
                stack.append(s)
        
        if len(stack) == 0:
            return True
        else:
            return False