# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """   
        stack = []
        stack.append(root)
        while stack:
            node = stack[-1]
            if node:
                while node.left :
                    node = node.left
                    stack.append(node)
            else:
                node = stack.pop()
            
            if not stack:
                break
            
            node = stack.pop()
            
            if k == 1:
                return node.val
            else:
                k-=1
            
            if node.right:
                stack.append(node.right)
            else:
                stack.append(None)
