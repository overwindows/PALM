# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        
        stack = []
        stack.append(root)
        ret = []
        
        while stack:
            node = stack[-1]
            while node.left:
                stack.append(node.left)
                node.left = None
                node = stack[-1]
            node = stack.pop()
            ret.append(node.val)
            if node.right:
                stack.append(node.right)
        return ret   
          