# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        res = []
        
        stack = []
        stack.append(root)
        while stack:
            node = stack[-1]
            if not node.right and not node.left:
                res.append(node.val)
                stack.pop()
            
            if node.right:
                stack.append(node.right)
                node.right = None
            if node.left:
                stack.append(node.left)
                node.left = None
            
        return res
            