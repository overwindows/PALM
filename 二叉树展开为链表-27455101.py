# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return 
        
        if root.left:
            self.flatten(root.left)
        
        if root.right:
            self.flatten(root.right)
        
        if root.left and root.right:
            max_lf = root.left
            while max_lf.right:
                max_lf = max_lf.right
            max_lf.right = root.right
            root.right = root.left
            root.left = None
        
        if root.left:
            root.right = root.left
            root.left = None
        
        if root.right:
            root.right = root.right