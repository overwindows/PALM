# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        if not root.right and not root.left:
            return root
        
        if root.right:
            invert_right = self.invertTree(root.right)
        else:
            invert_right = None
        
        if root.left:
            invert_left =self.invertTree(root.left)
        else:
            invert_left = None
        
        root.left = invert_right
        root.right = invert_left
            
        return root
    