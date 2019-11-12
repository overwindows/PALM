# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    s = 0
    def convertBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        
        if not root.right and not root.left:
            root.val += self.s
            self.s = root.val
            return root
        
        self.convertBST(root.right)
        root.val += self.s
        self.s = root.val
        self.convertBST(root.left)
        
        return root
        
        