# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        if not root.left and not root.right:
            return 1
        
        lf_dep = 1
        rt_dep = 1
        
        if root.left:
            lf_dep += self.maxDepth(root.left)
        
        if root.right:
            rt_dep += self.maxDepth(root.right)
            
        return max(lf_dep, rt_dep)