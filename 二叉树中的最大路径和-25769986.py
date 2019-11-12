# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def calcPath(self, root):
        if root.left is None and root.right == None:
            # local_max_sum, max(left_sum, right_sum)
            return (root.val, root.val)
        
        l_v = -1000000000
        r_v = -1000000000
        max_l_v = -1000000000
        max_r_v = -1000000000
        
        local_sum = root.val
        local_lt  = root.val
        local_rt  = root.val
       
        if root.left:
            l_v, max_l_v = self.calcPath(root.left)
            local_sum += max_l_v
            local_lt += max_l_v
            #local_sum = max(local_sum, l_v)
        
        if root.right:
            r_v, max_r_v = self.calcPath(root.right)
            local_sum += max_r_v
            local_rt += max_r_v
            #local_sum = max(local_sum, r_v)
        
        local_max_sum = max(local_sum, local_lt, local_rt, l_v, r_v, root.val)
        _max_sum = max(local_rt, local_lt, root.val)
        
        return (local_max_sum, _max_sum)
    
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        local_max_sum, _ = self.calcPath(root)
        
        return local_max_sum