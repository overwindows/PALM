# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return root
        if root.val == p.val:
            return root
        if root.val == q.val:
            return root
        
        if root.left and root.right:
            ret_left = self.lowestCommonAncestor(root.left, p, q)
            ret_right = self.lowestCommonAncestor(root.right, p, q)
            
            if ret_left == None:
                if ret_right == None:
                    return None
                else:
                    return ret_right
            else:
                if ret_right == None:
                    return ret_left
            
            if (ret_left.val == p.val and ret_right.val == q.val) or (ret_left.val == q.val and ret_right.val == p.val):
                return root
        elif root.left is None:
            ret_right = self.lowestCommonAncestor(root.right, p, q)
            return ret_right
        elif root.right is None:
            ret_left = self.lowestCommonAncestor(root.left, p, q)
            return ret_left
        else:
            return None
        