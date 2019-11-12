# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root.val == p.val:
            return root
        
        if root.val == q.val:
            return root
        
        if p.val < q.val:
            if root.val > p.val and root.val < q.val:
                return root
            if p.val > root.val:
                return self.lowestCommonAncestor(root.right, p, q)
            if q.val < root.val:
                return self.lowestCommonAncestor(root.left, p, q)
        else:
            if root.val < p.val and root.val > q.val:
                return root
            if p.val < root.val:
                return self.lowestCommonAncestor(root.left, p, q)
            if q.val > root.val:
                return self.lowestCommonAncestor(root.right, p, q)
           
            