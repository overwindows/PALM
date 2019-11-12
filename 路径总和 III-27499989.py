# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def _pathSum(self, root, sum, flag):
        cnt = 0
        
        if not root:
            return 0
        
        if root.val == sum:
            cnt = 1
        
        if not root.left and not root.right:
            cnt += 0

        if flag:
            cnt += self._pathSum(root.left,  sum-root.val, True)
            cnt += self._pathSum(root.right, sum-root.val, True)
        else:
            cnt += self._pathSum(root.left,  sum-root.val, True)
            cnt += self._pathSum(root.right, sum-root.val, True)
            cnt += self._pathSum(root.left,  sum, False)
            cnt += self._pathSum(root.right, sum, False)
        
        return cnt
        
    def pathSum(self, root: TreeNode, sum: int) -> int:
        cnt = 0
        if not root:
            return 0
        
        if root.val == sum:
            cnt = 1
        
        if not root.left and not root.right:
            cnt += 0
        
        cnt += self._pathSum(root.left,  sum-root.val, True)
        cnt += self._pathSum(root.right, sum-root.val, True)
        cnt += self._pathSum(root.left, sum, False)
        cnt += self._pathSum(root.right,sum, False)
        
        return cnt