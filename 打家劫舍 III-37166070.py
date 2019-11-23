# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root: TreeNode) -> int:
        
        def _rob(root: TreeNode):
            if not root:
                return 0,0
            ly,ln = _rob(root.left)
            ry,rn = _rob(root.right)
            
            _rob_y = root.val
            _rob_n = 0

            return _rob_y+ln+rn, _rob_n+max(ln,ly)+max(ry,rn)

        return max(_rob(root))
        '''
        def _rob(root: TreeNode, flag: bool) -> int:
            if not root:
                return 0
            rob = 0
            if flag:
                rob += root.val
                rob += _rob(root.left, False)
                rob += _rob(root.right, False)
            else:
                rob += max(_rob(root.left, True), _rob(root.left, False))
                rob += max(_rob(root.right, True), _rob(root.right, False))
            return rob

        max_rob = max(_rob(root, True), _rob(root, False))

        return max_rob
        '''
