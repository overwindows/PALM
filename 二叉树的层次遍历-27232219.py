# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        queue = []
        queue.append(root)
        queue.append('SEP')
        
        ret = []
        res = []
        
        if not root:
            return ret
        
        while queue:
            x = queue.pop(0)
            if x == 'SEP':
                if len(res) > 0:
                    ret.append(res)
                    res = []
                    queue.append('SEP')
            else:
                res.append(x.val)
                if x.left:
                    queue.append(x.left)
                if x.right:
                    queue.append(x.right)
        
        return ret
            
            