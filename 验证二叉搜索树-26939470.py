# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deterBST(self,root):
        if not root.left and not root.right:
            return (True,root.val,root.val)

        if root.left:
            ret, min_left, max_left = self.deterBST(root.left)
            #print(ret, min_left, max_left, root.val)
            if not ret or root.val <= max_left or root.val <= root.left.val:
                return False,None,None

        if root.right:
            ret, min_right, max_right = self.deterBST(root.right)
            if not ret or root.val >= min_right or root.val >= root.right.val:
                return False,None,None

        if not root.right:
            return True, min_left, root.val

        if not root.left:
            return True, root.val, max_right

        return True, min_left, max_right
    
    def isValidBST(self, root: TreeNode) -> bool:
        if root:
            ret,_,_ = self.deterBST(root)
            return ret
        else:
            return True
        
        