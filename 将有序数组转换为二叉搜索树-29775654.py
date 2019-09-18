# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        len_nums = len(nums)
        if len_nums == 0:
            return None
        if len_nums == 1:
            return TreeNode(nums[0])
        
        mid = len_nums//2
        root = TreeNode(nums[mid])
        left = self.sortedArrayToBST(nums[:mid])
        right = self.sortedArrayToBST(nums[mid+1:])
        
        root.left = left
        root.right = right
        
        return root