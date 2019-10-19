class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = len(nums1)
        len2 = len(nums2)
        
        nums1.extend(nums2)
        nums1.sort()
        
        if len(nums1) %2 :
            return nums1[len(nums1)//2]
        else:
            return (nums1[len(nums1)//2]+nums1[len(nums1)//2-1]) / 2.0