class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        
        n1 = m-1
        n2 = n-1
        
        tail = m+n-1
        
        while n1 > -1 and n2 > -1:
            if nums1[n1] > nums2[n2]:
                nums1[tail] = nums1[n1]
                n1 -= 1
            else:
                nums1[tail] = nums2[n2]
                n2 -= 1
            tail -= 1
        
        while n2 > -1:
            nums1[tail] = nums2[n2]
            n2 -= 1
            tail -= 1
        
            