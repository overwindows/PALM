class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        
        lens1 = len(nums1)
        lens2 = len(nums2)
        
        s1 = 0
        s2 = 0
        
        dup = []
        
        while s1 < lens1 and s2 < lens2:
            if nums1[s1] == nums2[s2]:
                dup.append(nums1[s1])
                s1 += 1
                s2 += 1
            elif nums1[s1] > nums2[s2]:
                s2+=1
            elif nums1[s1] < nums2[s2]:
                s1 +=1
        
        return dup
            