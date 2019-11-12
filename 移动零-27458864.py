class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)):
            while nums[i] == 0:
                flag = 0
                for j in range(i,len(nums)-1):
                    nums[j] = nums[j+1]
                    flag |= nums[j+1]
                if flag == 0:
                    return
                #print(nums[i])
                nums[len(nums)-1] = 0
