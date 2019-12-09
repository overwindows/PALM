class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = 0
        
        d = {}
        sub = []
        subSum = 0
        i = 0
        for num in nums:
            subSum += num
            sub.append(subSum)
            if subSum in d:
                d[subSum].append(i)
            else:
                d[subSum] = [i]
            if subSum == k:
                cnt += 1
            i += 1
        #print(sub,cnt)
        size = len(nums)
        
        for j in range(size):
            if sub[j]+k in d:
                for ix in d[sub[j]+k]:          
                    if j < ix:
                        cnt += 1
                
        '''
        for i in range(size-1):
            for j in range(i+1,size):
                if (sub[j]-sub[i]) == k:
                    cnt+=1
        '''
        
        return cnt