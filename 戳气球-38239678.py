class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # mem = {}

        while 0 in nums:
            nums.remove(0)
        
        N = len(nums)
        if N==0:
            return 0
        dp = [[0]*N for _ in range(N)]
        for i in range(N):
            dp[i][i] = nums[i]
        
        for n in range(N):
            for i in range(N-n):
            #for j in range(j,N,n):
                j = i+n
                #print(i,j)
                for k in range(i,j+1):
                    a = nums[k]
                    if j+1 < N:
                        a*=nums[j+1]
                    if i > 0:
                        a*=nums[i-1]
                    b = a

                    if k > i:
                        b += dp[i][k-1]
                    if k < j:
                        #print(k)
                        b += dp[k+1][j]
                    dp[i][j] = max(dp[i][j], b)
    
        return dp[0][N-1]
        '''
        N = len(nums)
        tbl = [None for _ in range(N)]
        tbl[0] = {}
        for i in range(N):
            tbl[0][str(i)] = nums[i]
        #print(tbl)
        for i in range(1,N):
            #print(tbl)
            tbl[i] = {}
            for j in range(i,N):
                for k,v in tbl[i-1].items():
                    key = list(map(int,k.split('|')))
                    if j > key[-1]:
                        key.append(j)
                        _k = '|'.join(list(map(str,key)))
                        tbl[i][_k] = 0
                        for n in range(len(key)):
                            _key = key[:]
                            del _key[n]
                            __k = '|'.join(list(map(str,_key)))
                            if n == 0:
                                tbl[i][_k] = max(tbl[i][_k], tbl[i-1][__k]+nums[key[n]]*nums[key[n+1]])
                            elif n == len(key)-1:
                                tbl[i][_k] = max(tbl[i][_k], tbl[i-1][__k]+nums[key[n]]*nums[key[n-1]])
                            else:
                                tbl[i][_k] = max(tbl[i][_k], tbl[i-1][__k]+nums[key[n]]*nums[key[n+1]]*nums[key[n-1]])
        return list(tbl[N-1].values())[0]
        '''

        '''
        def myMaxCoins(nums: List[int], mem) -> int:
            lens = len(nums)
            maxCoins = 0

            if lens == 2:
                return max(nums) + nums[0]*nums[1]
            if lens == 3:
                return max(nums[0]*nums[1]*nums[2] + nums[2]*nums[0] + max(nums[0],nums[2]), nums[0]*nums[1]+nums[1]*nums[2]+max(nums))
            
            nums = [x for x in nums if x >0]
            print(nums)

            for i in range(lens):
                if i == 0:
                    left = 1
                else:
                    left = nums[i-1]
                
                if i == lens-1:
                    right = 1
                else:
                    right = nums[i+1]
                _maxCoins = nums[i]*left*right
                _nums = nums[:]
                del _nums[i]

                key = "#".join(list(map(str, _nums)))
                if key not in mem:
                    mem[key] = myMaxCoins(_nums, mem)
                _maxCoins += mem[key]
                maxCoins = max(maxCoins,_maxCoins)
            return maxCoins

        max_coins = myMaxCoins(nums,mem)
        #print(mem)
        return max_coins
        '''

'''
[8,2,6,8,9,8,1,4,1,5,3,0,7,7,0,4,2,2,5]
[8,2,6,8,9,8,1,4,1,5,3,0,7,7,0,4,2]
[35,16,83,87,84,59,48,41,20,54]
[3,1,5,8]
'''