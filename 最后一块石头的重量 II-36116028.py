class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        if len(stones) == 0:
            return 0
        if len(stones) == 1:
            return stones[0]
        if len(stones) == 2:
            return abs(stones[1]-stones[0])
            
        quota = sum(stones)//2 + sum(stones)%2
        dp = [[0]*len(stones) for _ in range(15001)]
        #dp = [[0]*len(stones) for _ in range(quota+1)]
         
        for q in range(quota+1):
            for i in range(len(stones)):
                if stones[i] > q:
                    continue
                s = stones[i]
                dp[q][i] = s
                for j in range(i):
                    if(q-dp[q-s][j]+s) >= 0 and dp[q-s][j]+s > dp[q][i]:
                        dp[q][i] = dp[q-s][j] + s
            
            #print(q,dp[q])
        #print(quota,dp[quota],sum(stones))
        gap = abs(2*max(dp[quota]) - sum(stones))
        return gap
        
        
        '''
        mem = {} 
        
        for s in stones:
            mem[(s)] = s
        
        def _lastStoneWeight(stones,mem):
            #if False:
            if tuple(stones) in mem:
                #print('Bingo', stones)
                return mem[tuple(stones)]
            else:
                min_last = sum(stones)
                for i in range(len(stones)):
                    for j in range(i+1,len(stones)):
                        #_stones = stones[:]
                        s_i = stones[i]
                        s_j = stones[j]
                        _val = abs(stones[i] - stones[j])
                        stones.remove(s_i)
                        stones.remove(s_j)
                        if _val:
                            stones.append(_val)
                        stones.sort()
                        last = _lastStoneWeight(stones, mem)
                        min_last = min(last, min_last)
                        if _val:
                            stones.remove(_val)
                        stones.append(s_i)
                        stones.append(s_j)
                        
                mem[tuple(stones)] = min_last
                return min_last        
        stones.sort()
        return _lastStoneWeight(stones,mem)
        '''

'''
[57,32,40,27,35,61]
[31,26,33,21,40]
[2,7,4,1,8,1]
[2,7]
[1,1,2,3,5,8,13,21,34,55,89,14,23,37,61,98]
'''    
