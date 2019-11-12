class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = []
        heapq.heappush(heap,1)
        nth = 0
        ugly = set()
        
        for _ in range(n):
            nth = heapq.heappop(heap)
            
            while nth in ugly:
                nth = heapq.heappop(heap)
            
            ugly.add(nth)
            
            heapq.heappush(heap, nth*2)
            heapq.heappush(heap, nth*3)
            heapq.heappush(heap, nth*5)
        
        l = list(ugly)
        l.sort()
        #print(l)
        return l[-1]
        
        '''
        ugly = [False] * 30000000
        
        if n <= 6:
            return n
        
        for i in range(1,7):
            ugly[i] = True
        
        n -= 6
        ix = 7
        
        while n > 0:
            if ix % 2 == 0 and ugly[ix // 2] or ix % 3 == 0 and ugly[ix // 3] or ix % 5 == 0 and ugly[ix//5]:
                ugly[ix] = True
                n -= 1
            ix += 1
        
        
        return ix-1
        '''