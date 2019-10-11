class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        if k == 0:
            return None
        
        if len(arr) == 1:
            return arr
        
        if k >= len(arr):
            return arr
        
        assert k < len(arr)
        
        st = 0
        ed = len(arr)-1
        pivot = 0
        
        #print(st,ed)
        while st < ed:
            mid = (st+ed)//2
            #print(mid)
            if arr[mid] == x:
                #print(mid)
                st = mid
                break
            if arr[mid] > x:
                ed = mid-1
            else:
                st = mid+1
        
        pivot = st
        KNN = []
        
        #print(pivot,arr[pivot])
        if x == arr[pivot]:
            KNN.append(x)       
            if pivot == 0:
                KNN.extend(arr[pivot+1:pivot+k])
                KNN.sort()
                return KNN
            if pivot == len(arr)-1:
                KNN.extend(arr[pivot-k+1:pivot])
                KNN.sort()
                return KNN
            st = pivot - 1
            ed = pivot + 1
        elif x > arr[pivot]:
            if pivot+1 == len(arr):
                KNN.extend(arr[pivot-k:pivot])
                #KNN.sort()
                return KNN
            st = pivot
            ed = pivot + 1
        else:
            if pivot == 0:
                KNN.extend(arr[pivot:pivot+k])
                KNN.sort()
                return KNN
            st = pivot-1
            ed = pivot      
        
        '''
        if st < 0:
            if len(KNN) > 0:
                k = min(k-1,len(arr) - ed)
            else:
                k = min(k,len(arr) - ed)
            KNN.extend(arr[ed:ed+k])
        
        elif ed > len(arr) - 1:
            if len(KNN) > 0:
                k = min(k-1, len(arr)-st)
            else:
                k = min(k, len(arr)-st)
            
            KNN.extend(arr(ed-k+1,ed+1))
        else:
        '''
        while len(KNN) < k:
            # print(st,ed)
            if x - arr[st] <= arr[ed] - x:
                KNN.append(arr[st])
                st -= 1
                if st < 0:
                    KNN.extend(arr[ed:ed+k-len(KNN)])
                    break
            else:
                KNN.append(arr[ed])
                ed += 1
                if ed >= len(arr):
                    KNN.extend(arr[st-(k-len(KNN))+1:(st+1)])
                    break
        
        assert len(KNN) == k, len(KNN)
        KNN.sort()
        return KNN
