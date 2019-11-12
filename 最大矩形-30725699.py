class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        sub_mat = []
        size = len(matrix)
        if size < 1:
            return 0
        lens = len(matrix[0])
        
        def localMax(arr: List[int], n: int, lens: int) -> int:
            local_mx = 0
            _mx = 0
            
            for i in range(lens):
                if arr[i] == n:
                    _mx += n
                else:
                    local_mx = max(local_mx, _mx)
                    _mx = 0
            
            return max(local_mx,_mx)
        
        mx = 0

        for x in matrix:
            y = list(map(int,x))
            if len(sub_mat) > 0:
                for i in range(lens):
                    y[i] += sub_mat[-1][i]
            sub_mat.append(y)
            n = len(sub_mat)
            _mx = 0
            
            _mx = localMax(y,n,lens)
            mx = max(_mx, mx)
                    
                    
        
        for i in range(size-1):
            for j in range(i+1,size):
                x = sub_mat[i]
                y = sub_mat[j]
                z = []
                for k in range(lens):
                    z.append(y[k] - x[k])
                
                n = j-i
                _mx = 0
                
                _mx = localMax(z,n,lens)
            
                #print(x,y,z,n,_mx)
                '''
                st = -1
                for m in range(lens):
                    if z[m] == n:
                        st = m
                        _mx = n
                        break
                if st > -1 :
                    print(_mx,n,st,lens)
                    for c in range(st+1,lens):
                        if z[c] == z[c-1]:
                            _mx += n
                        else:
                            break
                    print(_mx)
                '''
                mx = max(mx,_mx)
        
        return mx