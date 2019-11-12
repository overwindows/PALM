class Solution:
    def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
        arr = []
        arr.append(a)
        arr.append(b)
        arr.append(c)
        arr.sort()
        
        min_steps = 0
        max_moves = 0
        
        if arr[1] - arr[0] == 1 and arr[2] - arr[1] == 1:
            return [0,0]
        
        if arr[1] - arr[0] > 1:
            min_steps += 1
            max_moves += (arr[1] - arr[0]-1)

            
        if arr[2] - arr[1] > 1:
            min_steps += 1
            max_moves += (arr[2] - arr[1]-1)

        if arr[1] - arr[0] == 2 or arr[2] - arr[1] == 2:
            min_steps = 1
            
        
        
        return [min_steps,max_moves]