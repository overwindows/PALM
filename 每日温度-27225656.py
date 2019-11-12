class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        res = [0] * len(T)
        stack.append((T[-1],len(T)-1))
        for i in range(len(T)-2,-1,-1):
            val, pos = stack[-1]
            if T[i] >= val:
                steps = 0
                while T[i] >= val:
                    stack.pop()
                    if stack:
                        val,pos = stack[-1]
                        steps = pos-i
                    else:
                        steps = 0
                        break
                res[i] = steps
                #stack.append((T[i],i))
            else:
                res[i] = 1
            stack.append((T[i],i))
        return res
        