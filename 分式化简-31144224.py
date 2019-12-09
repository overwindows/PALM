class Solution:
    def fraction(self, cont: List[int]) -> List[int]:
        cont.reverse()
        output = [0,1]
        for i in range(len(cont)):
            n = cont[i] * output[1] + output[0]
            m = output[1]
            output = [m, n]
        output[0],output[1] = output[1],output[0]
        return output
            