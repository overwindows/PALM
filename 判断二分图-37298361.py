class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        if len(graph) < 1:
            return True
        part = [-1] * len(graph) 
        def travalGraph(root, graph, part) -> bool:
            for i in graph[root]:
                if part[i] == -1:
                    part[i] = int(not part[root])
                    ret = travalGraph(i, graph, part)
                    if not ret:
                        return False
                elif part[i] == part[root]:
                    return False
            return True
        
        root = 0
        while part.count(-1) > 0:
            while root < len(graph) and (len(graph[root]) == 0 or part[root] != -1):
                root += 1
            if root == len(graph):
                return True
            if part[root] == -1:
                part[root] = 1
                ret = travalGraph(root, graph, part)
                if not ret:
                    return False
            else:
                root += 1
                continue
        return True
'''
[[2,4],[2,3,4],[0,1],[1],[0,1],[7],[9],[5],[],[6],[12,14],[],[10],[],[10],[19],[18],[],[16],[15],[23],[23],[],[20,21],[],[],[27],[26],[],[],[34],[33,34],[],[31],[30,31],[38,39],[37,38,39],[36],[35,36],[35,36],[43],[],[],[40],[],[49],[47,48,49],[46,48,49],[46,47,49],[45,46,47,48]]
[[1,3],[0,2],[1,3],[0,2]]
[[1,2,3], [0,2], [0,1,3], [0,2]]
[[4],[],[4],[4],[0,2,3]]
[[],[2,4,6],[1,4,8,9],[7,8],[1,2,8,9],[6,9],[1,5,7,8,9],[3,6,9],[2,3,4,6,9],[2,4,5,6,7,8]]
[[1],[0],[4],[4],[2,3]]
[]
[[]]
'''
        