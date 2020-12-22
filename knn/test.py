from collections import defaultdict


raw_data = {
    "EZ/TC": {"id": 1, "amount": 9},
    "LM/TH": {"id": 2, "amount": 8},
    "CD/EH": {"id": 3, "amount": 7},
    "EH/TC": {"id": 4, "amount": 6},
    "LM/TC": {"id": 5, "amount": 5},
    "CD/TC": {"id": 6, "amount": 4},
    "BT/TH": {"id": 7, "amount": 3},
    "BT/TX": {"id": 8, "amount": 2},
    "TX/TH": {"id": 9, "amount": 1},
}

def construct_graph(raw_data):      # here we will change representation
    graph = defaultdict(list)   # our graph
    for pair in raw_data:           # go through every edge
        u, v = pair.split("/")  # get from and to vertexes
        graph[u].append(v)      # and add this edge in our structure
    return graph


def dfs(g, u, dist):                # this is a simple dfs function
    if dist == 2:                   # we has a 'dist' from our start
        return [u]                  # and if we found already answer, return it
    for v in g.get(u, []):          # otherwise check all neighbours of current vertex
        ans = dfs(g, v, dist + 1)   # run dfs in every neighbour with dist+1
        if ans:                     # and if that dfs found something
            ans.append(u)           # store it in ouy answer
            return ans              # and return it
    return []                       # otherwise we found nothing

def main():
    graph = construct_graph(raw_data)
    for v in graph.keys():              # here we will try to find path
        ans = dfs(graph, v, 0)          # starting with 0 dist
        if ans:                         # and if we found something
            print(list(reversed(ans)))  # return it, but answer will be reversed

if __name__ == '__main__':
    main()
