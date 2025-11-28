#TODO get the graph from c library
#graph will be a list of lists of ints. index 0 will be an empty list.
# The rest will be the list of the number of each neighbor of the node whose number is this index
knn = []
node_count = 0
with open("c-knn") as file:
    iter_file = iter(file)
    knn.append(next(iter_file))
    for line in iter_file:
        node_count += 1
        elements = line.split()
        for i in range(len(elements)):
            elements[i] = int(elements[i])
        knn.append(elements)
#TODO turn into a weighted, undirected graph that's compatible with KaHIP
#create another list of lists of ints. Index 0 is empty list.
# Go over each index of the above outer list. For ever integer in the inner list of that index, add this integer to the
# list of that index in the new container, and the current index to the container whose index is the integer
vertex_set = set()
knn_temp = [ [] for _ in range(node_count+1)]
for i in range(1, node_count+1):
    for node in knn[i]:
        knn_temp[i].append(node)
        knn_temp[node].append(i)
        vertex_set.add(f"{min(i, node)} {max(i, node)}")
# then sort every inner list
for knnlist in knn_temp:
    knnlist.sort()
# then go index by index on the new outer, writing to a file each integer in the inner list. After each, check the
# next integer. If it is different, write 1 to the txt file and continue. If it is the same, write 2,
# skip the next index and continue
output = open("KaHIP-input.txt", "w")
output.write(f"{node_count} {len(vertex_set)} 1")
for knnlist in knn_temp:
    i=0
    while i < len(knnlist):
        output.write(str(knnlist[i]))
        output.write(' ')
        if (i + 1) < len(knnlist) and knnlist[i]==knnlist[i + 1]: #com
            output.write('2')
            i+=1
        else:
            output.write('1')
        output.write(' ')
        i+=1
    output.write('\n')
output.close()
#TODO run KaHIP
#that means using kaffpa (page 12) on the above file, a single call
#TODO train PyTorch
#specifics a little unclear. KaHIP makes a file where line i contains the block ID of vertex i. Pytorch must learn to
# predict those. Either we getline the line i, if that's possible efficiently, or load them to a map where key is the
# vertex number i, and value is the tag
#TODO make inverted file
#actually easy. Array of strings, and then for each key in the map, take the value, use it as index on the array of
# strings, append index there, then print them in order to file.