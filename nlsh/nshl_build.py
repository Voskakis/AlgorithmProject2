#TODO get the graph from c library
#graph will be a list of lists of ints. index 0 will be an empty list.
# The rest will be the list of the number of each neighbor of the node whose number is this index
#TODO turn into a weighted, undirected graph that's compatible with KaHIP
#create another list of lists of ints. Index 0 is empty list.
# Go over each index of the above outer list. For ever integer in the inner list of that index, add this integer to the
# list of that index in the new container, and the current index to the container whose index is the integer
# then sort every inner list
# then go index by index on the new outer, writing to a file each integer in the inner list. After each, check the
# next integer. If it is different, write 1 to the txt file and continue. If it is the same, write 2,
# skip the next index and continue
#TODO run KaHIP
#that means using kaffpa (page 12) on the above file, a single call
#TODO train PyTorch
#specifics a little unclear. KaHIP makes a file where line i contains the block ID of vertex i. Pytorch must learn to
# predict those. Either we getline the line i, if that's possible efficiently, or load them to a map where key is the
# vertex number i, and value is the tag
#TODO make inverted file
#actually easy. Array of strings, and then for each key in the map, take the value, use it as index on the array of
# strings, append index there, then print them in order to file.