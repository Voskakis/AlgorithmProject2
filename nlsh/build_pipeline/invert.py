def create_inverted_file(blocks:list[int], nblocks:int, output_file:str):
  inverted = [[] for _ in range(nblocks)]
  for i in range(len(blocks)):
    inverted[blocks[i]].append(i)
  output = open(output_file, "w")
  for line in inverted:
    output.write(f"{line}\n")
  output.close()