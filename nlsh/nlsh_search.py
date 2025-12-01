import math
import time

import torch
import torch.nn.functional as F

def format_output(image_id: int, results: list[list[int]], r:int=0) -> str:
    header = "Query: {qID}\n"
    bodyPiece = "Nearest neighbor-{count}: {neighborID}\ndistanceApproximate: {dis}\ndistanceTrue: {disT}\n\n}"
    rPiece = "R-near neighbors:\n"

    output = header.format(qID = image_id)
    for result, index in zip(results, range(len(results))):
        output += bodyPiece.format(count=index, neighborID = result[0], dis = result[1], disT=result[2])
    if r!=0:
        output += rPiece
        for result in results:
            if result[1]<r:
                output += str(result[0])
    return output

def format_footer(queryCount: int, N:int, apTime:float, truTime: float, AFcount:int, recNcount:int)->str:
    footer = "Average AF: {avAF}\nRecall@N: {recN}\nQPS: {qps}\ntApproximateAverage: {tAvg}\ntTrueAverage: {tTrueAvg}"
    avAF = AFcount/queryCount
    recN = recNcount/(queryCount*N)
    tAvg = apTime/N
    tTrue = truTime/N
    return footer.format(avAF=avAF, recN=recN, QPS = 1/tAvg, tAvg=tAvg, tTrue=tTrue)

def euclidean(vector1, vector2):
    total = 0
    for x, y in zip(vector1, vector2):
        total += (x - y)**2
    return math.sqrt(total)

def add_better_result(point, q, result_list, N):
    distance = euclidean(point, q)
    if len(result_list) < N:
        result_list.append([point, distance])
        if len(result_list) == N and N > 1:
            max_index = max(range(len(result_list)), key=lambda x: x[1])
            result_list[0], result_list[max_index] = result_list[max_index], result_list[0]
    elif distance < result_list[0][1]:
        result_list[0] = [point, distance]
        max_index = max(range(len(result_list)), key=lambda x: x[1])
        result_list[0], result_list[max_index] = result_list[max_index], result_list[0]
    return result_list

def exhaustive_search(point_set: list[list[int]], q, N) -> list[list[int]]:
    result_list = []
    for point in point_set:
        result_list = add_better_result(point, q, result_list, N)
    result_list = sorted(result_list, key=lambda x: x[1])
    return result_list


# TODO input handling
queries = [[]] # loaded by input
inverted_file = [[1,2,3],[4,5,6]] # loaded by input
point_set = [[]] # loaded by input
N = 1
T=1
R=0

model = torch.load("model.pth",map_location="cpu")
model.eval()

totalAproximateTime = 0
totalExhaustTime = 0
AFcount = 0
recNcount = 0

outputFile = open("filepath", "a")
outputFile.write("Neural LSH\n")

for q, index in zip(queries, range(len(queries))):
    start = time.time()
    # prediction
    query_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(query_tensor)
        probs = F.softmax(logits, dim=1)
    # multiprobe
    top_k_probs, top_k_labels = torch.topk(probs, T, 1)
    # candidates
    search_space = []
    for label in top_k_labels:
        for pointID in inverted_file[label]:
            search_space.append(point_set[pointID])
    # exhaustive search
    results = exhaustive_search(search_space, q, N)
    end = time.time()
    totalAproximateTime += (end - start)

    #brute force
    start = time.time()
    exhaustResults = exhaustive_search(point_set, q, N)
    end = time.time()
    totalExhaustTime += (end - start)

    # output
    AFcount += results[0][1]/results[0][2]
    recNcount += sum(x not in results for x in exhaustResults)

    outputFile.write(format_output(index, [r1 +r2[1:] for r1, r2 in zip(results, exhaustResults)],R))
outputFile.write(format_footer(len(queries), N, totalAproximateTime, totalExhaustTime, AFcount, recNcount))
outputFile.close()

