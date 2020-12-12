import matplotlib.pyplot as plt
import numpy as np

def generateMainEdgePathString(computeGraph, calDistVals=None):
    """
    Generate string representation of all edge connections corresponding to the outputs.
    :return:
    """
    supportedFunctionSet = computeGraph["supportedFunctionSet"]
    layerStructureComputeGraph = computeGraph["layerStructureComputeGraph"]
    inputData = [iv.name for iv in computeGraph["input_variables"]] + ["c1"]
    layerStructureData = [inputData] + [layerData for layerData in layerStructureComputeGraph]
    layerStructureData = layerStructureData + [[ov.name for ov in computeGraph["output_variables"]]]

    opt_dist_variables = computeGraph["opt_dist_variables"]

    layer_sizes = [len(l) for l in layerStructureData]

    # Draw nodes.
    edgeSet = []
    opVarIndex = 0
    outputNodeIndexes = []
    temp_node_index = 0
    previous_node_indexes = []
    for n, layer_size in enumerate(layer_sizes):
        temp_node_indexes = []
        for m in range(layer_size):
            temp_node_indexes.append(temp_node_index)
            if layerStructureData[n][m] == "sum":
                for temp_prev_node_index in previous_node_indexes:
                    edgeSet = edgeSet + [[temp_prev_node_index, temp_node_index]]

            if layerStructureData[n][m] in supportedFunctionSet.keys():
                for _ in range(supportedFunctionSet[layerStructureData[n][m]][0]):
                    opV = calDistVals[opt_dist_variables[opVarIndex]]
                    prevNodeIndex = np.argmax(opV)
                    edgeSet = edgeSet + [[previous_node_indexes[prevNodeIndex], temp_node_index]]
                    opVarIndex = opVarIndex + 1

            elif n == len(layer_sizes) - 1:
                outputNodeIndexes = outputNodeIndexes + [opVarIndex]
                opV = calDistVals[opt_dist_variables[opVarIndex]]
                prevNodeIndex = np.argmax(opV)
                edgeSet = edgeSet + [[previous_node_indexes[prevNodeIndex], temp_node_index]]
                opVarIndex = opVarIndex + 1

            temp_node_index = temp_node_index + 1
        previous_node_indexes = temp_node_indexes

    # Determine edges on the main path.
    mainPathEdges = []
    while len(previous_node_indexes) > 0:
        tempNodeIndex = previous_node_indexes.pop(0)
        for edge in edgeSet:
            if edge[1] == tempNodeIndex:
                mainPathEdges = mainPathEdges + [edge]
                previous_node_indexes.append(edge[0])
    mainPathEdges.sort(key=lambda x: x[0])
    return str(mainPathEdges)

def drawComputeGraph(session, computeGraph, figName, calDistVals=None, figSizeX=12, figSizeY=12):
    """
    Draw given computation graph.
    :return:
    """
    fig = plt.figure(figsize=(figSizeX, figSizeY))
    ax = fig.gca()
    ax.axis('off')

    left = 0.1
    right = 0.9
    bottom = 0.1
    top = 0.9

    supportedFunctionSet = computeGraph["supportedFunctionSet"]
    layerStructureComputeGraph = computeGraph["layerStructureComputeGraph"]

    inputData = [ iv.name for iv in computeGraph["input_variables"] ] + ["c1"]
    layerStructureData = [ inputData ] + [layerData for layerData in layerStructureComputeGraph ]
    layerStructureData = layerStructureData + [[ov.name for ov in computeGraph["output_variables"]]]

    opt_dist_variables = computeGraph["opt_dist_variables"]
    trainable_w_variables = computeGraph["trainable_w_variables"]
    trainable_sum_w_variables = computeGraph["trainable_sum_w_variables"]

    layer_sizes = [len(l) for l in layerStructureData]
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Draw nodes.
    previousLayerNodeCenters = []
    edgeSet = []
    textSet = []
    opVarIndex = 0
    sumWIndex = 0
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        tempNodeCenters = []
        for m in range(layer_size):
            nodeCenter = (n * h_spacing + left, layer_top - m * v_spacing)
            tempNodeCenters = tempNodeCenters + [nodeCenter]
            circle = plt.Circle(nodeCenter, v_spacing / 4., color='w', ec='k', zorder=4, fill=False)
            text = plt.text(nodeCenter[0], nodeCenter[1], layerStructureData[n][m])
            ax.add_artist(text)
            ax.add_artist(circle)

            if layerStructureData[n][m] == "sum":
                sumWVs = session.run(trainable_sum_w_variables[sumWIndex])
                positionChoices = [(i%2)*0.4 + 0.4 for i in range(len(sumWVs))]
                for tempPrevNode, tempWVs, positionChoice in zip(previousLayerNodeCenters, sumWVs, positionChoices):
                    edgeSet = edgeSet + [[tempPrevNode, nodeCenter, "k"]]
                    textSet = textSet + [['%.2f' % tempWVs, positionChoice * np.array(tempPrevNode) + (1 - positionChoice) * np.array(nodeCenter)]]
                sumWIndex = sumWIndex + 1

            if layerStructureData[n][m] in supportedFunctionSet.keys():
                for _ in range(supportedFunctionSet[layerStructureData[n][m]][0]):
                    opV = calDistVals[opt_dist_variables[opVarIndex]]
                    prevNodeIndex = np.argmax(opV)
                    edgeSet = edgeSet + [ [previousLayerNodeCenters[prevNodeIndex], nodeCenter, "k"] ]

                    wV = session.run(trainable_w_variables[opVarIndex])
                    positionChoice = [0.8, 0.4][opVarIndex % 2]
                    textSet = textSet + [['%.2f' % wV, positionChoice * np.array(previousLayerNodeCenters[prevNodeIndex]) + (1 - positionChoice) * np.array(nodeCenter)]]

                    opVarIndex = opVarIndex + 1
            elif n == len(layer_sizes)-1:
                opV = calDistVals[opt_dist_variables[opVarIndex]]
                prevNodeIndex = np.argmax(opV)
                edgeSet = edgeSet + [[previousLayerNodeCenters[prevNodeIndex], nodeCenter, "k"]]

                wV = session.run(trainable_w_variables[opVarIndex])
                positionChoice = [0.8, 0.4][opVarIndex % 2]
                textSet = textSet + [['%.2f' % wV, positionChoice * np.array(previousLayerNodeCenters[prevNodeIndex]) + (1-positionChoice) * np.array(nodeCenter)]]

                opVarIndex = opVarIndex + 1

        previousLayerNodeCenters = tempNodeCenters

    # Determine edges on the main path.
    while len(previousLayerNodeCenters) > 0:
        nodeCenter = previousLayerNodeCenters.pop(0)
        for edge in edgeSet:
            if edge[1] == nodeCenter:
                edge[2] = "r"
                previousLayerNodeCenters.append(edge[0])

    # Draw edges.
    for edge in edgeSet:
        line = plt.Line2D([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c=edge[2])
        ax.add_artist(line)

    # Draw edge weights.
    for t in textSet:
        text = plt.text(t[1][0], t[1][1], t[0])
        ax.add_artist(text)

    # fig.show()
    fig.savefig(figName)
    plt.close(fig)
