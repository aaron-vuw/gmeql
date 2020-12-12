import tensorflow as tf
import uuid

# Define numerical bounds.
div_bound = 100.
exp_bound = 1.e+10

supportedFunctionSet = {
    "plus": [2, lambda x, y: tf.reshape(x + y, [-1, 1])],
    "sub": [2, lambda x, y: tf.reshape(x - y, [-1, 1])],
    "multiplication": [2, lambda x, y: tf.reshape(x * y, [-1, 1])],
    "division": [2, lambda x, y: tf.reshape(tf.clip_by_value(tf.math.divide_no_nan(x, y), -div_bound, +div_bound), [-1, 1])],
    "inverse": [1, lambda x: tf.reshape(tf.clip_by_value(tf.math.divide_no_nan(tf.cast(1., dtype=tf.float64), x), -div_bound, +div_bound), [-1, 1])],
    "square": [1, lambda x: tf.reshape(tf.math.pow(x, 2), [-1, 1])],
    "cubic": [1, lambda x: tf.reshape(tf.math.pow(x, 3), [-1, 1])],
    "fourp": [1, lambda x: tf.reshape(tf.math.pow(x, 4), [-1, 1])],
    "fivep": [1, lambda x: tf.reshape(tf.math.pow(x, 5), [-1, 1])],
    "sixp": [1, lambda x: tf.reshape(tf.math.pow(x, 6), [-1, 1])],
    "sqrt": [1, lambda x: tf.reshape(tf.math.sqrt(tf.math.abs(x)), [-1, 1])],
    "exp": [1, lambda x: tf.reshape(tf.clip_by_value(tf.math.exp(x), 0.0, exp_bound), [-1, 1])],
    "sin": [1, lambda x: tf.reshape(tf.math.sin(x), [-1, 1])],
    "cos": [1, lambda x: tf.reshape(tf.math.cos(x), [-1, 1])],
    "identity": [1, lambda x: tf.reshape(x, [-1,1])]
}

layerStructureComputeGraph = [
    ["identity", "square", "cubic", "fourp", "fivep", "sixp", "sin", "cos"],
    ["plus", "multiplication", "division", "sum"],
    ["plus", "multiplication"]
]

def buildComputeGraph(dataLoader, supportedFunctionSet, layerStructureComputeGraph, train_var_op_initializer, train_var_w_initializer):
    """
    Build computation graph.
    :param dataLoader:
    :param layerStructureComputeGraph
    :return:
    """
    # Define constant.
    c1 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1], name="c1")

    # Define temperature.
    temperature = tf.compat.v1.placeholder(tf.float64, shape=(), name="temperature")

    # Collect all important elements of the computation graph.
    trainable_op_variables = []
    trainable_w_variables = []
    trainable_sum_w_variables = []
    random_variables = []
    opt_switch_variables = []
    opt_dist_variables = []
    opt_dist_values = []
    output_values = []
    logDistLoss = 0
    distLoss = 0

    # Build internal layers of the computation graph.
    previousLayerOutput = []
    for temp_i in range(len(layerStructureComputeGraph)):
        # Define input tensor.
        if len(previousLayerOutput) < 1:
            inputTensorLength = len(dataLoader.getInputVariables()) + 1
            inputTensor = tf.concat(dataLoader.getInputVariables() + [c1], axis=1)
        else:
            inputTensorLength = len(previousLayerOutput)
            inputTensor = tf.concat(previousLayerOutput, axis=1)

        previousLayerOutput = []
        with tf.compat.v1.variable_scope('layer' + str(temp_i)):
            for func in layerStructureComputeGraph[temp_i]:
                randID = str(uuid.uuid4())
                if func == "sum":
                    var_sum_w = tf.Variable(train_var_w_initializer(shape=[inputTensorLength], dtype=tf.float64), name="sum_w_" + str(len(trainable_sum_w_variables)) + "_" + str(temp_i) + "_" + randID)
                    trainable_sum_w_variables.append(var_sum_w)
                    previousLayerOutput.append(
                        tf.reshape(
                            tf.reduce_sum(var_sum_w * inputTensor, axis=1),
                            [-1,1]
                        )
                    )
                    continue

                if func not in supportedFunctionSet.keys():
                    continue

                if supportedFunctionSet[func][0] == 1:
                    # Define op, w, r, dist variables.
                    var_opt = tf.Variable(train_var_op_initializer(shape=[inputTensorLength], dtype=tf.float64), name=func + "_opt_" + str(0) + "_" + str(temp_i) + "_" + randID)
                    var_opt_switch = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_os_" + str(0) + "_" + str(temp_i) + "_" + randID)
                    var_w = tf.Variable(train_var_w_initializer(shape=(), dtype=tf.float64), name=func + "_w_" + str(0) + "_" + str(temp_i) + "_" + randID)
                    var_r = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_r_" + str(0) + "_" + str(temp_i) + "_" + randID)
                    var_dist = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_dist_" + str(0) + "_" + str(temp_i) + "_" + randID)

                    trainable_op_variables.append(var_opt)
                    opt_switch_variables.append(var_opt_switch)
                    trainable_w_variables.append(var_w)
                    random_variables.append(var_r)
                    opt_dist_variables.append(var_dist)

                    opr = tf.nn.softmax((var_opt + var_r) / temperature, axis=1) * var_opt_switch + var_dist * (tf.cast(1., dtype=tf.float64) - var_opt_switch)
                    op = var_w * tf.reduce_sum(opr * inputTensor, axis=1)
                    opt_dist_values.append(opr)
                    logDistLoss = logDistLoss + tf.reduce_mean(
                        tf.math.log((tf.exp(var_opt) * tf.pow(var_dist, -temperature - 1.)) / tf.reshape(tf.reduce_sum(tf.exp(var_opt) * tf.pow(var_dist, -temperature), axis=1),[-1, 1])),
                    )
                    distLoss = distLoss + tf.reduce_mean(
                        tf.reduce_sum(tf.abs(tf.exp(var_opt) - tf.pow(var_dist, temperature)), axis=1)
                    )
                    temp_output = supportedFunctionSet[func][1](op)
                    previousLayerOutput.append(temp_output)

                elif supportedFunctionSet[func][0] == 2:
                    # Define op, w, r, dist variables.
                    var_opt_1 = tf.Variable(train_var_op_initializer(shape=[inputTensorLength], dtype=tf.float64), name=func + "_opt_" + str(1) + "_" + str(temp_i) + "_" + randID)
                    var_opt_switch_1 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_os_" + str(1) + "_" + str(temp_i) + "_" + randID)
                    var_w_1 = tf.Variable(train_var_w_initializer(shape=(), dtype=tf.float64), name=func + "_w_" + str(1) + "_" + str(temp_i) + "_" + randID)
                    var_r_1 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_r_" + str(1) + "_" + str(temp_i) + "_" + randID)
                    var_dist_1 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_dist_" + str(1) + "_" + str(temp_i) + "_" + randID)

                    var_opt_2 = tf.Variable(train_var_op_initializer(shape=[inputTensorLength], dtype=tf.float64), name=func + "_opt_" + str(2) + "_" + str(temp_i) + "_" + randID)
                    var_opt_switch_2 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_os_" + str(2) + "_" + str(temp_i) + "_" + randID)
                    var_w_2 = tf.Variable(train_var_w_initializer(shape=(), dtype=tf.float64), name=func + "_w_" + str(2) + "_" + str(temp_i) + "_" + randID)
                    var_r_2 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_r_" + str(2) + "_" + str(temp_i) + "_" + randID)
                    var_dist_2 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name=func + "_dist_" + str(2) + "_" + str(temp_i) + "_" + randID)

                    trainable_op_variables = trainable_op_variables + [var_opt_1, var_opt_2]
                    opt_switch_variables = opt_switch_variables + [var_opt_switch_1, var_opt_switch_2]
                    trainable_w_variables = trainable_w_variables + [var_w_1, var_w_2]
                    random_variables = random_variables + [var_r_1, var_r_2]
                    opt_dist_variables = opt_dist_variables + [var_dist_1, var_dist_2]

                    opr1 = tf.nn.softmax((var_opt_1 + var_r_1) / temperature, axis=1) * var_opt_switch_1 + var_dist_1 * (tf.cast(1., dtype=tf.float64) - var_opt_switch_1)
                    op1 = var_w_1 * tf.reduce_sum(opr1 * inputTensor, axis=1)
                    opt_dist_values.append(opr1)
                    logDistLoss = logDistLoss + tf.reduce_mean(
                        tf.math.log((tf.exp(var_opt_1) * tf.pow(var_dist_1, -temperature - 1.)) / tf.reshape(tf.reduce_sum(tf.exp(var_opt_1) * tf.pow(var_dist_1, -temperature), axis=1), [-1, 1])),
                    )
                    distLoss = distLoss + tf.reduce_mean(
                        tf.reduce_sum(tf.abs(tf.exp(var_opt_1) - tf.pow(var_dist_1, temperature)), axis=1)
                    )

                    opr2 = tf.nn.softmax((var_opt_2 + var_r_2) / temperature, axis=1) * var_opt_switch_2 + var_dist_2 * (tf.cast(1., dtype=tf.float64) - var_opt_switch_2)
                    op2 = var_w_2 * tf.reduce_sum(opr2 * inputTensor, axis=1)
                    opt_dist_values.append(opr2)
                    logDistLoss = logDistLoss + tf.reduce_mean(
                        tf.math.log((tf.exp(var_opt_2) * tf.pow(var_dist_2, -temperature - 1.)) / tf.reshape(tf.reduce_sum(tf.exp(var_opt_2) * tf.pow(var_dist_2, -temperature), axis=1), [-1, 1])),
                    )
                    distLoss = distLoss + tf.reduce_mean(
                        tf.reduce_sum(tf.abs(tf.exp(var_opt_2) - tf.pow(var_dist_2, temperature)), axis=1)
                    )

                    previousLayerOutput.append(supportedFunctionSet[func][1](op1, op2))

    # Build output layer of the computation graph.
    inputTensorLength = len(previousLayerOutput)
    inputTensor = tf.concat(previousLayerOutput, axis=1)

    with tf.compat.v1.variable_scope('layer_O'):
        for _ in dataLoader.getOutputVariables():
            randID = str(uuid.uuid4())
            var_out_opt = tf.Variable(train_var_op_initializer(shape=[inputTensorLength], dtype=tf.float64), name="out_opt_" + randID)
            var_out_switch = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name="out_os_" + randID)
            var_out_w = tf.Variable(train_var_w_initializer(shape=(), dtype=tf.float64), name="out_w_" + randID)
            var_out_r = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name="out_r_" + randID)
            var_out_dist = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, inputTensorLength], name="out_dist_" + randID)

            trainable_op_variables.append(var_out_opt)
            opt_switch_variables.append(var_out_switch)
            trainable_w_variables.append(var_out_w)
            random_variables.append(var_out_r)
            opt_dist_variables.append(var_out_dist)

            opo = tf.nn.softmax((var_out_opt + var_out_r) / temperature, axis=1) * var_out_switch + var_out_dist * (tf.cast(1., dtype=tf.float64) - var_out_switch)
            output = var_out_w * tf.reduce_sum(opo * inputTensor, axis=1)
            opt_dist_values.append(opo)
            logDistLoss = logDistLoss + tf.reduce_mean(
                tf.math.log((tf.exp(var_out_opt) * tf.pow(var_out_dist, -temperature - 1.)) / tf.reshape(tf.reduce_sum(tf.exp(var_out_opt) * tf.pow(var_out_dist, -temperature), axis=1), [-1, 1])),
            )
            distLoss = distLoss + tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.exp(var_out_opt) - tf.pow(var_out_dist, temperature)), axis=1)
            )

            temp_output = tf.reshape(output, [-1, 1])
            output_values.append(temp_output)

    # Produce final output for the constructed computation graph.
    return {
        "c1": c1,
        "temperature": temperature,
        "trainable_op_variables": trainable_op_variables,
        "trainable_w_variables": trainable_w_variables,
        "random_variables": random_variables,
        "opt_dist_variables": opt_dist_variables,
        "opt_dist_values": opt_dist_values,
        "output_values": output_values,
        "logDistLoss": logDistLoss,
        "distLoss": distLoss,
        "opt_switch_variables": opt_switch_variables,
        "trainable_sum_w_variables": trainable_sum_w_variables,
        "supportedFunctionSet": supportedFunctionSet,
        "layerStructureComputeGraph": layerStructureComputeGraph,
        "input_variables": dataLoader.getInputVariables(),
        "output_variables": dataLoader.getOutputVariables()
    }
