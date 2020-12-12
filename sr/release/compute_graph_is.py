import tensorflow as tf
import numpy as np
import random
import os
import shutil
import math

# =====================
# All required constants.
# =====================
# Define close-to-zero constant.
ct0 = 1.0e-10

# Define division bound.
div_bound = 100.

# Define default temperature.
defaultTemperature = 2./3.
temperatureUpdateStep = 0.02
maxDefaultTemperature = 2./3.
minDefaultTemperature = 2./3.

# Define elite buffer size = 1000.
eliteBufferSize = 400
randomSampleSize = 40

# Define gradient bound.
gradientBound = 1.

class EliteBuffer:
    def __init__(self, sess, size, checkRecordFingerprint, computeGraph):
        self.sess = sess
        self.size = size
        self.records = []
        self.checkRecordFingerprint = checkRecordFingerprint
        self.computeGraph = computeGraph

    def capacity(self):
        return len(self.records)

    def clear(self):
        self.records = []

    def averageAccuracy(self):
        """
        Calculate average accuracy across all records.
        :return:
        """
        if self.capacity() <= 0:
            return 0.
        else:
            return np.mean([rec['accuracy'] for rec in self.records])

    def bestAccuracy(self):
        if self.capacity() <= 0:
            return 0.
        else:
            return np.min([rec['accuracy'] for rec in self.records])

    def recordStore(self, record):
        if self.checkRecordFingerprint:
            temp_hasMatch = False
            for temp_r_index in range(len(self.records)):
                if self.records[temp_r_index]['edge_path'] == record['edge_path']:
                    if self.records[temp_r_index]['accuracy'] > record['accuracy']:
                        self.records[temp_r_index] = record
                        self.sortRecords()
                    temp_hasMatch = True
                    break

            if not temp_hasMatch:
                if len(self.records) < self.size:
                    self.records.append(record)
                    self.records.sort(reverse=True, key=lambda x: x['accuracy'])
                elif self.records[0]['accuracy'] > record['accuracy']:
                    self.records[0] = record
                    self.records.sort(reverse=True, key=lambda x: x['accuracy'])
        else:
            if len(self.records) < self.size:
                self.records.append(record)
                self.records.sort(reverse=True, key=lambda x: x['accuracy'])
            elif self.records[0]['accuracy'] > record['accuracy']:
                self.records[0] = record
                self.records.sort(reverse=True, key=lambda x: x['accuracy'])

    def store(self, gumbelRandomVs, switchRandomVs, optDistVarValues, accuracies):
        """
        Store new records in the elite buffer.

        Every record is stored in the form {trainable_op_variables: operation distributions, ..., accuracy: accuracy_value (the smaller the better)}
        """
        random_variables = self.computeGraph["random_variables"]
        opt_switch_variables = self.computeGraph["opt_switch_variables"]
        opt_dist_variables = self.computeGraph["opt_dist_variables"]
        opt_dist_values = self.computeGraph["opt_dist_values"]
        temperature = self.computeGraph["temperature"]

        feed_dict = {}
        for vr in random_variables:
            feed_dict[vr] = gumbelRandomVs[vr]
        for vs in opt_switch_variables:
            feed_dict[vs] = switchRandomVs[vs]
        for vd in opt_dist_variables:
            feed_dict[vd] = optDistVarValues[vd]
        feed_dict[temperature] = defaultTemperature

        optDistVals = self.sess.run(opt_dist_values, feed_dict=feed_dict)

        fullRes = {}
        for vd, disVals in zip(opt_dist_variables, optDistVals):
            fullRes[vd] = disVals
        for temp_i in range(len(accuracies)):
            if accuracies[temp_i] is None or math.isnan(accuracies[temp_i]):
                continue
            recordOptDis = {}
            for vd in opt_dist_variables:
                recordOptDis[vd] = np.abs(fullRes[vd][temp_i])
            recordOptDis['accuracy'] = accuracies[temp_i]

            if self.checkRecordFingerprint:
                from sr.release.draw_compute_graph import generateMainEdgePathString
                recordOptDis['edge_path'] = generateMainEdgePathString(self.computeGraph, calDistVals=recordOptDis)

                temp_hasMatch = False
                for temp_r_index in range(len(self.records)):
                    if self.records[temp_r_index]['edge_path'] == recordOptDis['edge_path']:
                        if self.records[temp_r_index]['accuracy'] > recordOptDis['accuracy']:
                            self.records[temp_r_index] = recordOptDis
                            self.sortRecords()
                        temp_hasMatch = True
                        break

                if not temp_hasMatch:
                    if len(self.records) < self.size:
                        self.records.append(recordOptDis)
                        self.records.sort(reverse=True, key=lambda x: x['accuracy'])
                    elif self.records[0]['accuracy'] > recordOptDis['accuracy']:
                        self.records[0] = recordOptDis
                        self.records.sort(reverse=True, key=lambda x: x['accuracy'])
            else:
                if len(self.records) < self.size:
                    self.records.append(recordOptDis)
                    self.records.sort(reverse=True, key=lambda x: x['accuracy'])
                elif self.records[0]['accuracy'] > recordOptDis['accuracy']:
                    self.records[0] = recordOptDis
                    self.records.sort(reverse=True, key=lambda x: x['accuracy'])

    def sortRecords(self):
        self.records.sort(reverse=True, key=lambda x: x['accuracy'])

    def calEliteDistVariableVals(self, sampleSize=1, crossoverProb=0.0):
        opt_dist_variables = self.computeGraph["opt_dist_variables"]

        if self.capacity() < self.size:
            calDistVarVals = {}
            for vd in opt_dist_variables:
                calDistVarVals[vd] = np.exp(np.random.normal(0.0, 0.5, (randomSampleSize, vd.shape[1])) / defaultTemperature)
                calDistVarVals[vd] = calDistVarVals[vd] / np.reshape(np.sum(calDistVarVals[vd], axis=1), [-1, 1])
            return calDistVarVals

        eliteRecords1 = random.choices(self.records, weights=np.arange(self.capacity()) ** 1.5, k=sampleSize)
        eliteRecords2 = random.choices(self.records, weights=np.arange(self.capacity()) ** 1.5, k=sampleSize)
        calDistVarVals = {}
        for vd in opt_dist_variables:
            recordSelectionRand = [random.random() for _ in range(sampleSize)]
            calDistVarVals[vd] = np.array([er2[vd] if tr < crossoverProb else er1[vd] for (er1, er2, tr) in zip(eliteRecords1, eliteRecords2, recordSelectionRand)])
        return calDistVarVals

    def calOptSwitchVarValues(self, sampleSize=1, preserveBaseProb=0.8):
        opt_switch_variables = self.computeGraph["opt_switch_variables"]

        calOptSwitchVarValues = {}
        for vs in opt_switch_variables:
            calOptSwitchVarValues[vs] = np.repeat(np.random.choice(2, (sampleSize,1), p=[preserveBaseProb, 1. - preserveBaseProb]), vs.shape[1], axis=1)
        return calOptSwitchVarValues

    def calGumbelRandomVarValues(self, sampleSize=1):
        random_variables = self.computeGraph["random_variables"]

        calRandVarValues = {}
        for vr in random_variables:
            calRandVarValues[vr] = np.random.gumbel(0.0, 1.0, (sampleSize, vr.shape[1]))
        return calRandVarValues

    def retrieveNoPriority(self, batchSize):
        idxs = np.random.randint(0, len(self.records), size=batchSize)
        return self.records[idxs]

    def retrievePriority(self, batchSize):
        return random.choices(self.records, weights=np.arange(self.capacity()) ** 2.0, k=batchSize)

    def retrieveAll(self):
        return [r for r in self.records]

# Define random initializer for trainable op variables.
train_var_op_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None, dtype=tf.float64) # CG: preliminary test default to 3.0.

# Define random initializer for trainable w variables.
train_var_w_initializer = tf.random_normal_initializer(mean=1.0, stddev=0.02, seed=None, dtype=tf.float64)

# =============================
# Training the computation graph.
# =============================

def prepareFeedDict(dataset, repetition):
    """
    Prepare feed dict based on given dataset and number of repetitions.
    :param dataset:
    :param repetition:
    :return:
    """
    feed_dict = {}
    for k, v in dataset["input"].items():
        feed_dict[k] = np.concatenate([v for i in range(repetition)], axis=0)

    for k, v in dataset["output"].items():
        feed_dict[k] = np.concatenate([v for i in range(repetition)], axis=0)

    return feed_dict

def trainComputationGraph(dataLoader, w_iterations=1, logFile=None, checkRecordFingerprint=True, crossoverProb=0.0):
    """
    Train computation graph.
    :return:
    """
    global defaultTemperature

    from sr.release.build_compute_graph import layerStructureComputeGraph
    from sr.release.build_compute_graph import buildComputeGraph
    from sr.release.build_compute_graph import supportedFunctionSet

    # Get dataset.
    batchSize = dataLoader.getTrainingDataCapacity()
    dataset = dataLoader.getTrainingDataset()

    # Build computation graph.
    computeGraph = buildComputeGraph(dataLoader, supportedFunctionSet, layerStructureComputeGraph, train_var_op_initializer, train_var_w_initializer)

    c1 = computeGraph["c1"]
    temperature = computeGraph["temperature"]
    trainable_op_variables = computeGraph["trainable_op_variables"]
    trainable_w_variables = computeGraph["trainable_w_variables"]
    random_variables = computeGraph["random_variables"]
    opt_dist_variables = computeGraph["opt_dist_variables"]
    opt_dist_values = computeGraph["opt_dist_values"]
    output_values = computeGraph["output_values"]
    opt_switch_variables = computeGraph["opt_switch_variables"]
    trainable_sum_w_variables = computeGraph["trainable_sum_w_variables"]

    logDistLoss = computeGraph["logDistLoss"]
    distLoss = computeGraph["distLoss"]

    # ================================
    # Define loss function and optimizer.
    # ================================
    loss = 0
    absolute_loss = 0
    for y, o in zip(dataLoader.getOutputVariables(), output_values):
        loss = loss + tf.reduce_mean(tf.abs(y - o))
        absolute_loss = absolute_loss + tf.reduce_mean(tf.abs(y - o))

    optimizer1 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    optimizer3 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0003)

    gvs1 = optimizer1.compute_gradients(loss, trainable_op_variables)
    capped_gvs1 = [
        (tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), tf.clip_by_value(grad, -gradientBound, gradientBound)), var)
        for grad, var in gvs1
    ]
    train_op_graph = optimizer1.apply_gradients(capped_gvs1)

    gvs2 = optimizer2.compute_gradients(loss, trainable_w_variables + trainable_sum_w_variables)
    capped_gvs2 = [
        (tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), tf.clip_by_value(grad, -gradientBound, gradientBound)), var)
        for grad, var in gvs2
    ]
    train_w_graph = optimizer2.apply_gradients(capped_gvs2)

    gvs3 = optimizer3.compute_gradients(-logDistLoss, trainable_op_variables)
    capped_gvs3 = [
        (
        tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), tf.clip_by_value(grad, -gradientBound, gradientBound)), var)
        for grad, var in gvs3
    ]
    train_op_log_dist = optimizer3.apply_gradients(capped_gvs3)

    # Initialize tensorflow session.
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=session_conf)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Create the saver.
    saver = tf.compat.v1.train.Saver()

    # Create elite buffer.
    eliteBuffer = EliteBuffer(sess, eliteBufferSize, checkRecordFingerprint, computeGraph)

    # Set up log file in the log directory.
    if logFile is not None:
        temp_directory = "./log"
        os.makedirs(temp_directory, exist_ok=True)

        # Remove all files in the log directory.
        for filename in os.listdir(temp_directory):
            file_path = os.path.join(temp_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        # Create log file in the log directory.
        logFile = os.path.join(temp_directory, logFile)
        logFile = open(logFile, 'w')
        logFile.write("iteration,averageMAE,bestMAE,algorithm\n")
        logFile.flush()

    def saveLearnedGraph(gumbelRandVariableVals, optSwitchVariableVals, eliteDistVariableVals, accuracies,  t):
        temp_index = np.argmin(accuracies)
        temp_feed = {}
        for vr in random_variables:
            temp_feed[vr] = np.array([gumbelRandVariableVals[vr][temp_index]])
        for vs in opt_switch_variables:
            temp_feed[vs] = np.array([optSwitchVariableVals[vs][temp_index]])
        for vd in opt_dist_variables:
            temp_feed[vd] = np.array([eliteDistVariableVals[vd][temp_index]])
        feed_dict[temperature] = defaultTemperature
        tmpDistVals = sess.run(opt_dist_values, feed_dict=feed_dict)

        calDistVals = {}
        for vd, distV in zip(opt_dist_variables, tmpDistVals):
            calDistVals[vd] = distV[0]
        from sr.release.draw_compute_graph import drawComputeGraph
        drawComputeGraph(sess, computeGraph, "./log/test-{0}.png".format(t), calDistVals=calDistVals)

    print("Start training process")

    # =========================
    # Train op parameters only.
    # =========================
    print("\n\n\nTrain op parameters only.")
    totalRecords = []
    opParameterTrainingRecords = []
    totalOpTrainRound = 6
    for temp_op_t in range(totalOpTrainRound):
        print("\n\n\nNew op parameter training round.")
        sess.run(tf.compat.v1.global_variables_initializer())
        eliteBuffer.clear()
        for t in range(8000):
            # ==============
            # Online training.
            # ==============
            # Prepare random variables.
            feed_dict = prepareFeedDict(dataset, randomSampleSize)
            feed_dict[c1] = np.repeat(np.ones((batchSize, 1)), randomSampleSize, axis=0)
            feed_dict[temperature] = defaultTemperature

            eliteDistVariableVals = eliteBuffer.calEliteDistVariableVals(sampleSize=randomSampleSize, crossoverProb=0.0)
            for vd in opt_dist_variables:
                feed_dict[vd] = np.repeat(eliteDistVariableVals[vd], batchSize, axis=0)
            optSwitchVariableVals = eliteBuffer.calOptSwitchVarValues(sampleSize=randomSampleSize, preserveBaseProb=0.0) # 0.8
            for vs in opt_switch_variables:
                feed_dict[vs] = np.repeat(optSwitchVariableVals[vs], batchSize, axis=0)
            gumbelRandVariableVals = eliteBuffer.calGumbelRandomVarValues(sampleSize=randomSampleSize)
            for vr in random_variables:
                feed_dict[vr] = np.repeat(gumbelRandVariableVals[vr], batchSize, axis=0)

            # Train op parameters only.
            sess.run(train_op_graph, feed_dict=feed_dict)

            # Calculate accuracies for storing samples into elite buffer.
            accuracies = []
            for temp_i in range(randomSampleSize):
                feed_dict = prepareFeedDict(dataset, 1)
                feed_dict[c1] = np.ones((batchSize, 1))
                feed_dict[temperature] = defaultTemperature
                for vd in opt_dist_variables:
                    feed_dict[vd] = eliteDistVariableVals[vd][temp_i] * np.ones((batchSize, vd.shape[1]))
                for vs in opt_switch_variables:
                    feed_dict[vs] = optSwitchVariableVals[vs][temp_i] * np.ones((batchSize, vs.shape[1]))
                for vr in random_variables:
                    feed_dict[vr] = gumbelRandVariableVals[vr][temp_i] * np.ones((batchSize, vr.shape[1]))
                accuracies.append(sess.run(absolute_loss, feed_dict=feed_dict))

            # Add random samples into elite buffer.
            eliteBuffer.store(gumbelRandVariableVals, optSwitchVariableVals, eliteDistVariableVals, accuracies)

            if t == 0 or t % 5 == 0:
                print("Current loss in iteration ", t, " is ", np.average(accuracies), " ", eliteBuffer.bestAccuracy())
                print("Average elite accuracy: {0}, size: {1}".format(eliteBuffer.averageAccuracy(), eliteBuffer.capacity()))
                if temp_op_t == (totalOpTrainRound-1) and logFile is not None:
                    logFile.write("{0},{1},{2},{3}\n".format(t, np.average(accuracies), eliteBuffer.bestAccuracy(), "Gumbel-max EQL"))
                    logFile.flush()

        # Information collection after each round of op parameter training.
        totalRecords = totalRecords + eliteBuffer.records
        tmpOpParameterVals = sess.run(trainable_op_variables)
        tmpOpTrainRec = {"accuracy": eliteBuffer.bestAccuracy()}
        for op, opv in zip(trainable_op_variables, tmpOpParameterVals):
            tmpOpTrainRec[op] = opv
        opParameterTrainingRecords.append(tmpOpTrainRec)

    print("\n\n\nCombine all records obtained from training op parameters.")
    eliteBuffer.clear()
    for temp_rec in totalRecords:
        eliteBuffer.recordStore(temp_rec)

    print("\n\n\nReset op parameter values.")
    bestOpTrainRec = opParameterTrainingRecords[np.argmin(np.array([rec['accuracy'] for rec in opParameterTrainingRecords]))]
    for op in trainable_op_variables:
        op.assign(bestOpTrainRec[op])

    # ===============================
    # Train both op and w parameters.
    # ===============================
    print("\n\n\nTrain op and w parameters.")
    bestAccuracy = 0.0
    sess.run(tf.compat.v1.global_variables_initializer())
    for t in range(50000):
        # ==============
        # Online training.
        # ==============
        # Prepare random variables.
        feed_dict = prepareFeedDict(dataset, randomSampleSize)
        feed_dict[c1] = np.repeat(np.ones((batchSize, 1)), randomSampleSize, axis=0)
        feed_dict[temperature] = defaultTemperature

        eliteDistVariableVals = eliteBuffer.calEliteDistVariableVals(sampleSize=randomSampleSize, crossoverProb=crossoverProb)
        for vd in opt_dist_variables:
            feed_dict[vd] = np.repeat(eliteDistVariableVals[vd], batchSize, axis=0)
        optSwitchVariableVals = eliteBuffer.calOptSwitchVarValues(sampleSize=randomSampleSize, preserveBaseProb=0.0) # 0.8
        for vs in opt_switch_variables:
            feed_dict[vs] = np.repeat(optSwitchVariableVals[vs], batchSize, axis=0)
        gumbelRandVariableVals = eliteBuffer.calGumbelRandomVarValues(sampleSize=randomSampleSize)
        for vr in random_variables:
            feed_dict[vr] = np.repeat(gumbelRandVariableVals[vr], batchSize, axis=0)

        # Train compute graph.
        for _ in range(w_iterations):
            sess.run(train_w_graph, feed_dict=feed_dict)
        sess.run(train_op_graph, feed_dict=feed_dict)

        # Calculate accuracies for storing samples into elite buffer.
        accuracies = []
        for temp_i in range(randomSampleSize):
            feed_dict = prepareFeedDict(dataset, 1)
            feed_dict[c1] = np.ones((batchSize, 1))
            feed_dict[temperature] = defaultTemperature
            for vd in opt_dist_variables:
                feed_dict[vd] = eliteDistVariableVals[vd][temp_i] * np.ones((batchSize, vd.shape[1]))
            for vs in opt_switch_variables:
                feed_dict[vs] = optSwitchVariableVals[vs][temp_i] * np.ones((batchSize, vs.shape[1]))
            for vr in random_variables:
                feed_dict[vr] = gumbelRandVariableVals[vr][temp_i] * np.ones((batchSize, vr.shape[1]))
            accuracies.append(sess.run(absolute_loss, feed_dict=feed_dict))

        # Add random samples into elite buffer.
        eliteBuffer.store(gumbelRandVariableVals, optSwitchVariableVals, eliteDistVariableVals, accuracies)

        if t == 0:
            bestAccuracy = np.min(accuracies)
            saveLearnedGraph(gumbelRandVariableVals, optSwitchVariableVals, eliteDistVariableVals, accuracies,  t)
        elif bestAccuracy > np.min(accuracies):
            bestAccuracy = np.min(accuracies)
            saveLearnedGraph(gumbelRandVariableVals, optSwitchVariableVals, eliteDistVariableVals, accuracies,  t)

        if t == 0 or t % 5 == 0:
            print("Current loss in iteration ", t, " is ", np.average(accuracies), " ", bestAccuracy)
            print("Average elite accuracy: {0}, size: {1}".format(eliteBuffer.averageAccuracy(), eliteBuffer.capacity()))
            if logFile is not None:
                logFile.write("{0},{1},{2},{3}\n".format((t+8000), np.average(accuracies), bestAccuracy, "Gumbel-max EQL"))
                logFile.flush()

        eliteDistVariableVals = eliteBuffer.calEliteDistVariableVals(sampleSize=randomSampleSize, crossoverProb=0.0)
        feed_dict = eliteDistVariableVals
        feed_dict[temperature] = defaultTemperature
        sess.run(train_op_log_dist, feed_dict=feed_dict)

    # Save trained models.
    for i, var in enumerate(saver._var_list):
        print('Var {}: {}: {}'.format(i, var.name, sess.run(var)))
    saver.save(sess, './saved_model/saved_variable')

    # Close log file.
    if logFile is not None:
        logFile.close()

    print('Training is complete and model saved.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--witerations', type=int, default=1)
    parser.add_argument('--benchmark', type=str, default='b4')
    parser.add_argument('--temperature', type=float, default=0.666)
    parser.add_argument('--checkRecordFingerprint', type=int, default=1)
    parser.add_argument('--crossoverProb', type=float, default=0.)
    args = parser.parse_args()
    w_iterations = args.witerations
    defaultTemperature = args.temperature
    crossoverProb = args.crossoverProb
    if args.checkRecordFingerprint == 1:
        checkRecordFingerprint = True
    else:
        checkRecordFingerprint = False

    from sr.release.simple_data_loader_b1 import SimpleDataLoaderB1
    from sr.release.simple_data_loader_b2 import SimpleDataLoaderB2
    from sr.release.simple_data_loader_b3 import SimpleDataLoaderB3
    from sr.release.simple_data_loader_b4 import SimpleDataLoaderB4
    benchmarkSets = {
        "b1": SimpleDataLoaderB1,
        "b2": SimpleDataLoaderB2,
        "b3": SimpleDataLoaderB3,
        "b4": SimpleDataLoaderB4
    }

    dataLoader = benchmarkSets[args.benchmark]()
    trainComputationGraph(dataLoader=dataLoader,
                          w_iterations=w_iterations,
                          logFile="out.csv",
                          checkRecordFingerprint=checkRecordFingerprint,
                          crossoverProb=crossoverProb
                          )
