import DRMM
from DRMM import DRMMBlockHierarchy, dataStream, DataIn

def createModel(session, train, args):
    model = None
    if args.model_type == "baseline":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":7,"stride":2},   #input seq. length 32, output length 16
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=4,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "extended":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":6,"kernelSize":5,"stride":2},   #input seq. length 64, output length 32
                {"nClasses":256,"nLayers":8,"kernelSize":5,"stride":2},   #in 32, out 16
                {"nClasses":256,"nLayers":10,"kernelSize":5,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=10,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.002)
    elif args.model_type == "extended-lite":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},   #input seq. length 64, output length 32
                {"nClasses":256,"nLayers":4,"kernelSize":7,"stride":2},   #in 32, out 16
                {"nClasses":256,"nLayers":5,"kernelSize":7,"stride":2},   #in 16, out 8
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel-extended":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "large-kernel-extra-layer":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":3,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":9,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=6,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.004)
    elif args.model_type == "xl-kernel":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "deep-stack":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":2,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":3,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":4,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":7,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=6,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    elif args.model_type == "supersized":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":4,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":8,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":10,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=10,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.004)
    elif args.model_type == "supersized-2":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":4,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":6,"kernelSize":9,"stride":2},
                {"nClasses":256,"nLayers":8,"kernelSize":7,"stride":2},
                {"nClasses":256,"nLayers":10,"kernelSize":5,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=10,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.004)
    elif args.model_type == "all-fives":
        model = DRMMBlockHierarchy(session,
            inputs=dataStream(
                dataType="continuous",
                shape=[None,args.sequence_length,args.data_dimension],
                useGaussianPrior=True,
                useBoxConstraints=True
            ),
            blockDefs=[
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
                {"nClasses":256,"nLayers":5,"kernelSize":11,"stride":2},
            ],
            lastBlockClasses=256,
            lastBlockLayers=5,
            train=train,    #if False, optimization ops will not be created, which saves some time
            initialLearningRate=0.005)
    return model

# Get a list of keyframe timesteps for an animation
def getKeyFrameTimesteps(animation, args):
    timesteps = None
    if args.keyframe_mode == 'fixed':
        timesteps = getFixedKeyframes(args.sequence_length, args.keyframe_count)
    elif args.keyframe_mode == 'calculated':
        timesteps = calculateKeyFrames(animation, args.keyframe_count, args.keyframe_calculation_interval)
    if args.debug:
        print(timesteps)
    return timesteps

# Get a list of keyframes with regular intervals for an animation of given length
def getFixedKeyframes(sequence_length, keyframe_count):
    timesteps = [0]
    for i in range(1,keyframe_count):
        next_frame = i*sequence_length//keyframe_count
        timesteps.append(next_frame)
    timesteps.append(sequence_length-1)
    return timesteps

# Calculate the best keyframes for a sequence based on change rates
def calculateKeyFrames(sequence, keyframe_count, interval):
    change_rates = np.zeros(sequence.shape[0])
    # Calculate change rates for each frame except first and last
    for i in range(interval,sequence.shape[0]-interval):
        current_frame = sequence[i]
        prev_changes = np.subtract(current_frame, sequence[i-interval])
        next_changes = np.subtract(current_frame, sequence[i+interval])
        total_changes = np.absolute(prev_changes + next_changes)
        change_rates[i] = np.sum(total_changes)
    timesteps = np.zeros(1)
    idx = np.argpartition(change_rates, -keyframe_count-1)
    timesteps = np.sort(np.concatenate((timesteps, idx[-(keyframe_count-1):])))
    timesteps = np.append(timesteps, [sequence.shape[0]-1]).astype(int)
    return timesteps
