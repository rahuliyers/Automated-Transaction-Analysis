import itertools

class FaceFeaturePreprocessor:
    def __init__(self):
        pass

    def process(self, features):
        samples = []
        if len(features) > 4:
            #do stacking (window over 4 frames)
            for j in range(len(features) - 5):
                s = list(itertools.chain.from_iterable(f[j:(j+4)]))
                samples.append(s)
        else:
            #just flatten features
            samples.append(list(itertools.chain.from_iterable(features)))

        return samples

