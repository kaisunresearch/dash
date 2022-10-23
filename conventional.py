import json
import math
from sklearn import ensemble
import random
import xgboost as xgb
import argparse

auto_split = [0.05, 0.05]

def work(subset = "dn", m = "xgb", evaldev = True, evaltest = False):
    assert(subset in ["dn", "ea", "nft"])
    assert(m in ["rf", "ab", "xgb", "mean"])
    
    random.seed(42)

    if subset == "dn":
        punycode = True
        delimiter = "."
        with open("data/v1.0/dash_dn.json", "r", encoding='utf8') as f:
            data = json.load(f)
    elif subset == "ea":
        punycode = False
        delimiter = "@"
        split = None
        with open("data/v1.0/dash_ea.json", "r", encoding='utf8') as f:
            data = json.load(f)
    elif subset == "nft":
        punycode = False
        delimiter = "."
        split = None
        with open("data/v1.0/dash_nft.json", "r", encoding='utf8') as f:
            data = json.load(f)

    data = [[x["asset"], x["price"], x["date"]] for x  in data["data"]]

    split = [data[int(len(data)*(1-auto_split[0]-auto_split[1]))-1][2], \
             data[int(len(data)*(1-auto_split[1]))-1][2]]

    train, dev, test = [], [], []
    for i in range(len(data)):
        if data[i][2] <= split[0]:
            train += [data[i]]
        elif data[i][2] <= split[1]:
            dev += [data[i]]
        else:
            test += [data[i]]

    print(len(data), split, len(train), len(dev), len(test))

    suffixnum = {}
    for i in range(len(train)):
        tld = delimiter.join(train[i][0].split(delimiter)[1:])
        if tld not in suffixnum:
            suffixnum[tld] = 0
        suffixnum[tld] += 1

    suffixmap = {}
    for tld in suffixnum:
        if suffixnum[tld] >= 100:
            suffixmap[tld] = len(suffixmap)

    with open("features.json", "r", encoding='utf8') as f:
        features = json.load(f)

    def get_features(data):
        X = []
        Y = []
        for i in range(len(data)):
            y = math.log(float(data[i][1]))
            x = []
            tld = delimiter.join(data[i][0].split(delimiter)[1:])
            name = data[i][0].split(delimiter)[0]

            # "suffix"
            x += [1 if tld not in suffixmap else 0]
            for i in range(len(suffixmap)):
                x += [1 if tld in suffixmap and suffixmap[tld] == i else 0]

            # "length"
            x += [len(name)]

            # "character"
            if punycode:
                x += [1 if name.startswith("xn--") else 0]
                x += [1 if "-" in name.encode().decode("idna") else 0]
            else:
                x += [1 if name.isascii() else 0]
                x += [1 if "-" in name else 0]
            x += [1 if name.isdigit() else 0]
            x += [1 if name.isalpha() else 0]

            # "number of tokens"
            x += [features["#token"][name]]

            # "vocabulary"
            x += [features["glove"][name]]
            x += [features["adult"][name]]

            # "TLD count"
            x += [features["tldcnt"][name]]

            # "trademark"
            x += [features["trademark"][name]]
            
            X += [x]
            Y += [y]
        return X, Y

    trainX, trainY = get_features(train)
    devX, devY = get_features(dev)
    testX, testY = get_features(test)
    weights = [1 for i in range(len(trainY))]

    class Mean():
        def __init__(self):
            pass
        def fit(self, X, Y, sample_weight):
            y = 0
            n = 0
            for i in range(len(Y)):
                y += sample_weight[i] * Y[i]
                n += sample_weight[i]
            self.y = float(y) / float(n)
        def predict(self, X):
            Y = []
            for i in range(len(X)):
                Y += [self.y]
            return Y
            
    if m == "rf":
        model = ensemble.RandomForestRegressor(random_state=0)
    elif m == "ab":
        model  =ensemble.AdaBoostRegressor(random_state=0)
    elif m == "xgb":
        model = xgb.XGBRegressor(random_state=0)
    elif m == "mean":
        model = Mean()
        
    model.fit(trainX, trainY, sample_weight = weights)

    if evaldev:
        predict = model.predict(devX)
        msle, cnt = 0, 0
        for i in range(len(devY)):
            msle += (predict[i] - devY[i]) ** 2
            cnt += 1
        msle /= cnt
        print("dev", m, msle)

    if evaltest:
        predict = model.predict(testX)
        msle, cnt = 0, 0
        for i in range(len(testY)):
            msle += (predict[i] - testY[i]) ** 2
            cnt += 1
        msle /= cnt
        print("test", m, msle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", default="DN", type=str, required=False, help="The subset of DASH (DN: DASH_DN, EA: DASH_EA, NFT: DASH_NFT)")
    parser.add_argument("--model", default="XGB", type=str, required=False, help="The model to be trained (XGB: XGBoost, AB: AdaBoost, RF: Random Forest, Mean: Mean Value Baseline)")
    parser.add_argument("--evaldev", default=True, type=bool, required=False, help="Evaluation on the dev set")
    parser.add_argument("--evaltest", default=False, type=bool, required=False, help="Evaluation on the test set")

    args = parser.parse_args()

    work(args.subset.lower(), args.model.lower(), args.evaldev, args.evaltest)
    
