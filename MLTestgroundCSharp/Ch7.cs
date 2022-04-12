using MathNet.Numerics.LinearAlgebra;

namespace MLEx {
    static class Ch7 {
        static readonly VectorBuilder<double> vb = Vector<double>.Build;
        public const uint Continuous = uint.MaxValue;
        public static readonly List<Vector<double>> defaultData3 = new() {
            // 乌黑 蜷缩 清脆 清晰 凹陷 硬滑
            vb.Dense(new double[] { 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.697, 0.460, 1.0 }),
            vb.Dense(new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.774, 0.376, 1.0 }),
            vb.Dense(new double[] { 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.634, 0.264, 1.0 }),
            vb.Dense(new double[] { 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 0.608, 0.318, 1.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.556, 0.215, 1.0 }),
            vb.Dense(new double[] { 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 0.403, 0.237, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.481, 0.149, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.437, 0.211, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.0, 0.5, 0.5, 1.0, 0.666, 0.091, 0.0 }),
            vb.Dense(new double[] { 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.243, 0.267, 0.0 }),
            vb.Dense(new double[] { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.245, 0.057, 0.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.343, 0.099, 0.0 }),
            vb.Dense(new double[] { 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.639, 0.161, 0.0 }),
            vb.Dense(new double[] { 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.657, 0.198, 0.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.360, 0.370, 0.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.593, 0.042, 0.0 }),
            vb.Dense(new double[] { 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.719, 0.103, 0.0 }),
        };
        public static readonly uint[] discreteCount3 = new uint[] { 3, 3, 3, 3, 3, 2, Continuous, Continuous };

        static double Normal(double mean, double variance, double x) => Math.Exp(-((x - mean) * (x - mean)) / (2 * variance * variance)) / Math.Sqrt(2 * Math.PI * variance);

        public static double NaiveBayes(List<Vector<double>> train, Vector<double> test, uint[] discrete) {
            Dictionary<double, double> logPosterior = new();


            Dictionary<double, int> classCnt = new();
            foreach (var d in train)
                if (classCnt.ContainsKey(d[^1])) ++classCnt[d[^1]];
                else classCnt[d[^1]] = 1;
            foreach (var kv in classCnt)
                logPosterior[kv.Key] = Math.Log((double)(kv.Value + 1) / (train.Count + classCnt.Count)); // 这里log中的值即为使用拉普拉斯修正的先验概率(prior)
                                                                                                                 // logPosterior 之后会与下面 for 循环每一轮算出的值连加

            for (int i = 0; i < test.Count; ++i) {
                if (discrete[i] != Continuous) {
                    // <A,B>: Cnt(attr_i = data[attr_i] | class = A) = B
                    Dictionary<double, int> cnt = new();
                    foreach (var d in train)
                        if (d[i] == test[i])
                            if (cnt.ContainsKey(d[^1])) ++cnt[d[^1]];
                            else cnt[d[^1]] = 1;
                    foreach (var kv in cnt)
                        logPosterior[kv.Key] += Math.Log((double)(kv.Value + 1) / (classCnt[kv.Key] * classCnt.Count)); // 拉普拉斯修正
                } else {
                    Dictionary<double, double> mean = new(), variance = new();
                    foreach (var d in train)
                        if (mean.ContainsKey(d[^1])) mean[d[^1]] += d[i] / classCnt[d[^1]];
                        else mean[d[^1]] = d[i] / classCnt[d[^1]];
                    foreach (var d in train)
                        if (variance.ContainsKey(d[^1])) variance[d[^1]] += (d[i] - mean[d[^1]]) * (d[i] - mean[d[^1]]) / classCnt[d[^1]];
                        else variance[d[^1]] = (d[i] - mean[d[^1]]) * (d[i] - mean[d[^1]]) / classCnt[d[^1]];
                    foreach (var kv in mean)
                        logPosterior[kv.Key] += Math.Log(Normal(kv.Value, variance[kv.Key], test[i]));
                }
            }

            KeyValuePair<double, double> max = new(0.0, double.MinValue);
            foreach (var kv in logPosterior)
                if (kv.Value > max.Value) max = kv;
            return max.Key;
        }

        public static double AODE(List<Vector<double>> train, Vector<double> test, uint[] discrete) {
            Dictionary<double, double> posterior = new();

            HashSet<double> classSet = new();
            HashSet<double>[] attrSet = new HashSet<double>[test.Count];
            foreach (var d in train)
                classSet.Add(d[^1]);
            for (int i = 0; i < test.Count; ++i) {
                if (discrete[i] == Continuous) continue;
                attrSet[i] = new HashSet<double>();
                foreach (var d in train)
                    attrSet[i].Add(d[i]);
            }

            foreach (var c in classSet) {
                posterior[c] = 0;
                for (int i = 0; i < test.Count; ++i) {
                    if (discrete[i] == Continuous) continue;
                    List<Vector<double>> dcxi = new();
                    foreach (var d in train)
                        if (d[^1] == c && d[i] == test[i]) dcxi.Add(d);
                    double prior = (double)(dcxi.Count + 1) / (train.Count + classSet.Count * attrSet[i].Count);
                    double logLikelihood = 0;
                    for (int j = 0; j < test.Count; ++j) {
                        if (discrete[j] == Continuous) continue;
                        int cnt = 0;
                        foreach (var dprime in dcxi)
                            if (dprime[j] == test[j]) ++cnt;
                        logLikelihood += Math.Log((double)(cnt + 1) / (dcxi.Count + attrSet[j].Count));
                    }
                    posterior[c] += prior * logLikelihood;
                }
            }


            KeyValuePair<double, double> max = new(0.0, double.MinValue);
            foreach (var kv in posterior)
                if (kv.Value > max.Value) max = kv;
            return max.Key;
        }

    }
}