using MathNet.Numerics.LinearAlgebra;

namespace MLEx {
    static class Ch9 {
        static readonly Random rand = new Random((int)DateTime.Now.Ticks);
        static readonly VectorBuilder<double> vb = Vector<double>.Build;

        public static List<Vector<double>> defaultData4 = new() {
            vb.Dense(new double[] { 0.697, 0.460 }),
            vb.Dense(new double[] { 0.774, 0.376 }),
            vb.Dense(new double[] { 0.634, 0.264 }),
            vb.Dense(new double[] { 0.608, 0.318 }),
            vb.Dense(new double[] { 0.556, 0.215 }),
            vb.Dense(new double[] { 0.403, 0.237 }),
            vb.Dense(new double[] { 0.481, 0.149 }),
            vb.Dense(new double[] { 0.437, 0.211 }),
            vb.Dense(new double[] { 0.666, 0.091 }),
            vb.Dense(new double[] { 0.243, 0.267 }),
            vb.Dense(new double[] { 0.245, 0.057 }),
            vb.Dense(new double[] { 0.343, 0.099 }),
            vb.Dense(new double[] { 0.639, 0.161 }),
            vb.Dense(new double[] { 0.657, 0.198 }),
            vb.Dense(new double[] { 0.360, 0.370 }),
            vb.Dense(new double[] { 0.593, 0.042 }),
            vb.Dense(new double[] { 0.719, 0.103 }),
            vb.Dense(new double[] { 0.359, 0.188 }),
            vb.Dense(new double[] { 0.339, 0.241 }),
            vb.Dense(new double[] { 0.282, 0.257 }),
            vb.Dense(new double[] { 0.748, 0.232 }),
            vb.Dense(new double[] { 0.714, 0.346 }),
            vb.Dense(new double[] { 0.483, 0.312 }),
            vb.Dense(new double[] { 0.478, 0.437 }),
            vb.Dense(new double[] { 0.525, 0.369 }),
            vb.Dense(new double[] { 0.751, 0.489 }),
            vb.Dense(new double[] { 0.532, 0.472 }),
            vb.Dense(new double[] { 0.473, 0.376 }),
            vb.Dense(new double[] { 0.725, 0.445 }),
            vb.Dense(new double[] { 0.446, 0.459 }),
        };

        /// <summary>
        /// 给定簇划分的数目（即 k 值），使用 k 均值算法进行聚类
        /// </summary>
        /// <param name="data">样本集</param>
        /// <param name="k">聚类数</param>
        /// <param name="threshold">在计算新均值向量的环节中，每个均值向量移动的距离之和小于何值，则停止</param>
        /// <returns>包含 k 个簇的簇划分</returns>
        public static List<Vector<double>>[] KMeans(List<Vector<double>> data, int k, double threshold) => KMeansGetClustersAndMeans(data, k, threshold).Item1;

        public static Tuple<List<Vector<double>>[], Vector<double>[]> KMeansGetClustersAndMeans(List<Vector<double>> data, int k, double threshold) {
            List<Vector<double>>[] clusters;
            Vector<double>[] means = new Vector<double>[k];

            HashSet<double> selected = new();
            for (int i = 0; i < k; ++i) {
                int selection = rand.Next(data.Count);
                if (!selected.Contains(selection)) means[i] = data[selection];
                else --i;
            }

            while (true) {
                clusters = new List<Vector<double>>[k];
                for (int i = 0; i < clusters.Length; ++i)
                    clusters[i] = new();

                foreach (var x in data) {
                    double[] d = new double[k];
                    for (int i = 0; i < k; ++i)
                        d[i] = (x - means[i]).L2Norm();
                    Tuple<int, double> min = new(0, double.MaxValue);
                    for (int i = 0; i < k; ++i)
                        if (d[i] < min.Item2)
                            min = new(i, d[i]);
                    clusters[min.Item1].Add(x);
                }

                Vector<double>[] newMeans = new Vector<double>[k];
                for (int i = 0; i < k; ++i) {
                    newMeans[i] = vb.Dense(data[0].Count, 0.0);
                    foreach (var x in clusters[i])
                        newMeans[i] += x / clusters[i].Count;
                }

                double sumOfDist = 0;
                for (int i = 0; i < k; ++i)
                    sumOfDist += (newMeans[i] - means[i]).L2Norm();
                means = newMeans;
                if (sumOfDist < threshold)
                    break;
            }

            return new(clusters, means);
        }

        /// <summary>
        /// 计算轮廓系数，分析簇划分的散度
        /// </summary>
        /// <param name="clusters">簇划分</param>
        /// <param name="means">各个簇的均值向量</param>
        /// <returns>轮廓系数</returns>
        static double SilhouetteCoefficient(List<Vector<double>>[] clusters, Vector<double>[] means) {
            int dataCount = clusters.Sum(x => x.Count);
            double result = 0;
            for (int i = 0; i < clusters.Length; ++i) {
                foreach (var x in clusters[i]) {
                    double a = 0;
                    foreach (var xPrime in clusters[i])
                        a += (x - xPrime).L2Norm() / (clusters[i].Count - 1);

                    double[] bs = new double[clusters.Length];
                    for (int j = 0; j < clusters.Length; ++j) {
                        if (i == j) {
                            bs[j] = double.MaxValue;
                            continue;
                        }
                        bs[j] = 0;
                        foreach (var xPrime in clusters[j]) {
                            bs[j] += (x-xPrime).L2Norm() / clusters[j].Count;
                        }
                    }
                    double b = bs.Min();
                    result += (b - a) / (a > b ? a : b) / dataCount;
                }
            }
            return result;
        }

        /// <summary>
        /// 利用轮廓系数度量，自动选择一个较优的簇划分数目（即 k 值），使用 k 均值算法进行聚类
        /// </summary>
        /// <param name="data">样本集</param>
        /// <param name="threshold">在计算新均值向量的环节中，每个均值向量移动的距离之和小于何值，则停止</param>
        /// <returns>包含 k 个簇的簇划分</returns>
        public static List<Vector<double>>[] KMeans(List<Vector<double>> data, double threshold) {
            List<Vector<double>>[] formerClusters = Array.Empty<List<Vector<double>>>(), clusters;
            double formerSC = -1;
            for (int k = 2; ; ++k) {
                var t = KMeansGetClustersAndMeans(data, k, threshold);
                clusters = t.Item1;
                var means = t.Item2;
                double sc = SilhouetteCoefficient(clusters, means);
                if (sc > formerSC) {
                    formerSC = sc;
                    formerClusters = clusters;
                } else {
                    return formerClusters;
                }
            }
        }

    }
}
