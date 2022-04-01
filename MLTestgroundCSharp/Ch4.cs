using MathNet.Numerics.LinearAlgebra;

namespace MLEx {
    static class Ch4 {
        public enum PurityAlgorithm {
            Entropy, Gini, Logit
        }
        public struct Node {
            public enum Op {
                Equal, Greater, Less
            }
            public bool isLeaf = false;
            // for parents
            public int assocAttr = 0;
            public List<Tuple<Op, double, Node>> children = new();
            // for leaf
            public double result = 0.0;
            public Node() { }
            public override string ToString() => isLeaf ? "Leaf: " + result : "Condition: attr" + assocAttr;
        }

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
        public static readonly uint[] discreteCount3 = new uint[] {3, 3, 3, 3, 3, 2, Continuous, Continuous };
        public static readonly HashSet<int> defaultAttrSet3 = new() { 0, 1, 2, 3, 4, 5, 6, 7 };
        public static readonly List<Vector<double>> defaultData2Train = new() {
            // 乌黑 蜷缩 清脆 清晰 凹陷 硬滑
            vb.Dense(new double[] { 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }),
            vb.Dense(new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0 }),
            vb.Dense(new double[] { 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }),
            vb.Dense(new double[] { 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0 }),
            vb.Dense(new double[] { 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 }),
            vb.Dense(new double[] { 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0 }),
            vb.Dense(new double[] { 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0 }),
        };
        public static readonly List<Vector<double>> defaultData2Test = new() {
            // 乌黑 蜷缩 清脆 清晰 凹陷 硬滑
            vb.Dense(new double[] { 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0 }),
            vb.Dense(new double[] { 1.0, 0.5, 0.0, 0.5, 0.5, 1.0, 0.0 }),
            vb.Dense(new double[] { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0 }),
            vb.Dense(new double[] { 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0 }),
            vb.Dense(new double[] { 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0 }),
        };
        public static readonly uint[] discreteCount2 = new uint[] { 3, 3, 3, 3, 3, 2 };
        public static readonly HashSet<int> defaultAttrSet2 = new() { 0, 1, 2, 3, 4, 5 };

        /// <summary>
        /// 浮点数精确到小数点六位的等于判断
        /// </summary>
        static bool FEq(double a, double b) => Math.Abs(a - b) < 1E-6;
        static bool SameOnSpecifiedAttrSet(List<Vector<double>> d, HashSet<int> attrs) {
            foreach (var a in attrs)
                for (int i = 1; i < d.Count; ++i)
                    if (!FEq(d[i - 1][a], d[i][a]))
                        return false;
            return true;
        }
        /// <summary>
        /// 讲其属性等于特定值的所有样本提取出来
        /// </summary>
        static List<Vector<double>> DataWhoseAttrIs(List<Vector<double>> data, int attr, double val) {
            List<Vector<double>> result = new();
            foreach (var d in data)
                if (FEq(d[attr], val))
                    result.Add(d);
            return result;
        }
        /// <summary>
        /// 以某属性的特定值为划分点，讲样本划分为两个
        /// </summary>
        static Tuple<List<Vector<double>>, List<Vector<double>>> DivideDataByAttrAtPoint(List<Vector<double>> data, int attr, double divPnt) {
            List<Vector<double>> neg = new(), pos = new();
            foreach (var d in data)
                if (d[attr] < divPnt) neg.Add(d);
                else pos.Add(d);
            return new(neg, pos);
        }
        /// <summary>
        /// 样本中出现最多的类别值
        /// </summary>
        static double ClassHavingMostSamples(List<Vector<double>> data) {
            Dictionary<double, int> counts = new();
            foreach (var d in data)
                if (!counts.ContainsKey(d.Last())) counts[d.Last()] = 1;
                else ++counts[d.Last()];
            return counts.Aggregate((a, b) => a.Value > b.Value ? a : b).Key;
        }
        /// <summary>
        /// 信息熵
        /// </summary>
        static double Entropy(List<Vector<double>> data) {
            Dictionary<double, int> counts = new();
            foreach (var d in data)
                if (!counts.ContainsKey(d.Last())) counts[d.Last()] = 1;
                else ++counts[d.Last()];
            Dictionary<double, double> p = new();
            foreach (var kv in counts)
                p[kv.Key] = (double)kv.Value / data.Count;
            double sum = 0;
            foreach (var v in p.Values)
                sum += v * Math.Log2(v);
            return -sum;

        }
        /// <summary>
        /// 得出连续值属性的最优划分点和其划分增益
        /// </summary>
        static Tuple<double, double> OptDivPntAndContinuousGain(List<Vector<double>> data, int attr) {
            List<Vector<double>> dneg = new(), dpos = new(data);
            dpos.Sort((a, b) => a[attr] > b[attr] ? 1 : -1);
            Tuple<int, double> min = new(0, double.PositiveInfinity);
            for (int i = 0; dpos.Count > 1; ++i) {
                dneg.Add(dpos[0]);
                dpos.RemoveAt(0);

                double v = (Entropy(dneg) * dneg.Count + Entropy(dpos) * dpos.Count) / data.Count;
                if (v < min.Item2)
                    min = new(i, v);
            }
            return new((data[min.Item1][attr] + data[min.Item1 + 1][attr]) / 2, Entropy(data) - min.Item2);
        }
        /// <summary>
        /// 信息增益
        /// </summary>
        static double Gain(List<Vector<double>> data, int attr) {
            uint selCnt = discreteCount3[attr];
            if (selCnt != Continuous) {  // for discrete attribute
                double sum = 0;
                for (int i = 0; i < selCnt; ++i) {
                    var division = DataWhoseAttrIs(data, attr, Math.Round((double)i / (selCnt - 1), 6));
                    sum += Entropy(division) * division.Count / data.Count;
                }
                return Entropy(data) - sum;
            } else { // for continuous attribute
                return OptDivPntAndContinuousGain(data, attr).Item2;
            }
        }

        /// <summary>
        /// 基尼值
        /// </summary>
        static double Gini(List<Vector<double>> data) {
            Dictionary<double, int> counts = new();
            foreach (var d in data)
                if (!counts.ContainsKey(d.Last())) counts[d.Last()] = 1;
                else ++counts[d.Last()];
            Dictionary<double, double> p = new();
            foreach (var kv in counts)
                p[kv.Key] = (double)kv.Value / data.Count;
            double sum = 0;
            foreach (var v in p.Values)
                sum += v * v;
            return 1-sum;
        }

        /// <summary>
        /// 基尼系数
        /// </summary>
        static double GiniIndex(List<Vector<double>> data, int attr) {
            uint selCnt = discreteCount3[attr];
            if (selCnt != Continuous) {  // for discrete attribute
                double sum = 0;
                for (int i = 0; i < selCnt; ++i) {
                    var division = DataWhoseAttrIs(data, attr, Math.Round((double)i / (selCnt - 1), 6));
                    sum += Gini(division) * division.Count / data.Count;
                }
                return sum;
            } else { // for continuous attribute
                throw new Exception("不支持连续值属性");
            }
        }           

        /// <summary>
        /// 得出最优划分属性
        /// </summary>
        static int OptimaizedDividingAttr(List<Vector<double>> data, HashSet<int> attrs, PurityAlgorithm purAlgo) {
            switch (purAlgo) {
                case PurityAlgorithm.Entropy:
                    Tuple<int, double> maxGain = new(0, double.NegativeInfinity);
                    foreach (var a in attrs) {
                        double gain = Gain(data, a);
                        if (gain > maxGain.Item2)
                            maxGain = new(a, gain);
                    }
                    return maxGain.Item1;

                case PurityAlgorithm.Gini:
                    Tuple<int, double> minGiniIndex = new(0, double.PositiveInfinity);
                    foreach (var a in attrs) {
                        double giniIndex = GiniIndex(data, a);
                        if (giniIndex < minGiniIndex.Item2)
                            minGiniIndex = new(a, giniIndex);
                    }
                    return minGiniIndex.Item1;

                case PurityAlgorithm.Logit:
                    return 0;

                default:
                    return 0;
            }
        }
        /// <summary>
        /// 递归构建决策树
        /// </summary>
        public static Node TreeGenerate(List<Vector<double>> data, HashSet<int> attrs, PurityAlgorithm purAlgo) {
            Node n = new();
            bool flag = false;
            // 若当前节点包含的样本全属于同一类别，则无需划分，作叶节点
            for (int i = 1; i < data.Count; ++i) {
                if (!FEq(data[i - 1].Last(), data[i].Last())) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                n.isLeaf = true;
                n.result = data[0].Last();
                return n;
            }
            // 当前属性集为空，或所有样本再属性集上取值相同，作叶节点
            if (attrs.Count == 0 || SameOnSpecifiedAttrSet(data, attrs)) {
                n.isLeaf = true;
                n.result = ClassHavingMostSamples(data);
                return n;
            }
            // 需要划分的情形
            n.isLeaf = false;
            n.assocAttr = OptimaizedDividingAttr(data, attrs, purAlgo);
            HashSet<int> newAttrSet = attrs;
            newAttrSet.Remove(n.assocAttr);
            uint selCnt = discreteCount3[n.assocAttr];
            if (selCnt != Continuous) { // 离散值情况，依据图2算法
                for (int i = 0; i < selCnt; ++i) {
                    double v = Math.Round((double)i / (selCnt - 1), 6);
                    var division = DataWhoseAttrIs(data, n.assocAttr, v);
                    if (division.Count == 0) {  // 若当前划分条件下划分不出样本
                        Node child = new();
                        child.isLeaf = true;
                        child.result = ClassHavingMostSamples(data);
                        n.children.Add(new(Node.Op.Equal, v, child));
                    } else {
                        n.children.Add(new(Node.Op.Equal, v, TreeGenerate(division, newAttrSet, purAlgo)));
                    }
                }
            } else { // 连续值情况，依据4.4.1描述的算法
                double divPnt = OptDivPntAndContinuousGain(data, n.assocAttr).Item1;
                var division = DivideDataByAttrAtPoint(data, n.assocAttr, divPnt);
                n.children.Add(new(Node.Op.Less, divPnt, TreeGenerate(division.Item1, newAttrSet, purAlgo)));
                n.children.Add(new(Node.Op.Greater, divPnt, TreeGenerate(division.Item2, newAttrSet, purAlgo)));
            }
            return n;
        }
        public static void Classify(Node root, Vector<double> data) {
            Node cur = root;
            while (!cur.isLeaf) {
                foreach (var child in cur.children) {
                    if (child.Item1 == Node.Op.Equal && FEq(data[cur.assocAttr], child.Item2)) {
                        cur = child.Item3;
                        break;
                    } else if (child.Item1 == Node.Op.Less && data[cur.assocAttr] < child.Item2) {
                        cur = child.Item3;
                        break;
                    } else if (child.Item1 == Node.Op.Greater && data[cur.assocAttr] > child.Item2) {
                        cur = child.Item3;
                        break;
                    }
                }
            }
            Console.WriteLine("分类值为：" + cur.result);
        }
    }
}