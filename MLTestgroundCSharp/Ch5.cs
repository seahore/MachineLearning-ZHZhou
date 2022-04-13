using MathNet.Numerics.LinearAlgebra;

namespace MLEx {
    static class Ch5 {
        static VectorBuilder<double> vb = Vector<double>.Build;
        static MatrixBuilder<double> mb = Matrix<double>.Build;
        public struct Sample {
            public IList<double> Input;
            public IList<double> ExpectedOutput;
            public Sample(IList<double> i, IList<double> o) {
                Input = i;
                ExpectedOutput = o;
            }

        }
        public static IEnumerable<T> RandomSort<T>(IEnumerable<T> list) {
            var rand = new Random();
            var newList = new List<T>();
            foreach (var item in list)
                newList.Insert(rand.Next(newList.Count), item);
            return newList;
        }

        public static readonly List<Sample> defaultData3 = new() {
            // 乌黑 蜷缩 清脆 清晰 凹陷 硬滑
            new Sample(new List<double> { 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.697, 0.460 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.774, 0.376 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.634, 0.264 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 0.608, 0.318 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.556, 0.215 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 0.403, 0.237 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.481, 0.149 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.437, 0.211 }, new List<double> { 1.0 }),
            new Sample(new List<double> { 1.0, 0.5, 0.0, 0.5, 0.5, 1.0, 0.666, 0.091 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.243, 0.267 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.245, 0.057 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.343, 0.099 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.639, 0.161 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.657, 0.198 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 1.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.360, 0.370 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.593, 0.042 }, new List<double> { 0.0 }),
            new Sample(new List<double> { 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.719, 0.103 }, new List<double> { 0.0 }),
        };
        public class BPNetwork {
            readonly Random _rand;
            /// <summary>
            /// 层数
            /// </summary>
            public int LayersCount => _neuronsCnt.Length;
            /// <summary>
            /// 输入向量的维度，即输入层神经元个数
            /// </summary>
            public int InputDim => _neuronsCnt[0];
            /// <summary>
            /// 输出向量的维度，即输出层神经元个数
            /// </summary>
            public int OutputDim => _neuronsCnt[^1];
            /// <summary>
            /// 每层的神经元个数
            /// </summary>
            public int[] NeuronsCount => _neuronsCnt;
            int[] _neuronsCnt;
            /// <summary>
            /// weights[l][i][j]: 第 l 层第 i 个神经元向下一层第 j 个神经元输入的权值
            /// </summary>
            public Matrix<double>[] Weights => _weights;
            Matrix<double>[] _weights;
            /// <summary>
            /// thresholds[l][i]: 第 l 层第 i 个神经元的阈值
            /// </summary>
            public Vector<double>[] Thresholds => _thresholds;
            Vector<double>[] _thresholds;

            delegate double ActivationFunc(double x);
            static Vector<double> ApplyToVec(ActivationFunc f, Vector<double> v) => vb.Dense(v.Select(x => f(x)).ToArray());
            static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
            double dSigmoid(double x) => Math.Exp(-x) / Math.Pow(1.0 + Math.Exp(-x), 2);

            public BPNetwork(int[] neurons, Matrix<double>[]? w = null, Vector<double>[]? th = null) {
                _rand = new Random();
                _neuronsCnt = neurons;
                if (w is not null) _weights = w;
                else {
                    _weights = new Matrix<double>[LayersCount - 1];
                    for (int l = 0; l < LayersCount - 1; ++l) {
                        _weights[l] = mb.Dense(_neuronsCnt[l + 1], _neuronsCnt[l]);
                        for (int i = 0; i < _neuronsCnt[l + 1]; ++i)
                            for (int j = 0; j < _neuronsCnt[l]; j++)
                                _weights[l][i, j] = _rand.NextDouble()*2.0 - 1.0;
                    }
                }
                if (th is not null) _thresholds = th;
                else {
                    _thresholds = new Vector<double>[LayersCount];
                    for (int l = 1; l < LayersCount; ++l) {
                        _thresholds[l] = vb.Dense(_neuronsCnt[l]);
                        for (int i = 0; i < _neuronsCnt[l]; i++)
                            _thresholds[l][i] = _rand.NextDouble()*2.0 - 1.0;
                    }
                }
            }

            Vector<double>[] GradOfEachLayer(Sample sample, Vector<double>[] outputs) {
                Vector<double>[] grad = new Vector<double>[LayersCount];
                for (int l = 0; l < LayersCount; ++l)
                    grad[l] = vb.Dense(_neuronsCnt[l]);
                // 计算输出层的梯度
                // g_out = yHat ⊙ (ones - yHat) ⊙ (y - yHat) = diag(yHat) diag(ones - yHat) (y - yHat)
                grad[^1] = mb.Diagonal(outputs[^1].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[^1], 1.0) - outputs[^1]).ToArray()) * (vb.Dense(sample.ExpectedOutput.ToArray()) - outputs[^1]);
                // 依次计算输出以下层的梯度
                // g_l = diag(o_l) diag(ones - o_l) W^T g_{l+1}
                for (int l = LayersCount - 2; l >= 0; --l)
                    grad[l] = mb.Diagonal(outputs[l].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[l], 1.0) - outputs[l]).ToArray()) * _weights[l].Transpose() * grad[l + 1];
                return grad;
            }

            /// <summary>
            /// 用随机梯度下降（标准误差逆传播）算法训练神经网络
            /// </summary>
            public void SGD(IEnumerable<Sample> samples, double learningRate, int epoch) {
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }
                for (int t = 0; t < epoch; ++t) {
                    samples = RandomSort(samples);
                    foreach (var s in samples) {
                        // 获取每层输出，当然也包含最后的输出
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        Vector<double>[] grad = GradOfEachLayer(s, outputs);
                        // 更新权重
                        // W_l = W_l + η g_{l+1} o_l^T
                        for (int l = 0; l < LayersCount - 1; ++l)
                            _weights[l] += learningRate * grad[l + 1].ToColumnMatrix() * outputs[l].ToRowMatrix();
                        // 更新阈值
                        // θ_l = θ_l + η g_l
                        for (int l = 1; l < LayersCount; ++l)
                            _thresholds[l] -= learningRate * grad[l];
                    }
                }
            }

            /// <summary>
            /// 用批量梯度下降（累积误差逆传播）算法训练神经网络
            /// </summary>
            public void BGD(IEnumerable<Sample> samples, double learningRate, int epoch) {
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }

                for (int t = 0; t < epoch; ++t) {
                    Matrix<double>[] sumForWeights = new Matrix<double>[LayersCount - 1];
                    for (int l = 0; l < LayersCount - 1; ++l)
                        sumForWeights[l] = mb.Dense(_neuronsCnt[l + 1], _neuronsCnt[l], 0.0);
                    Vector<double>[] sumForThreshold = new Vector<double>[LayersCount];
                    for (int l = 1; l < LayersCount; ++l)
                        sumForThreshold[l] = vb.Dense(_neuronsCnt[l], 0.0);
                    foreach (var s in samples) {
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        Vector<double>[] grad = GradOfEachLayer(s, outputs);
                        // 累加上层梯度和输出的积，供更新权重用
                        for (int l = 0; l < LayersCount - 1; ++l)
                            sumForWeights[l] += grad[l + 1].ToColumnMatrix() * outputs[l].ToRowMatrix();
                        // 累加梯度，供更新阈值用
                        for (int l = 1; l < LayersCount; ++l)
                            sumForThreshold[l] += grad[l];
                    }
                    // 利用累加的梯度和输出的积，更新权重
                    for (int l = 0; l < LayersCount - 1; ++l)
                        _weights[l] += sumForWeights[l] * learningRate / samples.Count();
                    // 利用累加的梯度，更新阈值
                    for (int l = 1; l < LayersCount; ++l)
                        _thresholds[l] -= sumForThreshold[l] * learningRate / samples.Count();
                }
            }

            /// <summary>
            /// 讲向量输入当前神经网络计算一个输出
            /// </summary>
            public IList<double> Evaluate(IList<double> input) {
                if (input.Count != InputDim)
                    throw new ArgumentException("输入向量维数和输入层神经元数不一致。", nameof(input));
                Vector<double> layerInput = vb.Dense(input.ToArray()), layerOutput;
                for (int l = 0; l < LayersCount - 1; ++l) {
                    layerOutput = ApplyToVec(Sigmoid, _weights[l] * layerInput - _thresholds[l + 1]);
                    layerInput = layerOutput;
                }
                return layerInput;
            }

            /// <summary>
            /// 与 Evaluate 方法不同的是，该方法讲记录每一层生成的输出，并返回之
            /// </summary>
            public Vector<double>[] EvaluateAndGetEachLayersOutput(IList<double> input) {
                if (input.Count != InputDim)
                    throw new ArgumentException("输入向量维数和输入层神经元数不一致。", nameof(input));
                Vector<double>[] layerOutputs = new Vector<double>[LayersCount];
                layerOutputs[0] = vb.Dense(input.ToArray());
                for (int l = 1; l < LayersCount; ++l)
                    layerOutputs[l] = vb.Dense(_neuronsCnt[l]);
                for (int l = 0; l < LayersCount - 1; ++l)
                    layerOutputs[l + 1] = ApplyToVec(Sigmoid, _weights[l] * layerOutputs[l] - _thresholds[l + 1]);
                return layerOutputs;
            }

            /// <summary>
            /// Adam 算法
            /// </summary>
            public void Adam(IEnumerable<Sample> samples, double learningRate, double beta1, double beta2, int epoch, double epsilon = 1E-8) {
                if (beta1 < 0 || beta1 >= 1) throw new ArgumentException("系数β1不在[0,1)范围", nameof(beta1));
                if (beta2 < 0 || beta2 >= 1) throw new ArgumentException("系数β2不在[0,1)范围", nameof(beta2));
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }
                Matrix<double>[] meanOnW = new Matrix<double>[LayersCount - 1];
                Matrix<double>[] varianceOnW = new Matrix<double>[LayersCount - 1];
                Vector<double>[] meanOnTheta = new Vector<double>[LayersCount];
                Vector<double>[] varianceOnTheta = new Vector<double>[LayersCount]; 
                for (int l = 0; l < LayersCount - 1; ++l) {
                    meanOnW[l] = mb.Dense(_neuronsCnt[l + 1], _neuronsCnt[l], 0.0);
                    varianceOnW[l] = mb.Dense(_neuronsCnt[l + 1], _neuronsCnt[l], 0.0);
                }
                for (int l = 1; l < LayersCount; ++l) {
                    meanOnTheta[l] = vb.Dense(_neuronsCnt[l], 0.0);
                    varianceOnTheta[l] = vb.Dense(_neuronsCnt[l], 0.0);
                }

                for (int t = 1; t <= epoch; ++t) {  // 因为要以t为指数计算误差修正的两个beta参数所以从1开始，参考原文献
                    Vector<double>[] grad = new Vector<double>[LayersCount];
                    for (int l = 0; l < LayersCount; ++l)
                        grad[l] = vb.Dense(_neuronsCnt[l], 0.0);
                    foreach (var s in samples) {
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        for (int l = 0; l < LayersCount; ++l)
                            grad[l] += GradOfEachLayer(s, outputs)[l] / samples.Count();  // batch
                    }
                    double newLR = learningRate * Math.Sqrt(1.0 - Math.Pow(beta2, t)) / (1.0 - Math.Pow(beta1, t));
                    // Update weights
                    for (int l = 0; l < LayersCount - 1; ++l) {
                        for (int i = 0; i < _neuronsCnt[l]; ++i) {
                            for (int j = 0; j < _neuronsCnt[l + 1]; ++j) {
                                meanOnW[l][j, i] = beta1 * meanOnW[l][j, i] + (1.0 - beta1) * grad[l + 1][j];
                                varianceOnW[l][j, i] = beta2 * varianceOnW[l][j, i] + (1.0 - beta2) * grad[l + 1][j] * grad[l + 1][j];
                            }
                        }
                        for (int i = 0; i < _neuronsCnt[l]; ++i)
                            for (int j = 0; j < _neuronsCnt[l + 1]; ++j)
                                _weights[l][j, i] -= newLR * meanOnW[l][j, i] / (Math.Sqrt(varianceOnW[l][j, i]) + epsilon);
                    }
                    // Update thresholds
                    for (int l = 1; l < LayersCount; ++l) {
                        meanOnTheta[l] = beta1 * meanOnTheta[l] + (1.0 - beta1) * grad[l];
                        varianceOnTheta[l] = beta2 * varianceOnTheta[l] + (1.0 - beta2) * mb.Diagonal(grad[l].ToArray()) * grad[l];
                        _thresholds[l] -= newLR * mb.Diagonal(meanOnTheta[l].ToArray()) * vb.Dense(varianceOnTheta[l].Select(x => 1.0 / (Math.Sqrt(x) + epsilon)).ToArray());
                    }
                }
            }
        }
    }
}
