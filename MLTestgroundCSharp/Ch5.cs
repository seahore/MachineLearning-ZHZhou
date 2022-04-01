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

            double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
            Vector<double> SigmoidVec(Vector<double> v) => vb.Dense(v.Select(x => Sigmoid(x)).ToArray());
            double dSigmoid(double x) => Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2);

            public BPNetwork(int[] neurons, Matrix<double>[]? w = null, Vector<double>[]? th = null) {
                _rand = new Random();
                _neuronsCnt = neurons;
                if (w is not null) _weights = w;
                else {
                    _weights = new Matrix<double>[LayersCount - 1];
                    for (int l = 0; l < LayersCount - 1; ++l) {
                        _weights[l] = mb.Dense(_neuronsCnt[l+1], _neuronsCnt[l]);
                        for (int i = 0; i < _neuronsCnt[l + 1]; ++i)
                            for (int j = 0; j < _neuronsCnt[l]; j++)
                                _weights[l][i,j] = _rand.NextDouble();
                    }
                }
                if (th is not null) _thresholds = th;
                else {
                    _thresholds = new Vector<double>[LayersCount];
                    for (int l = 1; l < LayersCount; ++l) {
                        _thresholds[l] = vb.Dense(_neuronsCnt[l]);
                        for (int i = 0; i < _neuronsCnt[l]; i++)
                            _thresholds[l][i] = _rand.NextDouble();
                    }
                }
            }

            /// <summary>
            /// 用标准误差逆传播算法训练神经网络
            /// </summary>
            public void BP(IEnumerable<Sample> samples, double learningRate, int epoch) {
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }
                for (int e = 0; e < epoch; ++e) {
                    samples = RandomSort(samples);
                    foreach (var s in samples) {
                        // 获取每层输出，当然也包含最后的输出
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        Vector<double>[] grad = new Vector<double>[LayersCount];
                        for (int l = 0; l < LayersCount; ++l)
                            grad[l] = vb.Dense(_neuronsCnt[l]);
                        // 计算输出层的梯度
                        // g_out = diag(yHat) diag(ones - yHat) (y - yHat)
                        grad[^1] = mb.Diagonal(outputs[^1].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[^1], 1.0) - outputs[^1]).ToArray()) * (vb.Dense(s.ExpectedOutput.ToArray()) - outputs[^1]);
                        for (int i = 0; i < OutputDim; ++i)
                            grad[^1][i] = outputs[^1][i] * (1 - outputs[^1][i]) * (s.ExpectedOutput[i] - outputs[^1][i]);
                        // 依次计算输出以下层的梯度
                        // g_l = diag(o_l) diag(ones - o_l) W^T g_{l+1}
                        for (int l = LayersCount - 2; l >= 0; --l)
                            grad[l] = mb.Diagonal(outputs[l].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[l], 1.0) - outputs[l]).ToArray()) * _weights[l].Transpose() * grad[l+1];
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
            /// 用累积误差逆传播算法训练神经网络
            /// </summary>
            public void AccumulatedBP(IEnumerable<Sample> samples, double learningRate, int epoch) {
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }

                for (int e = 0; e < epoch; ++e) {
                    Matrix<double>[] sumForWeights = new Matrix<double>[LayersCount-1];
                    for(int l = 0; l<LayersCount-1; ++l) 
                         sumForWeights[l] = mb.Dense(_neuronsCnt[l+1], _neuronsCnt[l], 0.0);
                    Vector<double>[] sumForThreshold = new Vector<double>[LayersCount];
                    for (int l = 1; l < LayersCount; ++l)
                        sumForThreshold[l] = vb.Dense(_neuronsCnt[l], 0.0);
                    foreach (var s in samples) {
                        Vector<double>[] grad = new Vector<double>[LayersCount];
                        for (int l = 0; l < LayersCount; ++l)
                            grad[l] = vb.Dense(_neuronsCnt[l]);
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        grad[^1] = mb.Diagonal(outputs[^1].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[^1], 1.0) - outputs[^1]).ToArray()) * (vb.Dense(s.ExpectedOutput.ToArray()) - outputs[^1]);
                        for (int l = LayersCount - 2; l >= 0; --l)
                            grad[l] = mb.Diagonal(outputs[l].ToArray()) * mb.Diagonal((vb.Dense(_neuronsCnt[l], 1.0) - outputs[l]).ToArray()) * _weights[l].Transpose() * grad[l + 1];
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
                    layerOutput = SigmoidVec(_weights[l] * layerInput - _thresholds[l + 1]);
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
                    layerOutputs[l + 1] = SigmoidVec(_weights[l] * layerOutputs[l] - _thresholds[l + 1]);
                return layerOutputs;
            }
        }
    }
}
