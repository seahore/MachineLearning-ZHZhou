namespace MLEx {
    static class Ch5 {
        public struct Sample {
            public IList<double> Input;
            public IList<double> ExpectedOutput;
            public Sample(IList<double> i, IList<double> o) {
                Input = i;
                ExpectedOutput = o;
            }

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
            public int OutputDim => _neuronsCnt.Last();
            /// <summary>
            /// 每层的神经元个数
            /// </summary>
            public int[] NeuronsCount => _neuronsCnt;
            int[] _neuronsCnt;
            /// <summary>
            /// weights[l][i][j]: 第 l 层第 i 个神经元向下一层第 j 个神经元输入的权值
            /// </summary>
            public double[][][] Weights => _weights;
            double[][][] _weights;
            /// <summary>
            /// thresholds[l][i]: 第 l 层第 i 个神经元的阈值
            /// </summary>
            public double[][] Thresholds => _thresholds;
            double[][] _thresholds;

            double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
            double dSigmoid(double x) => Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2);

            public BPNetwork(int[] neurons, double[][][]? w = null, double[][]? th = null) {
                _rand = new Random();
                _neuronsCnt = neurons;
                if (w is not null) _weights = w;
                else {
                    _weights = new double[LayersCount - 1][][];
                    for (int l = 0; l < LayersCount - 1; ++l) {
                        _weights[l] = new double[_neuronsCnt[l]][];
                        for (int i = 0; i < _neuronsCnt[l]; ++i) {
                            _weights[l][i] = new double[_neuronsCnt[l + 1]];
                            for (int j = 0; j < _neuronsCnt[l + 1]; j++) {
                                _weights[l][i][j] = _rand.NextDouble();
                            }
                        }
                    }
                }
                if (th is not null) _thresholds = th;
                else {
                    _thresholds = new double[LayersCount][];
                    for (int l = 1; l < LayersCount; ++l) {
                        _thresholds[l] = new double[_neuronsCnt[l]];
                        for (int i = 0; i < _neuronsCnt[l]; i++) {
                            _thresholds[l][i] = _rand.NextDouble();
                        }
                    }
                }
            }

            /// <summary>
            /// 用误差逆传播算法训练神经网络
            /// </summary>
            public void BP(IEnumerable<Sample> samples, double learningRate, int epoch) {
                foreach (var s in samples) {
                    if (s.Input.Count != InputDim) throw new ArgumentException("某个样本的输入向量维数和输入层神经元数不一致。", nameof(samples));
                    if (s.ExpectedOutput.Count != OutputDim) throw new ArgumentException("某个样本的预期输出向量的维数和输出层神经元数不一致。", nameof(samples));
                }
                for (int e = 0; e < epoch; ++e) {
                    foreach (var s in samples) {
                        // 获取每层输出，当然也包含最后的输出
                        var outputs = EvaluateAndGetEachLayersOutput(s.Input);
                        double[][] grad = new double[LayersCount][];
                        for (int l = 0; l < LayersCount; ++l) {
                            grad[l] = new double[_neuronsCnt[l]];
                        }
                        // 计算输出层的梯度
                        for (int i = 0; i < OutputDim; ++i) {
                            grad.Last()[i] = outputs.Last()[i] * (1 - outputs.Last()[i]) * (s.ExpectedOutput[i] - outputs.Last()[i]);
                        }
                        // 依次计算输出以下层的梯度
                        for (int l = LayersCount - 2; l>=0; --l) {
                            for (int i = 0; i < _neuronsCnt[l]; ++i) {
                                double sum = 0;
                                for (int j = 0; j < _neuronsCnt[l + 1]; ++j) {
                                    sum += _weights[l][i][j] * grad[l + 1][j];
                                }
                                grad[l][i] = outputs[l][i] * (1 - outputs[l][i]) * sum;
                            }
                        }
                        // 更新权重
                        for (int l = 0; l < LayersCount-1; ++l) {
                            for (int i = 0; i < _neuronsCnt[l]; ++i) {
                                for (int j = 0; j < _neuronsCnt[l+1]; ++j) {
                                    _weights[l][i][j] += learningRate * grad[l + 1][j] * outputs[l][i];
                                }
                            }
                        }
                        // 更新阈值
                        for (int l = 1; l < LayersCount; ++l) {
                            for (int i = 0; i < _neuronsCnt[l]; ++i) {
                                _thresholds[l][i] -= learningRate * grad[l][i];
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// 讲向量输入当前神经网络计算一个输出
            /// </summary>
            public IList<double> Evaluate(IList<double> input) {
                if (input.Count != InputDim)
                    throw new ArgumentException("输入向量维数和输入层神经元数不一致。", nameof(input));
                double[] layerInput = input.ToArray(), layerOutput;
                for (int l = 0; l < LayersCount - 1; ++l) {
                    layerOutput = new double[_neuronsCnt[l + 1]];
                    for (int j = 0; j < _neuronsCnt[l + 1]; ++j) {
                        double x = 0.0;
                        for (int i = 0; i < _neuronsCnt[l]; ++i)
                            x += _weights[l][i][j] * layerInput[i];
                        x -= _thresholds[l + 1][j];
                        layerOutput[j] = Sigmoid(x);
                    }
                    layerInput = layerOutput;
                }
                return layerInput;
            }

            /// <summary>
            /// 与 Evaluate 方法不同的是，该方法讲记录每一层生成的输出，并返回之
            /// </summary>
            public double[][] EvaluateAndGetEachLayersOutput(IList<double> input) {
                if (input.Count != InputDim)
                    throw new ArgumentException("输入向量维数和输入层神经元数不一致。", nameof(input));
                double[][] layerOutputs = new double[LayersCount][];
                layerOutputs[0] = input.ToArray();
                for (int l = 1; l < LayersCount; ++l) {
                    layerOutputs[l] = new double[_neuronsCnt[l]];
                }
                for (int l = 0; l < LayersCount - 1; ++l) {
                    for (int j = 0; j < _neuronsCnt[l + 1]; ++j) {
                        double x = 0.0;
                        for (int i = 0; i < _neuronsCnt[l]; ++i)
                            x += _weights[l][i][j] * layerOutputs[l][i];
                        x -= _thresholds[l + 1][j];
                        layerOutputs[l + 1][j] = Sigmoid(x);
                    }
                }
                return layerOutputs;
            }
        }
    }
}
