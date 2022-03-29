using MathNet.Numerics.LinearAlgebra;

var vb = Vector<double>.Build;
var mb = Matrix<double>.Build;
List<Vector<double>> data = new() {
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
};
List<double> label = new() {
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
};

void Logistic() {
    Vector<double> dl(Vector<double> b, List<Vector<double>> xs, List<double> ys) {
        var sum = vb.Dense(xs[0].Count + 1, 0.0);
        for (int i = 0; i < xs.Count; ++i) {
            var xl = xs[i].ToList();
            xl.Add(1.0);
            var xHat = vb.Dense(xl.ToArray());
            var p1 = 1 / (1 + Math.Exp(-b * xHat));
            sum -= xHat * (ys[i] - p1);
        }
        return sum;
    }

    Matrix<double> d2l(Vector<double> b, List<Vector<double>> xs, List<double> ys) {
        var sum = mb.Dense(xs[0].Count + 1, xs[0].Count + 1, 0.0);
        for (int i = 0; i < xs.Count; ++i) {
            var xl = xs[i].ToList();
            xl.Add(1.0);
            var xHat = vb.Dense(xl.ToArray());
            var p1 = 1 / (1 + Math.Exp(-b * xHat));
            sum += xHat.ToColumnMatrix() * xHat.ToRowMatrix() * p1 * (1 - p1);
        }
        return sum;
    }

    Vector<double> beta = vb.Dense(new double[] { 0.0, 0.0, 0.0 });
    Console.WriteLine("对数几率回归，按T键使用当前迭代的参数进行测试，其他任意键进行下一轮牛顿法迭代\nβ0 = (0.0  0.0  0.0)");

    for (int i = 1; Console.ReadKey().Key != ConsoleKey.T; ++i) {
        beta -= d2l(beta, data, label).Inverse() * dl(beta, data, label);
        Console.WriteLine($"β{i} = ({beta[0]}  {beta[1]}  {beta[2]})");
    }

    while (Console.ReadKey().Key != ConsoleKey.Escape) {
        Console.Write("输入西瓜的密度和含糖率：");
        var t = (Console.ReadLine() ?? "").Split(' ', '\n');
        var x = vb.Dense(new double[] { double.Parse(t[0]), double.Parse(t[1]), 1.0 });
        var result = 1 / (1 + Math.Exp(-beta * x));
        Console.Write($"好瓜概率：{result}，判断为");
        if (result > 0.5) {
            Console.WriteLine("好瓜");
        } else {
            Console.WriteLine("非好瓜");
        }
    }
}

void LDA() {
    var mean = new Vector<double>[2] { vb.Dense(data[0].Count, 0.0), vb.Dense(data[0].Count, 0.0) };
    var cov = new Matrix<double>[2] { mb.Dense(data[0].Count, data[0].Count, 0.0), mb.Dense(data[0].Count, data[0].Count, 0.0) };
    int posCnt = 0, negCnt = 0;
    for (int i = 0; i < data.Count; ++i) {
        if (label[i] <= 0.5) ++negCnt; else ++posCnt;
    }
    for (int i = 0; i < data.Count; ++i) {
        mean[label[i] <= 0.5 ? 0 : 1] += data[i];
    }
    mean[0] /= negCnt;
    mean[1] /= posCnt;
    for (int i = 0; i < data.Count; ++i) {
        int t = label[i] <= 0.5 ? 0 : 1;
        cov[t] += (data[i] - mean[t]).ToColumnMatrix() * (data[i] - mean[t]).ToRowMatrix();
    }
    cov[0] /= negCnt;
    cov[1] /= posCnt;
    var sw = cov[0] + cov[1];
    // var sb = (mean[0] - mean[1]).ToColumnMatrix() * (mean[0] - mean[1]).ToRowMatrix();
    var w = sw.Inverse() * (mean[0] - mean[1]);

    Console.WriteLine($"LDA得到的投影向量w为：{w}");
    while (Console.ReadKey().Key != ConsoleKey.Escape) {
        Console.Write("输入西瓜的密度和含糖率：");
        var t = (Console.ReadLine() ?? "").Split(' ', '\n');
        var x = vb.Dense(new double[] { double.Parse(t[0]), double.Parse(t[1]) });
        var result = w*x < w*(mean[0] + mean[1])*0.5;
        Console.Write($"判断为");
        if (result) {
            Console.WriteLine("好瓜");
        } else {
            Console.WriteLine("非好瓜");
        }
    }
}
