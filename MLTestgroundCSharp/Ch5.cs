using MathNet.Numerics.LinearAlgebra;

namespace MLEx {
    static class Ch5 {
        static readonly VectorBuilder<double> vb = Vector<double>.Build;
        public class BPNetwork {
            public BPNetwork(int hiddenLayers) {

            }
            public Vector<double> Output;
        }
    }
}
