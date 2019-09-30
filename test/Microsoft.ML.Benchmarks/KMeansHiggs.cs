using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Benchmarks
{
    class KMeansHiggs
    {
        public void TrainKMeansAndLR()
        {
            var _dataPath = "C:/Users/kumarpoo/Documents/data/HIGGS/HIGGS.csv";
            var _modelPath = "C:/Users/kumarpoo/Documents/data/HIGGS/higgsmodel.zip";

            string kmeans_cmd = @"train data=" + _dataPath +
                                " loader=TextLoader{sep=, col=Label:R4:0 col=Features:R4:1-28}" +
                                " trainer=km{k=25 maxiter=5}" +
                                " out=" + _modelPath;

            var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, ITransformer, KMeansTrainer, KMeansModelParameters>();
            kmeans_cmd.ExecuteMamlCommand(environment);
        }
    }
}
