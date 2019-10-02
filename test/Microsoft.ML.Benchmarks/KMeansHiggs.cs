using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Benchmarks
{
    class KMeansHiggs
    {
        static readonly string _dataPath = "C:/Users/kumarpoo/Documents/data/HIGGS/HIGGS.csv";
        

        public void TrainKMeansAndLRMaml()
        {
            var _modelPath = "C:/Users/kumarpoo/Documents/data/HIGGS/higgsmodelmaml.zip";
            string kmeans_cmd = @"train data=" + _dataPath +
                                " loader=TextLoader{sep=, col=Label:R4:0 col=Features:R4:1-28}" +
                                " trainer=km{k=25 maxiter=5}" +
                                " out=" + _modelPath;

            var environment = EnvironmentFactory.CreateClassificationEnvironment<TextLoader, ITransformer, KMeansTrainer, KMeansModelParameters>();
            kmeans_cmd.ExecuteMamlCommand(environment);
        }

        public void TrainKMeansAndLR()
        {
            var _modelPath = "C:/Users/kumarpoo/Documents/data/HIGGS/higgsmodel.zip";
            MLContext mLContext = new MLContext();
            var loader = mLContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Separators = new char[] { ',' },
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("Features", DataKind.Single, new [] {
                        new TextLoader.Range(){ Min = 1, Max = 27 } }),
                }
            });

            var options = new KMeansTrainer.Options
            {
                NumberOfClusters = 25,
                MaximumNumberOfIterations = 5,
                FeatureColumnName = "Features",
            };

            Console.WriteLine("Reading file");
            var data = loader.Load(_dataPath);
            var estimatorPipeline = mLContext.Clustering.Trainers.KMeans(options);

            Console.WriteLine("Training started");
            var model = estimatorPipeline.Fit(data);

            Console.WriteLine("Starting to save file");
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mLContext.Model.Save(model, data.Schema, fileStream);
            }

            Console.WriteLine("Saved model");
        }
    }
}
    