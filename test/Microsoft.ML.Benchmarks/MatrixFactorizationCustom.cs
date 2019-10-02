using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    class MatrixFactorizationCustom
    {
        static readonly string _modelPath = "C:/Users/kumarpoo/Documents/data/MovieRatings.zip";
        public void TrainMatrixFactorizationModel()
        {
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column("UserId", DataKind.UInt32, 0),
                    new TextLoader.Column("MovieId", DataKind.Int32, 1),
                    new TextLoader.Column("Rating", DataKind.Single, 2),
                    new TextLoader.Column("Timestamp", DataKind.Int64, 3)
                },
                HasHeader = true,
                Separators = new char[] { ',' }
            });

            var data = loader.Load("C:/Users/kumarpoo/Documents/data/Movie Ratings.csv");
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(new InputOutputColumnPair[] { new InputOutputColumnPair("UserId"), new InputOutputColumnPair("MovieId") })
            .Append(mlContext.Recommendation().Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options()
            {
                MatrixColumnIndexColumnName = "UserId",
                MatrixRowIndexColumnName = "MovieId",
                LabelColumnName = "Rating"
            }));


            var model = pipeline.Fit(data);


            //Save model into file
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, data.Schema, fileStream);
            }

            Console.WriteLine("Saved model");
        }
    }
}
